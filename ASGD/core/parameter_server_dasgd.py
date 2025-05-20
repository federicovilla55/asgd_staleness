from .parameter_server import ParameterServer, ParameterServerStatus
from ..config import ConfigParameters
from torch import nn
import time
import torch

class ParameterServerDASGD(ParameterServer):
    def __init__(self, model: nn.Module, param: ConfigParameters):
        """
        Initialize the Parameter Server for ASAP-SGD.

        :param model: PyTorch model instance.
        :type model: nn.Module
        :param param: Configuration parameters.
        :type param: ConfigParameters
        """
        super().__init__(model, param)

    def push(self, wid, w_version: int, grads: list[torch.Tensor]) -> ParameterServerStatus:
        with self._lock:
            server_start_push = time.perf_counter()

            curr = self._current_ver.value
            st = curr - w_version
            # record staleness of each worker regardless of accept/reject
            self._staleness[wid].append(st)

            tau = st
            k = self.param.num_workers

            # Store current theta before updating
            for i, p in enumerate(self.theta):
                self.prev_theta[i].data.copy_(p.data)

            # ASGD update with dynamic staleness (DASGD) 
            # (see : https://doi.org/10.1016/j.ins.2024.121220)
            for i, (p, g) in enumerate(zip(self.theta, grads)):

                delta_W = p - self.prev_theta[i]
                denom = (tau + k)

                dynamic_bias = (-tau / denom) * delta_W
                dynamic_scale = (k / denom)

                p.sub_(dynamic_bias + dynamic_scale * self.param.lr * g.to(p.device))

            server_end_push = time.perf_counter()    
            self.count_time_push += (server_end_push-server_start_push)
            self._current_ver.value += 1

        return ParameterServerStatus.ACCEPTED

