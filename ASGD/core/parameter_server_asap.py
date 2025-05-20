from .parameter_server import ParameterServer, ParameterServerStatus
from ..config import ConfigParameters
from torch import nn
import time
import torch

class ParameterServerASAP_SGD(ParameterServer):
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

            if st >= self.param.staleness:
                return ParameterServerStatus.REJECTED
            
            self.hist[st] += 1
            self.total += 1

            # empirical CDF of staleness up to (and including) this value => ASAP SGD implementation
            cum = sum(self.hist[: st+1])
            F = cum / self.total
            CA = 1 + self.param.Amplitude * (1 - 2 * F)
            scaled_lr = CA * self.param.lr

            # SGD update
            for p, g in zip(self.theta, grads):
                p.sub_(scaled_lr * g.to(p.device))

            server_end_push = time.perf_counter()    
            self.count_time_push += (server_end_push-server_start_push)
            self._current_ver.value += 1

        return ParameterServerStatus.ACCEPTED

