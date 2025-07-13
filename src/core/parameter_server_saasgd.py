from .parameter_server import ParameterServer, ParameterServerStatus
from ..config import ConfigParameters
from torch import nn
import time
import torch

class ParameterServerSAASGD(ParameterServer):
    """
    Staleness-aware Async-SGD Implementation from the paper:
    "Staleness-aware Asynchronous SGD for Distributed Deep Learning" (https://arxiv.org/pdf/1511.05950).
    """

    def __init__(self, model: nn.Module, param: ConfigParameters):
        """
        Initialize the Parameter Server for ASAP-SGD.

        :param model: PyTorch model instance.
        :type model: nn.Module
        :param param: Configuration parameters.
        :type param: ConfigParameters
        """
        super().__init__(model, param)
        self.c = max(1, param.num_workers // param.staleness)
        self.accumulated_grads = [torch.zeros_like(p) for p in self.theta]
        self.accumulated_count = 0       


    def push(self, wid, w_version: int, grads: list[torch.Tensor]) -> ParameterServerStatus:
        server_start_push = time.perf_counter()
        with self._lock:
            current_ver = self._current_ver.value
            tau = current_ver - w_version
            if tau < 0:
                server_end_push = time.perf_counter()
                self.count_time_push += (server_end_push - server_start_push)
                return ParameterServerStatus.REJECTED

            # Record staleness
            self._staleness[wid].append(tau)
            if tau < len(self.hist):
                self.hist[tau] += 1
            else:
                pass

            # Scale gradients by alpha0 / tau (if tau > 0)
            alpha0 = self.param.lr
            scale = alpha0 / tau if tau != 0 else alpha0
            scaled_grads = [g * scale for g in grads]

            # Accumulate gradients
            for acc_g, s_g in zip(self.accumulated_grads, scaled_grads):
                acc_g.add_(s_g)
            self.accumulated_count += 1

            # Apply update if enough gradients accumulated
            if self.accumulated_count >= self.c:
                # Compute average gradient
                avg_grads = [acc_g / self.accumulated_count for acc_g in self.accumulated_grads]
                # Update parameters
                for theta, avg_g in zip(self.theta, avg_grads):
                    theta.sub_(avg_g)
                # Increment version
                self._current_ver.value += 1
                # Reset accumulators
                self.accumulated_grads = [torch.zeros_like(p) for p in self.theta]
                self.accumulated_count = 0

            server_end_push = time.perf_counter()
            self.count_time_push += (server_end_push - server_start_push)

            return ParameterServerStatus.ACCEPTED