import threading
import multiprocessing as mp
from collections import defaultdict
import numpy as np
import torch
from enum import Enum, auto
from .data_types import ParameterServerStatus
from .schedulers import LrScaler, ASAPScaler, PushResult

from .parameter_server import ParameterServer

import collections
from typing import Any, Dict, List, Tuple

class ParameterServerASAP(ParameterServer):
    """
    Parameter Server for Stale Synchronous Parallel training with ASAP SGD.
    
    Arguments:
    :param model: PyTorch model instance 
    :type model: nn.Module
    :param param: Configuration parameters
    :type param: ConfigParameters
    """
    def __init__(self, model, param):
        super().__init__(model, param)

        # one list of staleness values per worker for tracking staleness stats
        self.hist = [0] * (param.staleness +1) # We assume max staleness is 50, so easier data structure for F computation possible
        self.total = 0

    def push(self, wid, w_version: int, grads: list[torch.Tensor]) -> ParameterServerStatus:
        with self._lock:
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

        self._current_ver.value += 1
        return ParameterServerStatus.ACCEPTED
