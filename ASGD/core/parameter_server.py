import threading
import multiprocessing as mp
from collections import defaultdict
import numpy as np
import torch
from enum import Enum, auto
from .data_types import ParameterServerStatus
from .schedulers import LrScaler, ASAPScaler, PushResult

import collections
from typing import Any, Dict, List, Tuple

class ParameterServer:
    """
    Parameter Server for Stale Synchronous Parallel training. 
    The server manages the global model parameters and coordinates the gradient updates from multiple workers.
    Each worker computes gradients locally and with a `push` operation sends the result to the server, which aggregates the gradients and updates the model parameters.
    Each worker can receive the latest model parameters with a `pull` operation.
    
    Arguments:
    :param model: PyTorch model instance 
    :type model: nn.Module
    :param param: Configuration parameters
    :type param: ConfigParameters
    :type learning_rule: str
    """
    def __init__(self, model, param, learning_rule='DASGD'):
        self.param = param
        self.theta = [p.detach().share_memory_() for p in model.parameters()]
        self._current_ver = mp.Value("i", 0)
        self.prev_theta = [p.clone().detach() for p in self.theta]
        self._lock = threading.Lock()
        # one list of staleness values per worker for tracking staleness stats
        self._staleness = defaultdict(list)
        self.hist = [0] * (param.staleness +1) # We assume max staleness is 50, so easier data structure for F computation possible
        self.total = 0
        self.learning_rule = learning_rule

    def pull(self):
        return [p.clone() for p in self.theta], self._current_ver.value

    def push(self, wid, w_version: int, grads: list[torch.Tensor]) -> ParameterServerStatus:
        with self._lock:
            curr = self._current_ver.value
            st = curr - w_version
            # record staleness of each worker regardless of accept/reject
            self._staleness[wid].append(st)

            if self.learning_rule == "DASGD":
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

            else:

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

    def get_version(self):
        with self._lock:
            return self._current_ver.value
        
    def get_staleness_stats(self):
        """
        Returns a dict:
        {"per_worker": { wid: { "mean":…, "median":…, "std":…, "pct_over_bound":…}, …},"combined": {"mean":…,"median":…,"std":…,"pct_over_bound":…}}
        """
        per_worker = {}
        all_vals = []

        bound = self.param.staleness

        for wid, vals in self._staleness.items():
            arr = np.array(vals, dtype=float)
            if arr.size:
                mean   = float(arr.mean())
                median = float(np.median(arr))
                std    = float(arr.std())
                # compute fraction > bound
                over   = (arr > bound).sum()
                pct    = float(over) / arr.size * 100.0

                per_worker[wid] = {"mean":mean, "median": median, "std":std, "pct_over_bound": pct}
                all_vals.append(arr)
            else:
                per_worker[wid] = {"mean": None, "median": None, "std":None, "pct_over_bound": None}

        if all_vals:
            all_concat = np.concatenate(all_vals)
            combined = {"mean":float(all_concat.mean()), "median":float(np.median(all_concat)), "std":float(all_concat.std()), "pct_over_bound": float((all_concat > bound).sum()) / all_concat.size * 100.0}
        else:
            combined = {"mean":None, "median":None, "std":None, "pct_over_bound": None}

        return {"per_worker": per_worker, "combined": combined}