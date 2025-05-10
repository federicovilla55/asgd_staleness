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
    """
    def __init__(self, model, param):
        self.param = param
        self.theta = [p.detach().share_memory_() for p in model.parameters()]
        self._current_ver = mp.Value("i", 0)
        self._lock = threading.Lock()
        self._staleness = defaultdict(list)

    def pull(self):
        return [p.clone() for p in self.theta], self._current_ver.value

    def push(self, wid, w_version: int, grads: list[torch.Tensor]) -> ParameterServerStatus:
        with self._lock:
            if w_version < self._current_ver.value - self.param.staleness:
                return ParameterServerStatus.REJECTED

            # SGD update
            for p, g in zip(self.theta, grads):
                p.sub_(self.param.lr * g.to(p.device))

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