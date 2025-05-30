import threading
import multiprocessing as mp
from collections import defaultdict
import numpy as np
import torch
from enum import Enum, auto
from .data_types import ParameterServerStatus
import time

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
        self.prev_theta = [p.clone().detach() for p in self.theta]
        self._lock = threading.Lock()
        # one list of staleness values per worker for tracking staleness stats
        self._staleness = defaultdict(list)
        # One list of the global staleness count
        self.hist = [0] * (param.staleness +1) # We assume max staleness is 50, so easier data structure for F computation possible
        self.total = 0
        self.count_time_push = 0
        self.count_time_pull = 0

    def pull(self):
        server_start_pull = time.perf_counter()
        result = [p.clone() for p in self.theta], self._current_ver.value
        server_end_pull = time.perf_counter()
        self.count_time_pull += (server_end_pull-server_start_pull)
        return result

    # Method not implemented
    def push(self, wid, w_version: int, grads: list[torch.Tensor]) -> ParameterServerStatus:
        return ParameterServerStatus.REJECTED

    def get_version(self):
        """Return the current version of the model parameters."""
        with self._lock:
            return self._current_ver.value
        
    def get_time_push(self):
        """Return the time spent in push and pull operations."""
        return (self.count_time_push, self.count_time_pull)
    
    def get_hist(self) -> list[int]:
        """Return the raw counts of staleness occurrences for this run."""
        # note: self.hist is of length staleness+1
        return list(self.hist)
        
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