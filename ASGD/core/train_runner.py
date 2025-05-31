from __future__ import annotations
import multiprocessing as mp
from types import SimpleNamespace
from typing import Tuple, Type,  Callable

import torch
import torch.nn as nn

from .worker import worker
from .parameter_server import ParameterServer
from ..config import ConfigParameters
from ..data.base import AbstractDataBuilder
from typing import Callable
from multiprocessing.managers import BaseManager

class PSManager(BaseManager):
    pass

# register a generic “factory” by name here, picklable because it's a top‐level function
def _ps_factory(model, cfg):
    # this will be monkey‐patched at runtime to point at the right class
    raise RuntimeError("_ps_factory was not replaced!")

PSManager.register("ParameterServer", callable=_ps_factory)
PSManager.register("get_staleness_stats", ParameterServer.get_staleness_stats)
PSManager.register("get_hist",              ParameterServer.get_hist)
PSManager.register("get_time_push",         ParameterServer.get_time_push)

def run_training(
    dataset_builder: Callable[[int, int,int], Tuple[torch.utils.data.DataLoader,int]],
    model: Callable[[int], nn.Module],
    param: ConfigParameters = ConfigParameters(),
) -> list[torch.Tensor]:
    """
    Helper function to run the Stale Synchronous Parallel training with the provided dataset builder, model and configuration parameters.

    :param dataset_builder: Function used to build the dataset.
    :param model: Model class to be trained.
    :param param: SSP Configuration parameters.
    :type param: ConfigParameters
    :return: The final model parameters after training.
    :rtype: list[torch.Tensor]
    """
     # — before starting the manager, swap out the factory function we registered —
    PSManager.register("ParameterServer", callable=parameter_server)
    
    # Retrieve input dimension from dataset builder with provided batch size and number of workers
    _, input_dim = dataset_builder(param.num_workers, param.batch_size, 0)

    # Initialize the model and parameter server
    init_model = model(input_dim)

    manager = PSManager()
    manager.start()
    ps_proxy = manager.ParameterServer(init_model, param)

    # Create a process for each worker
    # Use either "fork" or "spawn" based on your OS ("fork" on Linux)
    ctx = mp.get_context("spawn") # Context for multiprocessing
    procs = [] # List to hold the processes
    start_evt = ctx.Event() # Create event so that all workers start at the same time
    for id in range(param.num_workers):
        p = ctx.Process(
            target=worker, # Worker function
            args=(id, ps_proxy, model, input_dim, dataset_builder, param, start_evt), # Arguments for the worker function
            daemon=False, # Daemon processes are not used as they are killed when the main process exits
        )
        p.start() # Start the worker process
        procs.append(p) # Append the process to the list
    
    start_evt.set() # Start all the workers at the same time
    for p in procs:
        p.join() # Wait for all processes to finish
        if p.exitcode != 0: # Check if the process exited with an error
            raise RuntimeError(f"Worker {p.name} crashed (exitcode {p.exitcode})")


    theta, _ = ps_proxy.pull() # Get the final parameter theta from the server

    time_push = ps_proxy.get_time_push()
    print(f"Final time for all (pushes, pulls) = {time_push}")

    #print("Final Version: ", ps.get_version())
    #logging.info("SSP training finished")

    # Return the staleness stats for the workers
    stats    = ps_proxy.get_staleness_stats()

    # Return a list containing the staleness counts
    staleness_distr = ps_proxy.get_hist()

    return theta, input_dim, stats, staleness_distr
