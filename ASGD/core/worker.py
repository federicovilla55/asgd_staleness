import logging
import time
from typing import Callable, Tuple
import numpy as np
import torch
import torch.nn as nn
from .parameter_server import ParameterServer
from .data_types import ParameterServerStatus
from ..config import ConfigParameters

def worker(
    w_id: int,
    server:  ParameterServer,
    model: Callable[[int], nn.Module],
    input_dim:  int,
    dataset_builder: Callable[[int,int,int], Tuple[torch.utils.data.DataLoader,int]],
    param: ConfigParameters,
    start_evt
) -> None:
    """
    Worker function for Stale Synchronous Parallel training.

    :param w_id: Worker ID.
    :type w_id: int
    :param server: Parameter server
    :type server: ParameterServer
    :param model: Model class to be trained.
    :type model: Callable[[int], nn.Module]
    :param input_dim: Input dimension of the model.
    :type input_dim: int
    :param dataset_builder: Function used to build the dataset.
    :type dataset_builder: Callable[[int,int,int], Tuple[torch.utils.data.DataLoader,int]]+
    :param param: SSP Configuration parameters.
    :type param: ConfigParameters
    :return: None
    """
    start_evt.wait() # Wait untill all workers are created so they start at the same time
    # Basic logging configuration
    logging.basicConfig(
        level=param.log_level,
        format=f"%(asctime)s [Worker-{w_id}] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Data loader from the dataset builder and the model parameters
    # Dataset contains an unique subset of data for each worker (changing `random_state` parameter)
    loader, _ = dataset_builder(param.num_workers, param.batch_size, w_id)

    # Create the model and loss function
    device = torch.device(param.device)
    model = model(input_dim).to(device)
    criterion = nn.MSELoss()
    tol = 1e-8
    data_it = iter(loader)  


    # Run the local updates and push updates to the server
    for step in range(param.local_steps):
        state, local_ver = server.pull() # Get the latest model parameters and version from the server
        with torch.no_grad():
            for p, s in zip(model.parameters(), state):
                p.copy_(s.to(device))

        # Get the next batch of data
        try:
            Xb, yb = next(data_it)
        # If the iterator is exhausted, reinitialize it
        except StopIteration:
            data_it = iter(loader)
            Xb, yb  = next(data_it)

        # Move data to the device
        Xb, yb = Xb.to(device), yb.to(device)

        # Forward pass
        out   = model(Xb)
        loss  = criterion(out, yb.float())
        loss.backward()
        if loss.item() < tol:
            #print(f" Worker {w_id} stopping at step {step}: loss {loss.item():.2e} < {tol:.0e}")
            break
        # Detach and move gradients to CPU
        grads = [p.grad.detach().cpu() for p in model.parameters()]
        for p in model.parameters():
            p.grad = None

        # simulate a continuous delay => Using random exponential distribution (so it mimics real case) 
        mean_stale = param.num_workers - 1
        time_scale = 1e-4
        delay = np.random.exponential(scale=mean_stale*time_scale)
        time.sleep(delay)

        # Compute gradients and push them to the server
        status : ParameterServerStatus = server.push(w_id, local_ver, grads)

        # If the update was accepted, it means the worker was too much stale
        if status == ParameterServerStatus.REJECTED:
            # Should we do something in this case?
            continue
        elif status == ParameterServerStatus.SHUTDOWN:
            # Server is shutting down and so should the worker
            break