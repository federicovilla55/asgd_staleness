import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time
from getpass import getpass
from tqdm.notebook import tqdm
from numpy.linalg import svd
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Callable, Tuple
from multiprocessing.managers import BaseManager
from collections import defaultdict
from enum import Enum
import threading
import random
import pickle
import os
from scipy.stats import ttest_rel

class ParameterServerStatus(Enum):
    """
    Enum for the status of the parameter server.
    """
    ACCEPTED = 0
    REJECTED = 1
    SHUTDOWN = 2

def create_linear_dataset(n_samples=100,
                          n_features=110,
                          noise=0.0,
                          random_state=None):
    """
    Overparameterized linear regression dataset:
      - X sampled U(-3, 3)
      - y = X @ w_true + noise
    """
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-3, 3, size=(n_samples, n_features))
    w_true = rng.randn(n_features)
    y = X.dot(w_true) + noise * rng.randn(n_samples)
    return X.astype(np.float32), y.astype(np.float32)

def create_poly_varied_dataset(n_samples=100,
                               n_features=110,
                               max_degree=4,
                               noise=0.0,
                               random_state=None):
    """
    Overparameterized nonlinear regression dataset:
      - X sampled U(-3, 3)
      - Each feature i raised to its own degree_i ∈ [1, max_degree]
      - y = sum_i w_true[i] * (X[:, i] ** degree_i) + noise
    Returns:
      X_raw, y, degrees
    """
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-3, 3, size=(n_samples, n_features))
    w_true = rng.randn(n_features)
    degrees = rng.randint(1, max_degree + 1, size=n_features)
    X_pow = np.stack([X[:, i] ** deg for i, deg in enumerate(degrees)], axis=1)
    y = X_pow.dot(w_true) + noise * rng.randn(n_samples)
    return X.astype(np.float32), y.astype(np.float32), degrees

def split_data(X, y, val_size=0.0, test_size=0.2, random_state=None):
    """
    Splits (X, y) into train/val/test.
      - train: (1 - val_size - test_size)
      - val: val_size
      - test: test_size
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    val_rel = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_rel, random_state=random_state)
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_linear_data(n_samples=100,
                     n_features=110,
                     noise=0.0,
                     val_size=0.01,
                     test_size=0.2,
                     random_state=None):
    """
    Generate a linear overparam dataset and split it.
    Returns: X_train, y_train, X_val, y_val, X_test, y_test
    """
    X, y = create_linear_dataset(n_samples, n_features, noise, random_state)
    return split_data(X, y, val_size, test_size, random_state)

def load_poly_varied_data(n_samples=100,
                          n_features=110,
                          max_degree=4,
                          noise=0.0,
                          val_size=0.2,
                          test_size=0.2,
                          random_state=42):
    """
    Generate a polynomial-varied dataset, split it, and also return degrees.
    Returns: (X_train, y_train, X_val, y_val, X_test, y_test, degrees)
    """
    X, y, degrees = create_poly_varied_dataset(
        n_samples, n_features, max_degree, noise, random_state)
    splits = split_data(X, y, val_size, test_size, random_state)
    return (*splits, degrees)

def create_linear_data_loader(num_workers,
                              batch_size,
                              worker_id,
                              n_samples=100,
                              n_features=110,
                              noise=0.0,
                              val_size=0.01,
                              test_size=0.2,
                              random_state=42):
    """
    Return a DataLoader for a shard of the linear training set.
    Also returns the input dimension.
    """
    X_train, y_train, _, _, _, _ = load_linear_data(
        n_samples, n_features, noise, val_size, test_size, random_state) #Give full dataset to each worker
    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loader, X_train.shape[1]

def create_poly_varied_data_loader(num_workers,
                                   batch_size,
                                   worker_id,
                                   n_samples=100,
                                   n_features=110,
                                   max_degree=4,
                                   noise=0.0,
                                   val_size=0.2,
                                   test_size=0.2,
                                   random_state=42):
    """
    Return a DataLoader for a shard of the poly-varied training set.
    Also returns the input dimension (same as n_features).
    """
    X_train, y_train, _, _, _, _, degrees = load_poly_varied_data(
        n_samples, n_features, max_degree, noise, val_size, test_size, random_state + worker_id)
    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loader, X_train.shape[1], degrees



@dataclass
class ConfigParameters:
    """
    Configuration for Stale Synchronous Parallel training for Asynchronous SGD (SSP-ASGD).

    :param num_workers: Number of worker processes.
    :type num_workers: int
    :param staleness: Staleness bound allowed for the workers during training. Represents the maximum number of versions a worker can be behind the latest version.
    :type staleness: int
    :param lr: Learning rate for the model. Represents the step size for updating the model parameters.
    :type lr: float
    :param local_steps: Number of steps/updates each worker locally computes before pushing gradients to the server.
    :type local_steps: int
    :param batch_size: Batch size for each training step and the data loader.
    :type batch_size: int
    :param device: Device to use for training (e.g., "cuda" or "cpu").
    :type device: str
    :param log_level: Logging verbosity level.
    :type log_level: int
    """
    num_workers: int = 4
    staleness: int = 2
    lr: float  = 0.01
    local_steps: int = 1 
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_level: int = logging.INFO
    tol: float = 1e-8
    Amplitude: float = 1  # The maximum amplitude deviation from the base step size

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
        # one list of staleness values per worker for tracking staleness stats
        self._staleness = defaultdict(list)
        self.hist = [0] * (param.staleness +1) # We assume max staleness is 50, so easier data structure for F computation possible
        self.total = 0

    def pull(self):
        return [p.clone() for p in self.theta], self._current_ver.value

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


# 1) Tell the manager how to create a ParameterServer proxy
class PSManager(BaseManager): pass
PSManager.register('ParameterServer', ParameterServer)
PSManager.register('get_staleness_stats', ParameterServer.get_staleness_stats)

def run_ssp_training(
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

    # Retrieve input dimension from dataset builder with provided batch size and number of workers
    _, input_dim = dataset_builder(param.num_workers, param.batch_size, 0)

    # Initialize the model and parameter server
    init_model = model(input_dim)
    # Start a custom manager server
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

    #print("Final Version: ", ps.get_version())
    #logging.info("SSP training finished")

    # Return the staleness stats for the workers
    stats    = ps_proxy.get_staleness_stats()
    return theta, input_dim, stats


def build_model(theta: list[torch.Tensor], model, input_dim: int) -> nn.Module:
    """
    Build a model instance from the provided parameters.

    :param theta: List of model parameters.
    :type theta: list[torch.Tensor]
    :param model_cls: Model class to be instantiated.
    :type model_cls: Callable[[int], nn.Module]
    :param input_dim: Input dimension of the model.
    :type input_dim: int
    :return: Model instance with the provided parameters.
    :rtype: nn.Module
    """
    model = model(input_dim)
    with torch.no_grad():
        for param, trained_param in zip(model.parameters(), theta):
            param.copy_(trained_param)
    return model

def evaluate_model(name:str, model: nn.Module, X_eval: np.ndarray, y_eval: np.ndarray) -> float:
    """
    Evaluate the model on the provided evaluation dataset.

    :param model: Model instance to be evaluated.
    :type model: nn.Module
    :param X_eval: Evaluation dataset features.
    :type X_eval: np.ndarray
    :param y_eval: Evaluation dataset labels.
    :type y_eval: np.ndarray
    :return: Accuracy of the model on the evaluation dataset.
    :rtype: float
    :raises ValueError: If the model is not in evaluation mode.
    """
    # ensure in eval mode
    model.eval()

    # Move data into torch tensors
    X_tensor = torch.from_numpy(X_eval).float()
    y_tensor = torch.from_numpy(y_eval).float()

    # For MSE, use the built‑in loss (mean reduction by default)
    criterion = nn.MSELoss()

    with torch.no_grad():
        # Forward pass
        y_pred = model(X_tensor)

        # If model outputs extra dims, flatten to match y_eval
        # e.g. y_pred = y_pred.view_as(y_tensor)

        # Compute MSE
        mse = criterion(y_pred, y_tensor).item()
    return mse
        

class LinearNetModel(nn.Module):
    """
    Simple Linear Regression Model

    :param input_dim: Number of model input features.
    :type input_dim: int
    :param bias: Whether to include a bias term in the linear layer.
    :type input_dim: int
    """
    def __init__(self, input_dim: int, bias: bool = True):
        super().__init__()
        # A single linear layer mapping input_dim features to a single output
        self.linear = nn.Linear(input_dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Applies a linear transformation without any activation, returning the raw output.

        :param x: Input tensor of shape (batch_size, input_dim).
        :type x: torch.Tensor
        :return: Output tensor of shape (batch_size,).
        :rtype: torch.Tensor
        """
        # Linear layer returns (batch_size, 1), so squeeze to (batch_size,)
        return self.linear(x).squeeze(-1)
    
def sgd_training(X_train, y_train, num_epochs = 10000, criterion = nn.MSELoss(), batch_size = 32, lr = 0.01, tol=1e-8):

    # Create a linear model with dimention equal to the number of features
    # in the dataset
    model   = LinearNetModel(X_train.shape[1])

    # Train the model using standard SGD
    loader  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
           torch.from_numpy(X_train), torch.from_numpy(y_train)
        ),
        batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_epoch_loss = 0.0
        num_batches = 0

        # Iterate over the batches of training data
        for Xb, yb in loader:
            optimizer.zero_grad() # Reset the gradients
            out  = model(Xb) # Forward pass
            loss = criterion(out, yb.float()) # Compute the loss
            loss.backward() # Backward pass
            optimizer.step() # Update the model parameters
            total_epoch_loss += loss.item() # Accumulate the loss
            num_batches += 1
        
        avg_loss = total_epoch_loss / num_batches
        # Early stopping
        if avg_loss < tol:
            print(f"Stopping early at epoch {epoch} with avg loss {avg_loss:.6f} < tol={tol}")
            break

    return model

class FullDataLoaderBuilder:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __call__(self, num_workers: int, batch_size: int, worker_id: int):
        ds     = TensorDataset(torch.from_numpy(self.X),
                               torch.from_numpy(self.y))
        loader = DataLoader(ds, batch_size=batch_size,
                            shuffle=True, drop_last=False)
        return loader, self.X.shape[1]
    
def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Fix the master seed so you always get the same “sub‑seeds”
    random.seed(1234)
    # Draw 100 integers in [0, 2^8)
    seeds = [random.randrange(2**8) for _ in range(100)]  # If you change the amount of seeds, the first n will still always be the same !

    # FILES FOR CHECKPOINTING
    sgd_losses_f = 'sgd_losses.pkl'
    asgd_losses_f = 'ASGD_first_losses.pkl'
    asgd_stats_f  = 'ASGD_first_stats.pkl'

    # get the directory this script lives in
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # then for each checkpoint file
    sgd_losses_file = os.path.join(script_dir, sgd_losses_f)
    asgd_losses_file = os.path.join(script_dir, asgd_losses_f)
    asgd_stats_file  = os.path.join(script_dir, asgd_stats_f)

    # AMOUNT OF SEEDS YOU WANT TO COMPUTE NOW
    RUNS_REGULAR_SGD = 1       # Set always min to 1 for both methods (if want to retrieve/use the stored values)
    RUNS_ASGD = 1

    if RUNS_REGULAR_SGD > 0:
        losses_file = sgd_losses_file
        if os.path.exists(losses_file):
            with open(losses_file, 'rb') as f:
                SGD_losses = pickle.load(f)
            logging.info(f"Resuming: {len(SGD_losses)}/{len(seeds)} seeds done")
        else:
            SGD_losses = []
            logging.info("Starting fresh, no existing losses file found")

        # Pick up where you left off
        start_idx = len(SGD_losses)
        for idx in range(start_idx, len(seeds)):
            seed = seeds[idx]
            
            if RUNS_REGULAR_SGD == 0:
                print("Performed the specified amount of runs for regular SGD")
                break
            RUNS_REGULAR_SGD = RUNS_REGULAR_SGD - 1

            # full splits => Always the same when using the same seed
            X_tr_lin, y_tr_lin, X_val_lin, y_val_lin, X_te_lin, y_te_lin = load_linear_data(n_samples=100, n_features=110, noise=0.0,val_size=0.01,test_size=0.2, random_state= seed)

            X_comb = np.vstack([X_tr_lin, X_val_lin])
            y_comb = np.concatenate([y_tr_lin, y_val_lin])

            # 3) Compute 95% of max stable step size η₉₅
            _, S_comb, _ = svd(X_comb, full_matrices=False)
            eta_max = 2.0 / (S_comb[0]**2)
            eta_95  = 0.95 * eta_max

            start = time.perf_counter()
            sgd_model = sgd_training(X_comb, y_comb, num_epochs = 10000, criterion = nn.MSELoss(), batch_size = 32, lr = eta_95, tol=1e-8)
            end = time.perf_counter()
            sgd_time = end-start

            SGD_loss = evaluate_model("SGD", sgd_model, X_te_lin, y_te_lin)

            SGD_losses.append(SGD_loss)

            print("Time Comparison for run:" + str(idx) + f":SGD {sgd_time:2f} sec")
        

        # SAVE THIS LIST
        with open(sgd_losses_file, 'wb') as f:
            pickle.dump(SGD_losses, f)

        with open(sgd_losses_file, 'rb') as f:
            SGD_losses = pickle.load(f)
        print("Retrieved regular SGD losses")

        avg_SGD_loss = sum(SGD_losses)/len(SGD_losses)
        print("Average SGD loss =" + str(avg_SGD_loss))

    if RUNS_ASGD > 0:
        # INIT/RETRIEVE LOSSES
        losses_file = asgd_losses_file
        if os.path.exists(losses_file):
            with open(losses_file, 'rb') as f:
                ASGD_losses = pickle.load(f)
            logging.info(f"Resuming: {len(ASGD_losses)}/{len(seeds)} seeds done")
        else:
            ASGD_losses = []
            logging.info("Starting fresh, no existing losses file found")

        # INIT/RETRIEVE WORKER STATS
        if os.path.exists(asgd_stats_file):
            with open(asgd_stats_file, 'rb') as f:
                ASGD_stats = pickle.load(f)
            logging.info(f"Resuming stats: {len(ASGD_stats)}/{len(seeds)} done")
        else:
            ASGD_stats = []
            logging.info("Starting fresh on stats")

        # Pick up where you left off
        start_idx = len(ASGD_losses)
        for idx in range(start_idx, len(seeds)):
            seed = seeds[idx]

            if RUNS_ASGD == 0:
                print("Performed the specified amount of runs for ASGD")
                break
            RUNS_ASGD = RUNS_ASGD - 1

            # full splits => Always the same when using the same seed
            X_tr_lin, y_tr_lin, X_val_lin, y_val_lin, X_te_lin, y_te_lin = load_linear_data(n_samples=100, n_features=110, noise=0.0,val_size=0.01,test_size=0.2, random_state= seed)

            X_comb = np.vstack([X_tr_lin, X_val_lin])
            y_comb = np.concatenate([y_tr_lin, y_val_lin])

            # 3) Compute 95% of max stable step size η₉₅
            _, S_comb, _ = svd(X_comb, full_matrices=False)
            eta_max = 2.0 / (S_comb[0]**2)
            eta_95  = 0.95 * eta_max
            
            # Dataset builder function
            dataset_builder = FullDataLoaderBuilder(X_comb, y_comb)
            # Model class
            model = LinearNetModel

            # Set up the configuration for the SSP training
            params_ssp = ConfigParameters(
                num_workers = 10,
                staleness = 50, 
                lr = eta_95/2,                          # HERE DIVIDED BY 2 SO THAT MAX LR = (1+A)*LR = ETA_95 => Otherwise very high test loss and bad convergence !!
                local_steps = 10000,
                batch_size = 32,
                device = "cuda" if torch.cuda.is_available() else "cpu",
                log_level = logging.DEBUG,
                tol = 1e-8,                             # The tol for workers is currently set at tol = 1e-8
                Amplitude = 1                           # The max amplitude deviation from the base stepsize
            )

            # Run the SSP training and measure the time taken
            start = time.perf_counter()
            asgd_params, dim, stats = run_ssp_training(dataset_builder, model, params_ssp)
            end = time.perf_counter()
            asgd_time = end - start
            ASGD_stats.append(stats)

            '''
            print(f"{'Worker':>6s}  {'Mean':>8s}  {'Median':>8s}  {'Std':>8s}  {'%Over':>8s}")
            print("-" * 45) 

            # Per-worker stats
            for wid, s in sorted(stats["per_worker"].items()):
                mean    = s["mean"]
                median  = s["median"]
                std     = s["std"]
                pct_over = s["pct_over_bound"]
                print(f"{wid:6d}  {mean:8.4f}  {median:8.4f}  {std:8.4f}  {pct_over:8.2f}")

            # Combined stats
            c = stats["combined"]
            print("\nCombined over all workers:")
            print(f"  Mean         = {c['mean']:.4f}")
            print(f"  Median       = {c['median']:.4f}")
            print(f"  Std          = {c['std']:.4f}")
            print(f"  % Over Bound = {c['pct_over_bound']:.2f}%")
            '''

            # Evaluate the trained model on the test set
            asgd_model = build_model(asgd_params, model, dim)

            ASGD_loss = evaluate_model("ASGD", asgd_model, X_te_lin, y_te_lin)

            ASGD_losses.append(ASGD_loss)

            print("Time Comparison for run:" + str(idx) + f": ASGD {asgd_time:2f} sec")

        # SAVE THE LOSSES
        with open(asgd_losses_file, 'wb') as f:
            pickle.dump(ASGD_losses, f)

        with open(asgd_losses_file, 'rb') as f:
            ASGD_losses = pickle.load(f)
        print("Retrieved ASGD losses")
        
        avg_ASGD_loss = sum(ASGD_losses)/len(ASGD_losses)

        print("Average ASGD loss =" + str(avg_ASGD_loss))

        #SAVE THE WORKER STATS
        with open(asgd_stats_file, 'wb') as f:
            pickle.dump(ASGD_stats, f)

        # If you want to inspect the stats you can do:
        # with open(stats_file, 'rb') as f:
        #     ASGD_stats = pickle.load(f)
        # now ASGD_stats is a list of dicts, each having
        #   stats["per_worker"] and stats["combined"]
    
    # COMPARE LOSSES FOR THE SEEDS THAT HAVE BEEN USED IN BOTH METHODS UNTIL NOW

    # Align lengths (in case one list is longer because of incomplete runs)
    n = min(len(SGD_losses), len(ASGD_losses))
    sgd_losses = SGD_losses[:n]
    asgd_losses = ASGD_losses[:n]

    # Compute difference: SGD_loss - ASGD_loss
    diffs = np.array(sgd_losses) - np.array(asgd_losses)

    # COMPUTE PAIRED T-TEST
    if n > 1:
        t_stat, p_value = ttest_rel(sgd_losses, asgd_losses, nan_policy='omit')

        print(f"Paired t-test over {n} runs:")
        print(f"  t-statistic = {t_stat:.4f}")
        print(f"  p-value     = {p_value:.4e}")

    # Summary statistics
    mean_diff = np.mean(diffs)
    median_diff = np.median(diffs)
    std_diff = np.std(diffs)

    print(f"Computed over {n} seeds:")
    print(f"Mean difference (SGD - ASGD): {mean_diff:.4e}")
    print(f"Median difference: {median_diff:.4e}")
    print(f"Std of difference: {std_diff:.4e}")

    # Plot histogram of differences
    plt.figure()
    plt.hist(diffs, bins=20, edgecolor='black')
    plt.axvline(mean_diff, color='red', linestyle='dashed', linewidth=1, label=f"Mean: {mean_diff:.2e}")
    plt.axvline(median_diff, color='blue', linestyle='dotted', linewidth=1, label=f"Median: {median_diff:.2e}")
    plt.xlabel("SGD_loss - ASGD_loss")
    plt.ylabel("Frequency")
    plt.title("Distribution of Loss Differences (SGD vs. ASGD)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()