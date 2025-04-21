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

def split_data(X, y, val_size=0.01, test_size=0.2, random_state=None):
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
                     random_state=42):
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

# For linear dataset 

# full splits
X_tr, y_tr, X_val, y_val, X_te, y_te = load_linear_data(
    n_samples=101, n_features=110, noise=0.0,val_size=0.01,test_size=0.2, random_state=42 )

# single-worker loader
loader, dim = create_linear_data_loader(
    num_workers=1, batch_size=32, worker_id=0,
    n_samples=200, n_features=50, noise=0.0)



#X_tr_lin, y_tr_lin, X_val_lin, y_val_lin, X_te_lin, y_te_lin = lin_splits
X_tr_lin, y_tr_lin, X_val_lin, y_val_lin, X_te_lin, y_te_lin = X_tr, y_tr, X_val, y_val, X_te, y_te
X_comb = np.vstack([X_tr_lin, X_val_lin])
y_comb = np.concatenate([y_tr_lin, y_val_lin])
n, d = X_comb.shape
rng = np.random.RandomState(42)
scale = 5   # avoids huge outliers
# Amount of initializations
init_ws = rng.uniform(-scale, scale, size=(1, d))
np.save('linear_init_weights.npy', init_ws)

# 3) Compute 95% of max stable step size η₉₅
_, S_comb, _ = svd(X_comb, full_matrices=False)
eta_max = 2.0 / (S_comb[0]**2)
eta_95  = 0.95 * eta_max
def create_full_data_loader(num_workers: int,
                            batch_size:   int,
                            worker_id:    int
                           ) -> Tuple[DataLoader, int]:
    """
    Gives *every* worker the same full train+val set (X_comb, y_comb),
    but shuffle=True so each draws random mini‑batches independently.
    """
    # X_comb, y_comb come from your earlier split:
    #   X_comb = np.vstack([X_tr, X_val])
    #   y_comb = np.concatenate([y_tr, y_val])
    ds     = TensorDataset(
                torch.from_numpy(X_comb).float(),
                torch.from_numpy(y_comb).float()
             )
    loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False
             )
    return loader, X_comb.shape[1]

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
    local_steps: int = 500 
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_level: int = logging.INFO
    tol: float = 1e-8

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

    def __init__(self, model: nn.Module, param: ConfigParameters) -> None:
        self.param = param

        # Define a list with the global model parameters shared among all workers
        self.theta = [p.detach() for p in model.parameters()]
        for p in self.theta:
            p.share_memory_() # Parameters are stored in a shared memory among all workers

        # Lock and Multiprocessing condition variable for synchronization
        # For safe access access to shared resources
        self._lock = mp.Lock() 
        self._cv = mp.Condition(self._lock)

        # Shared integer for the current global version
        self._current_version = mp.Value("i", 0)

        # Shared array of integers for the current versions of each worker
        self._worker_versions = mp.Array("i", [0]*param.num_workers)

        # Multiprocessing Manager object to manage shared objects
        self.manager = mp.Manager()
        # Shared dictionary to store pending gradients pushed by workers
        self._pending = self.manager.dict()

    def pull(self) -> Tuple[list[torch.Tensor], int]:
        """
        Fetch the current model parameters and the current version from the server.
        This method is called by the workers to get the latest model parameters before starting their local training or when they need to synchronize with the server.
        
        :return: A tuple containing the current model parameters and the current version.
        :rtype: Tuple[list[torch.Tensor], int]
        """
        with self._lock:
            return [p.clone() for p in self.theta], self._current_version.value

    def push(self, wid: int, version: int, grads: list[torch.Tensor]) -> None:
        """
        Push the gradients computed by a worker to the server for later aggregation.
        This method is called by the workers after they have computed gradients locally.
        If all workers have pushed their gradients for the same version, the server will aggregate them and update the model parameters.
        The server will notify all workers when the aggregation is complete.

        :param wid: Worker ID of the worker pushing the gradients.
        :type wid: int
        :param version: Current version of the model parameters.
        :type version: int
        :param grads: List of gradients computed by the worker.
        :type grads: list[torch.Tensor]
        :return: None
        """

        with self._lock:
            grads_np = [grad.cpu().numpy() for grad in grads] # Convert gradients to numpy arrays

            # Store the gradients in the shared dictionary of pending gradient updates
            key = (version, wid) # the key is a tuple of the update version and worker ID
            self._pending[key] = grads_np 

            # Check if all workers have pushed their gradients for this version
            if sum(1 for key in self._pending.keys() if key[0] == version) == self.param.num_workers:
                
                # Aggregate all the gradients the worker pushed for this version
                aggregated_grad = []
                for i in range(len(grads)):
                    grad_list = [self._pending[(version, id)][i] for id in range(self.param.num_workers)]
                    avg_grad = np.mean(grad_list, axis=0)
                    aggregated_grad.append(torch.from_numpy(avg_grad).to(device=self.theta[i].device))

                # Update the model parameters with the aggregated gradients
                for idx, avg_grad in enumerate(aggregated_grad):
                    self.theta[idx].sub_(self.param.lr * avg_grad)
                
                # Remove the pending gradients for this version from the shared dictionary
                for w in range(self.param.num_workers):
                    del self._pending[(version, w)]

                # Notify all workers that the global value for this version is computed
                self._current_version.value = version
                self._cv.notify_all()

    def get_version(self) -> int:
        """
        Get current global version.
        
        :return: The current version of the model parameters.
        :rtype: int
        """

        # A lock can be avoided as in worst case a worker will get a previous value and 
        # therefore wait (but even with the lock a process will wait).
        # with self._lock:
        return self._current_version.value

def worker(
    w_id: int,
    server:  ParameterServer,
    model: Callable[[int], nn.Module],
    input_dim:  int,
    dataset_builder: Callable[[int,int,int], Tuple[torch.utils.data.DataLoader,int]],
    param: ConfigParameters
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
    #criterion = nn.BCELoss() # Binary Cross Entropy Loss for binary classification
    criterion = nn.MSELoss()

    # Initialize the model parameters by retrieving the global model parameters from the server
    state, version = server.pull()
    with torch.no_grad():
        for p, s in zip(model.parameters(), state):
            p.copy_(s.to(device))
    
    # Define the local version and last checked remote version for staleness
    local_ver = version
    last_check = version    

    # Run the local updates and push updates to the server
    for step in range(param.local_steps):
        totall_loss = 0
        num_batches = 0
        # Each step is a full loop over the loaded portion of dataset
        for X_batch, y_batch in loader:
            # For each mini-batch from the worker's own data subset:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            model.train()
            output = model(X_batch)
            loss = criterion(output, y_batch.float())
            loss.backward()
            
            # Collect gradients and store them on CPU tensors
            grads = [p.grad.detach().cpu() for p in model.parameters()]
            for p in model.parameters():
                p.grad = None # Clear gradients to avoid accumulation in the next step
            
            # Obey to the staleness constraint by syncronizing with the server version and 
            # updating the local model parameters if needed.
            
            while (local_ver - last_check) > param.staleness: # Get central parameters after param.staleness updates or less
            #while True:
                last_check = server.get_version()
                state, g_ver = server.pull()
                with torch.no_grad():
                    for p, s in zip(model.parameters(), state):
                        p.copy_(s.to(device))
                local_ver = g_ver
            

            # Update local version and push the gradients to the server
            local_ver += 1
            server.push(w_id, local_ver, grads)
            num_batches += 1
            totall_loss += loss.item()

        avg_loss = totall_loss / num_batches
        # EARLY‐STOP check:
        if avg_loss < param.tol:
            logging.info(
                f"[Worker {w_id}] early stopping at step {step}, "
                f"loss={loss.item():.6e} < tol={param.tol:.0e}"
            )
            return


# 1) Tell the manager how to create a ParameterServer proxy
class PSManager(BaseManager): pass
PSManager.register('ParameterServer', ParameterServer)

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
    for id in range(param.num_workers):
        p = ctx.Process(
            target=worker, # Worker function
            args=(id, ps_proxy, model, input_dim, dataset_builder, param), # Arguments for the worker function
            daemon=False, # Daemon processes are not used as they are killed when the main process exits
        )
        p.start() # Start the worker process
        procs.append(p) # Append the process to the list

    for p in procs:
        p.join() # Wait for all processes to finish
        if p.exitcode != 0: # Check if the process exited with an error
            raise RuntimeError(f"Worker {p.name} crashed (exitcode {p.exitcode})")


    theta, _ = ps_proxy.pull() # Get the final parameter theta from the server

    #print("Final Version: ", ps.get_version())
    #logging.info("SSP training finished")

    return theta, input_dim

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

    print(f"{name} Test MSE = {mse:.6f}")
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
    


def sgd_training(num_epochs = 10000, criterion = nn.MSELoss(), batch_size = 32, lr = eta_95, tol=1e-8):
    X_train, y_train = X_comb, y_comb 

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
        # Print every 1000 epochs
        if epoch % 1000 == 0:
            print(f"[Epoch {epoch:5d}] Avg Loss = {avg_loss:.6f}")

        # Early stopping
        if avg_loss < tol:
            print(f"Stopping early at epoch {epoch} with avg loss {avg_loss:.6f} < tol={tol}")
            break

    return model

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Set up the configuration for the SSP training
    params_ssp = ConfigParameters(
        num_workers = 3,
        staleness = 10,
        lr = eta_95,
        local_steps = 7000,
        batch_size = 32,
        device = "cuda" if torch.cuda.is_available() else "cpu",
        log_level = logging.DEBUG,
        tol = 1e-8,
    )

    # Dataset builder function
    dataset_builder = create_full_data_loader
    # Model class
    model = LinearNetModel
    
    #Run the baseline
    # run baseline for comparison
    print("start baseline training")
    start = time.perf_counter()
    sgd_model = sgd_training()
    end = time.perf_counter()
    sgd_time = end-start
    print("Baseline part is done")

    # Run the SSP training and measure the time taken
    print("Start ASGD training")
    start = time.perf_counter()
    asgd_params, dim = run_ssp_training(dataset_builder, model, params_ssp)
    end = time.perf_counter()
    asgd_time = end - start

    # Evaluate the trained model on the test set
    #_, X_test, _, y_test = load_adult_data()
    asgd_model = build_model(asgd_params, model, dim)

    evaluate_model("ASGD", asgd_model, X_te_lin, y_te_lin)

    evaluate_model("SGD", sgd_model, X_te_lin, y_te_lin)

    print(f"Time Comparison: ASGD {asgd_time:2f} sec;\tSGD {sgd_time:2f} sec")

if __name__ == "__main__":
    main()