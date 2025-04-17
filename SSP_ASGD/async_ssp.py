import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Callable, Tuple
import torch
import torch.nn as nn
import numpy as np

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
    batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_level: int = logging.INFO

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
    criterion = nn.BCELoss() # Binary Cross Entropy Loss for binary classification

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
            while (local_ver - last_check) > param.staleness:
                last_check = server.get_version()
                state, g_ver = server.pull()
                with torch.no_grad():
                    for p, s in zip(model.parameters(), state):
                        p.copy_(s.to(device))
                local_ver = g_ver

            # Update local version and push the gradients to the server
            local_ver += 1
            server.push(w_id, local_ver, grads)

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
    ps = ParameterServer(init_model, param)

    # Create a process for each worker
    # Use either "fork" or "spawn" based on your OS ("fork" on Linux)
    ctx = mp.get_context("fork") # Context for multiprocessing
    procs = [] # List to hold the processes
    for id in range(param.num_workers):
        p = ctx.Process(
            target=worker, # Worker function
            args=(id, ps, model, input_dim, dataset_builder, param), # Arguments for the worker function
            daemon=False, # Daemon processes are not used as they are killed when the main process exits
        )
        p.start() # Start the worker process
        procs.append(p) # Append the process to the list

    for p in procs:
        p.join() # Wait for all processes to finish
        if p.exitcode != 0: # Check if the process exited with an error
            raise RuntimeError(f"Worker {p.name} crashed (exitcode {p.exitcode})")


    theta, _ = ps.pull() # Get the final parameter theta from the server

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
    
    model.eval()
    with torch.no_grad():
        # Create a model instance from the trained parameters
        predictions = model(torch.from_numpy(X_eval)).numpy() > 0.5
        accuracy = np.mean(predictions == y_eval)

        print(f"{name} Test accuracy: {accuracy:.4f}")

        return accuracy
