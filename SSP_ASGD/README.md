## *Asynchronous Stochastic Gradient Descent* (*ASGD*) Implementation

*Stale Synchronous Parallel* (*SSP*) has explicit staleness, compared to other methods such as *Elastic Averaging SGD* (*EASGD*) where it is implicitly handled through the averaging mechanism.

This directory provides an implementation of a *Stale Synchronous Parallel* *ASGD* implementation. 

- `model.py` contains a simple linear network architecture composed of a simple binary classifier with one linear layer and sigmoid activation.
- `datasets.py` contains data loading and other preprocessing utilities for the model defined, used to create *PyTorch* `DataLoaders` for the *ASGD* workers. Used along the UCI Adult dataset for income prediction.
- `async_ssp.py` contains the implementation of the Stale Synchronous Parallel *ASGD*.
- `train_ssp.py` contains the script that executes both the SSP *ASGD* and compares it with the standard (*PyTorch* implemented) *SGD*. Comparison is based on both the accuracy and the time required.

To compute the comparison:
```bash
python train_ssp.py
```

### *EASGD* Averaging Mechanism

The method, used in *EASGD* to let the workers be independent of each other and at the same time keep them tethered to a central model.
Each worker continuously pulls a global value $\theta$ from a central server and uses it to update its local parameter $x_i$, so the parameters that the worker updates at every optimization step.
A worker $i$ updates its local parameter using a combination of the classic *SGD* algorithm and a value that keeps the model near the global value $\theta$, something similar to: $x_i = x_i - \gamma \cdot \nabla L(x_i) + \rho \cdot (\theta - x_i)$.
Where $\gamma$ is the gradient stepsize and $\rho$ is a parameter that measures how little the model can deviate from the global update.  
The latter sum in the update step should correct any drift that might happen due to stale updates.
Where each value $x_i$ and the global parameter $\theta$ are either simple floats or multi-dimensional vectors/matrices.
Each worker gets the global value $\theta$, computes its gradient update step ($x_i$) and then updates the global model $\theta = \theta + \rho \cdot (x_i - \theta)$.

### *SSP* Update Mechanism

With *Stale Synchronous Parallel* there is an explicit handling of each worker staleness: there is a staleness threshold $s$ which is a limit on how far each worker can be compared to the further away (the slowest). 

The updates work by letting all the workers perform updates on their own data independently and asynchronously. But the system can enforce a synchronization step if some workers get too far away from each other (when the number of updates between such models is above $s$). After the forced synchronization the workers can continue to update their model independently as before. 

At first each worker fetches a copy of the global model parameters from the parameter server. Each worker's local update step is done using the normal optimization algorithm, such as for the SGD: $x_i = x_i - \gamma \cdot \nabla L(x_i)$. Each worker maintains a versioning of each update he has made (like a simple counter of the number of update steps or a timestamp of that update). After each local update the worker sends the parameter server the computed gradient update (update vector) and the version. Each server when receiving the update from a worker will update the global model and save the versioning of that worker. When the staleness threshold is reached, each worker will update its local model with the global model values. Periodic pulls are done by workers to ensure that they are working on a recent model state.

Each worker $i$ uses a mini-batch of data $\mathbb{B_i}$.

The parameter server receives from each worker their updated $x_i$, or $- \gamma \cdot \nabla L (x_i)$, and the versioning. Each worker then continue to compute using their local parameters even if these values might be stale compared to the global version.


#### Worker:
Suppose that each worker first pulls a local copy of the model parameters, $\theta$, and at each local iteration $t$, the worker processes a mini-batch of data and computes the gradient of its loss function, and performs a gradient descent update locally: $\Delta x_{i, t} = - \gamma \cdot \nabla L (x_{i, t}, \mathbb{B_i})$, where $ \Delta x_{i, t}$ is the gradient update, $L$ is the loss function and $\mathbb{B_i}$ is the worker's mini-batch and $\gamma$ is the learning rate. Then the local model is updated as: $x_{i, t+1} = x_{i, t} + \Delta x_{i, t}$. After computing an update the worker sends to the server $\Delta x_{i, t}$ and the current iteration count. Due to SSP, (if the staleness is within the threshold $s$) the worker continues to perform these updates even though its local model might be *stale* compared to the updated one on the parameter server, therefore each worker once in a while pull a new model version from the parameter server. 

#### Server:
The parameter server keeps the global model $\theta$ updated by collecting all the workers updates. 
The server whenever receives an update keeps it saved until all the workers finished the update for the $t$-th task. Then the server aggregates those incoming updates and updates the global model $\theta$. One way to do that is through *Parameter Averaging*, so computing the averaging on the model updates the workers sent: $\Delta \theta_t = \frac{1}{n} \cdot \sum_{i=1}^{n} \Delta x_{i, t}$, where $n$ is the number of workers, and then $\theta_{t+1} = \theta_t + \Delta \theta_t$.
The workers enforce SSP by keeping the staleness among the workers under control, sending them updates and forcing a synchronization step in case the staleness gap between workers is above the threshold $s$.