# train_ssp.py
import logging
import torch
from DASGD import run_ssp_training, ConfigParameters
from datasets import create_full_data_loader
from model import LinearNetModel
from datasets import load_linear_data, create_linear_data_loader
from async_ssp import nn
from async_ssp import evaluate_model, build_model
import time
import numpy as np
from scipy.linalg import svd

X_tr, y_tr, X_val, y_val, X_te, y_te = load_linear_data(
n_samples=201, n_features=210, noise=0.0,val_size=0.01,test_size=0.2, random_state=42 )

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


_, S_comb, _ = svd(X_comb, full_matrices=False)
eta_max = 2.0 / (S_comb[0]**2)
eta_95  = 0.95 * eta_max


def sgd_training(num_epochs = 10000, criterion = nn.MSELoss(), batch_size = 32, lr = eta_95, tol=1e-8):
    # Combine the training and validation data

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
        num_workers = 4,
        staleness = 10,
        lr = eta_95,
        local_steps = 400,
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