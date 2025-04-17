# train_ssp.py
import logging
import torch
from async_ssp import run_ssp_training, ConfigParameters
from datasets import create_adult_dataset
from model import LinearNetModel
from datasets import load_adult_data
from async_ssp import nn
from async_ssp import evaluate_model, build_model
import time

def sgd_training(num_epochs = 10, criterion = nn.BCELoss(), batch_size = 64, lr = 0.01):
    """
    Run SGD training on the UCI Adult dataset for income prediction.

    :param num_epochs: Number of epochs to train the model.
    :param criterion: Loss function to use for training.
    :return: None
    """
    X_train, X_test, y_train, y_test = load_adult_data()

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

        # Iterate over the batches of training data
        for Xb, yb in loader:
            optimizer.zero_grad() # Reset the gradients
            out  = model(Xb) # Forward pass
            loss = criterion(out, yb.float()) # Compute the loss
            loss.backward() # Backward pass
            optimizer.step() # Update the model parameters
            total_epoch_loss += loss.item() # Accumulate the loss

    return model

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Set up the configuration for the SSP training
    params_ssp = ConfigParameters(
        num_workers = 2,
        staleness = 10,
        lr = 0.01,
        local_steps = 5,
        batch_size = 64,
        device = "cuda" if torch.cuda.is_available() else "cpu",
        log_level = logging.DEBUG,
    )

    # Dataset builder function
    dataset_builder = create_adult_dataset
    # Model class
    model = LinearNetModel

    # Run the SSP training and measure the time taken
    start = time.perf_counter()
    asgd_params, dim = run_ssp_training(dataset_builder, model, params_ssp)
    end = time.perf_counter()
    asgd_time = end - start

    # Evaluate the trained model on the test set
    _, X_test, _, y_test = load_adult_data()
    asgd_model = build_model(asgd_params, model, dim)
    evaluate_model("ASGD", asgd_model, X_test, y_test)

    # run baseline for comparison
    start = time.perf_counter()
    sgd_model = sgd_training()
    end = time.perf_counter()
    sgd_time = end-start

    evaluate_model("SGD", sgd_model, X_test, y_test)

    print(f"Time Comparison: ASGD {asgd_time:2f} sec;\tSGD {sgd_time:2f} sec")