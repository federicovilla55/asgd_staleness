"""
Test ASAP-SGD

From the base repository directory:  
`python -m ASGD.experiments.asap_sgd`
"""
from __future__ import annotations
import time, pathlib, pickle, random, sys
import numpy as np
import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt
from numpy.linalg import svd
from scipy.stats import ttest_rel
import os

from .. import *

# Checkpoint directory
CHECKPOINT_DIR = pathlib.Path(__file__).with_suffix("").with_name("ckpt") / "ASAP_SGD"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoint files
SGD_LOSS_F   = CHECKPOINT_DIR / "sgd_losses.pkl"
ASAP_LOSS_F  = CHECKPOINT_DIR / "asap_losses.pkl"
ASAP_STAT_F  = CHECKPOINT_DIR / "asap_worker_stats.pkl"

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Fix the master seed so you always get the same “sub‑seeds”
    random.seed(1234)
    # Draw 100 integers in [0, 2^8)
    seeds = [random.randrange(2**8) for _ in range(100)]  # If you change the amount of seeds, the first n will still always be the same !

    # FILES FOR CHECKPOINTING
    sgd_losses_f = SGD_LOSS_F
    asgd_losses_f = ASAP_LOSS_F
    asgd_stats_f  = ASAP_STAT_F

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
            asgd_params, dim, stats = run_ssp_training(dataset_builder, model, params_ssp, ParameterServer, worker)
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
    '''plt.figure()
    plt.hist(diffs, bins=20, edgecolor='black')
    plt.axvline(mean_diff, color='red', linestyle='dashed', linewidth=1, label=f"Mean: {mean_diff:.2e}")
    plt.axvline(median_diff, color='blue', linestyle='dotted', linewidth=1, label=f"Median: {median_diff:.2e}")
    plt.xlabel("SGD_loss - ASGD_loss")
    plt.ylabel("Frequency")
    plt.title("Distribution of Loss Differences (SGD vs. ASGD)")
    plt.legend()
    plt.tight_layout()
    plt.show()'''

if __name__ == "__main__":
    main()
