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

'''
# Inside SGD training loop (simulate ASGD-like noise)
for p in model.parameters():
    p.grad += torch.randn_like(p.grad) * noise_scale
Compare test loss between:

SGD + noise (vary noise_scale)

ASGD (vary staleness)




Purpose: Test if staleness-induced regularization matches explicit methods like dropout or L2.
Method:

Train SGD + L2 (weight decay) and SGD + Dropout on the same dataset.

Compare test loss with ASGD.
Key Metrics:

Test loss for SGD + L2, SGD + Dropout, and ASGD.

Parameter norms (ASGD vs. SGD + L2).
'''


# Checkpoint directory
CHECKPOINT_DIR = pathlib.Path(__file__).with_suffix("").with_name("ckpt") / "ASGD_DROPOUT"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# SGD Checkpoint files
SGD_LOSS_F   = CHECKPOINT_DIR / "sgd_losses.pkl"
SGD_DROPOUT_LOSS_F   = CHECKPOINT_DIR / "sgd_dropout_losses.pkl"
SGD_L2_LOSS_F   = CHECKPOINT_DIR / "sgd_l2_losses.pkl"
SGD_GAUSS_LOSS_F   = CHECKPOINT_DIR / "sgd_gauss_losses.pkl"

# ASGD checkpoint files
ASAP_LOSS_F  = CHECKPOINT_DIR / "asap_losses"
ASAP_STAT_F  = CHECKPOINT_DIR / "asap_worker_stats"

asgd_configs = {
    "A": dict(num_workers=4, staleness=10, local_steps=10000, batch_size=32),
    "B": dict(num_workers=10, staleness=20,  local_steps=10000, batch_size=32),
    "C": dict(num_workers=10, staleness=10, local_steps=10000, batch_size=32),
    "D": dict(num_workers=10, staleness=20, local_steps=5000, batch_size=32),
    "E": dict(num_workers=8, staleness=10, local_steps=10000, batch_size=32),
    "F": dict(num_workers=5, staleness=10, local_steps=20000, batch_size=32),
    "G": dict(num_workers=8, staleness=10, local_steps=10000, batch_size=64),
    #"H": dict(num_workers=2, staleness=5, local_steps=10000, batch_size=32),
}


def compare_loss(sgd_name, SGD_losses, ASGD_losses) -> None:
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
    print(f"Mean difference ({sgd_name} - ASGD): {mean_diff:.4e}")
    print(f"Median difference: {median_diff:.4e}")
    print(f"Std of difference: {std_diff:.4e}")

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Fix the master seed so you always get the same “sub‑seeds”
    random.seed(1234)
    num_seeds =30
    
    # Draw `num_seeds`` integers in [0, 2^8]
    # If you change the amount of seeds, the first n will still always be the same !
    seeds = [random.randrange(2**8) for _ in range(num_seeds)]  

    # FILES FOR CHECKPOINTING
    sgd_losses_f = SGD_LOSS_F
    sgd_dropout_losses_f = SGD_DROPOUT_LOSS_F
    sgd_l2_losses_f = SGD_L2_LOSS_F
    sgd_gauss_losses_f = SGD_GAUSS_LOSS_F
    sgd_losses_files = {
        "SGD" : sgd_losses_f, 
        "SGD_Dropout" : sgd_dropout_losses_f,
        "SGD_L2" : sgd_l2_losses_f,
        "SGD_Gaussian_noise" : sgd_gauss_losses_f
    }

    # get the directory this script lives in
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # then for each checkpoint file
    #asgd_losses_f = ASAP_LOSS_F
    #asgd_stats_f  = ASAP_STAT_F
    #asgd_losses_file = os.path.join(script_dir, asgd_losses_f)
    #asgd_stats_file  = os.path.join(script_dir, asgd_stats_f)

    # AMOUNT OF SEEDS YOU WANT TO COMPUTE NOW
    RUNS_SGD = 1       # Set always min to 1 for both methods (if want to retrieve/use the stored values)
    RUNS_ASGD = 1


    num_samples = 100
    num_features = 110

    if RUNS_SGD > 0:
        for name, losses_file in sgd_losses_files.items():
            remaining_runs = RUNS_SGD
            print(f"> Running {name}")
            if os.path.exists(losses_file):
                with open(losses_file, 'rb') as f:
                    SGD_losses = pickle.load(f)
                logging.info(f"Resuming: {len(SGD_losses)}/{len(seeds)} seeds done for {name}")
            else:
                SGD_losses = []
                logging.info("Starting fresh, no existing losses file found")

            # Pick up where you left off
            start_idx = len(SGD_losses)
            for idx in range(start_idx, len(seeds)):
                seed = seeds[idx]
                
                if remaining_runs == 0:
                    print(f"Performed the specified amount of runs for {name}")
                    break
                remaining_runs -= 1

                # full splits => Always the same when using the same seed
                X_tr_lin, y_tr_lin, X_val_lin, y_val_lin, X_te_lin, y_te_lin = load_linear_data(
                    n_samples=num_samples, n_features=num_features, noise=0.0, 
                    val_size=0.01,test_size=0.2, random_state=seed
                )

                X_comb = np.vstack([X_tr_lin, X_val_lin])
                y_comb = np.concatenate([y_tr_lin, y_val_lin])

                # 3) Compute 95% of max stable step size η₉₅
                _, S_comb, _ = svd(X_comb, full_matrices=False)
                eta_max = 2.0 / (S_comb[0]**2)
                eta_95  = 0.95 * eta_max


                start = time.perf_counter()

                if name == "SGD":
                    sgd_model = sgd_training(
                        X_comb, 
                        y_comb, 
                        num_epochs = 10000, 
                        criterion = nn.MSELoss(), 
                        batch_size = 32, 
                        lr = eta_95, 
                        tol=1e-8, 
                    )      
                elif name == "SGD_Dropout":
                    sgd_model = sgd_training_l2(
                        X_comb, 
                        y_comb, 
                        num_epochs = 10000, 
                        criterion = nn.MSELoss(), 
                        batch_size = 32, 
                        lr = eta_95, 
                        tol=1e-8, 
                        weight_decay=0.01
                    )
                elif name == "SGD_L2":
                    sgd_model = sgd_training_noise(
                        X_comb, 
                        y_comb, 
                        num_epochs = 10000, 
                        criterion = nn.MSELoss(), 
                        batch_size = 32, 
                        lr = eta_95, 
                        tol=1e-8, 
                        noise_scale=0.01
                    )
                elif name == "SGD_Gaussian_noise":
                    sgd_model = sgd_training_dropout(
                        X_comb, 
                        y_comb, 
                        num_epochs = 10000, 
                        criterion = nn.MSELoss(), 
                        batch_size = 32, 
                        lr = eta_95, 
                        tol=1e-8, 
                        dropout_p=0.1
                    )   
                else:
                    print(f"ERROR: No model associated with `{name}`.")
                    break

                end = time.perf_counter()
                sgd_time = end-start

                SGD_loss = evaluate_model(name, sgd_model, X_te_lin, y_te_lin)

                SGD_losses.append(SGD_loss)

                print("Time Comparison for run:" + str(idx) + f":{name} {sgd_time:2f} sec")
            
            sgd_losses_file = os.path.join(script_dir, losses_file)

            with open(sgd_losses_file, 'wb') as f:
                pickle.dump(SGD_losses, f)

            with open(sgd_losses_file, 'rb') as f:
                SGD_losses = pickle.load(f)

            #print(f"Losses: {SGD_losses}")

            avg_SGD_loss = sum(SGD_losses)/len(SGD_losses)
            print(f"Average {name} loss = {str(avg_SGD_loss)}")

    if RUNS_ASGD > 0:
        for cid, cfg in asgd_configs.items():
            print(f"\n=== Running ASGD config {cid} ===")
            # checkpoint paths per config
            loss_file = CHECKPOINT_DIR / f"asap_losses_{cid}.pkl"
            stat_file = CHECKPOINT_DIR / f"asap_stats_{cid}.pkl"

            # load or init losses and stats
            if os.path.exists(loss_file):
                with open(loss_file, 'rb') as f:
                    ASGD_losses = pickle.load(f)
                logging.info(f"Resuming {cid}: {len(ASGD_losses)}/{num_seeds} seeds done")
            else:
                ASGD_losses = []

            if os.path.exists(stat_file):
                with open(stat_file, 'rb') as f:
                    ASGD_stats = pickle.load(f)
            else:
                ASGD_stats = []

            # Pick up where you left off
            start_idx = len(ASGD_losses)
            remaining_runs = RUNS_ASGD
            for idx in range(start_idx, len(seeds)):
                seed = seeds[idx]

                if remaining_runs == 0:
                    print("Performed the specified amount of runs for ASGD")
                    break
                remaining_runs -= 1

                # full splits => Always the same when using the same seed
                X_tr_lin, y_tr_lin, X_val_lin, y_val_lin, X_te_lin, y_te_lin = load_linear_data(
                    n_samples=num_samples, n_features=num_features, noise=0.0,
                    val_size=0.01,test_size=0.2, random_state=seed
                )

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
                params = ConfigParameters(
                    num_workers=cfg['num_workers'],
                    staleness=cfg['staleness'],
                    lr=eta_95/2,
                    local_steps=cfg['local_steps'],
                    batch_size=cfg['batch_size'],
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    log_level=logging.DEBUG,
                    tol=1e-8,
                    Amplitude=1
                )

                # Run the SSP training and measure the time taken
                start = time.perf_counter()
                asgd_params, dim, stats = run_training(dataset_builder, model, params, ParameterServerASAP, worker)
                end = time.perf_counter()
                asgd_time = end - start
                ASGD_stats.append(stats)

                #'''
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
                #'''

                # Evaluate the trained model on the test set
                asgd_model = build_model(asgd_params, model, dim)

                ASGD_loss = evaluate_model(f"ASGD_{cid}", asgd_model, X_te_lin, y_te_lin)

                ASGD_losses.append(ASGD_loss)

                print(f"Time Comparison for configuration ASGD_{cid} run {str(idx)} : {asgd_time:2f} sec")

            # SAVE THE LOSSES
            with open(loss_file, 'wb') as f:
                pickle.dump(ASGD_losses, f)

            with open(loss_file, 'rb') as f:
                ASGD_losses = pickle.load(f)
            
            avg_ASGD_loss = sum(ASGD_losses)/len(ASGD_losses)

            print(f"Average ASGD_{cid} loss = {str(avg_ASGD_loss)}")

            #SAVE THE WORKER STATS
            with open(stat_file, 'wb') as f:
                pickle.dump(ASGD_stats, f)

        # If you want to inspect the stats you can do:
        # with open(stats_file, 'rb') as f:
        #     ASGD_stats = pickle.load(f)
        # now ASGD_stats is a list of dicts, each having
        #   stats["per_worker"] and stats["combined"]

    '''for name, losses_file in sgd_losses_files.items():
        if os.path.exists(losses_file):
                with open(losses_file, 'rb') as f:
                    SGD_losses = pickle.load(f)
        else:
            SGD_losses = []
        compare_loss(
            sgd_name=name,
            SGD_losses=SGD_losses,
            ASGD_losses=ASGD_losses
        )'''    

    print(f"{'='*50}\n\n")

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
    for i in range(100):
        main()
