"""
Test DASGD

From the base repository directory:  
`python -m ASGD.experiments.dasgd`
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
import scipy.stats as stats_mod

from .. import *
#from ASGD import *


def main():
    # AMOUNT OF SEEDS YOU WANT TO COMPUTE NOW
    # TODO : change to 20 runs !
    RUNS_REGULAR_SGD = 200
    RUNS_ASGD = 200

    # USER WILL HAVE TO CHOOSE THE AMOUNT OF OVERPARAMETRIZATION
    args = parse_args()

    # every run uses n_samples=100
    n_samples = 100
    # compute features = level% of samples
    n_features = int(n_samples * args.overparam / 100)

    # base checkpoint tree
    BASE_CKPT = pathlib.Path(__file__).parent / "ckpt"
    # e.g. ckpt/overparam_150/SGD  and ckpt/overparam_150/ASAP_SGD
    cfg_dir = BASE_CKPT / f"overparam_{args.overparam}"
    SGD_DIR   = cfg_dir / "SGD"
    DASGD_DIR  = cfg_dir / "DASGD"
    for d in (SGD_DIR, DASGD_DIR):
        d.mkdir(parents=True, exist_ok=True)


    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Fix the master seed so you always get the same “sub‑seeds”
    random.seed(1234)
    # Draw 100 integers in [0, 2^8)
    seeds = [random.randrange(2**8) for _ in range(200)]  # If you change the amount of seeds, the first n will still always be the same !

    # FILES FOR CHECKPOINTING
    sgd_losses_f = 'sgd_losses.pkl'
    asgd_losses_f = 'ASGD_losses.pkl'
    asgd_stats_f  = 'ASGD_stats.pkl'
    staleness_distr_f = 'ASGD_staleness_distr.pkl'
    SGD_weight_properties_f = 'sgd_weight_properties.pkl'
    ASGD_weight_properties_f = 'ASGD_weight_properties.pkl'
    true_weight_properties_f = 'true_weight_properties.pkl'

    # For each checkpoint file
    sgd_losses_file = os.path.join(SGD_DIR, sgd_losses_f)
    asgd_losses_file = os.path.join(DASGD_DIR, asgd_losses_f)
    asgd_stats_file  = os.path.join(DASGD_DIR, asgd_stats_f)
    staleness_distr_file = os.path.join(DASGD_DIR, staleness_distr_f)
    SGD_weight_properties_file = os.path.join(SGD_DIR, SGD_weight_properties_f)
    ASGD_weight_properties_file = os.path.join(DASGD_DIR, ASGD_weight_properties_f)
    true_weight_properties_file = os.path.join(SGD_DIR, true_weight_properties_f)

    if RUNS_REGULAR_SGD > 0:
        #RETRIEVE LOSSES
        losses_file = sgd_losses_file
        if os.path.exists(losses_file):
            with open(losses_file, 'rb') as f:
                SGD_losses = pickle.load(f)
            logging.info(f"Resuming: {len(SGD_losses)}/{len(seeds)} seeds done")
        else:
            SGD_losses = []
            logging.info("Starting fresh, no existing losses file found")

        # RETRIEVE/INIT WEIGHT PROPERTIES
        if os.path.exists(SGD_weight_properties_file):
            with open(SGD_weight_properties_file, 'rb') as f:
                SGD_weight_properties = pickle.load(f)
        else:
            if len(SGD_losses) == 0:
                SGD_weight_properties = [] 
            else: # In the case that you start tracking  after some runs already have been computed
                SGD_weight_properties = [None] * len(SGD_losses)
            logging.info("Starting fresh on weigth metrics")

        # RETRIEVE/INIT TRUE WEIGHT PROPERTIES
        if os.path.exists(true_weight_properties_file):
            with open(true_weight_properties_file, 'rb') as f:
                true_weights = pickle.load(f)
        else:
            if len(SGD_losses) == 0:
                true_weights = [] 
            else: # In the case that you start tracking  after some runs already have been computed
                true_weights = [None] * len(SGD_losses)
            logging.info("Starting fresh on weigth metrics")
        

        # Pick up where you left off
        start_idx = len(SGD_losses)
        for idx in range(start_idx, len(seeds)):
            seed = seeds[idx]
            
            if RUNS_REGULAR_SGD == 0:
                print("Performed the specified amount of runs for regular SGD")
                break
            RUNS_REGULAR_SGD = RUNS_REGULAR_SGD - 1

            # full splits => Always the same when using the same seed
            X_tr_lin, y_tr_lin, X_val_lin, y_val_lin, X_te_lin, y_te_lin, true_w = load_linear_data(n_samples=n_samples, n_features=n_features, noise=0.0,val_size=0.01,test_size=0.2, random_state= seed)

            X_comb = np.vstack([X_tr_lin, X_val_lin])
            y_comb = np.concatenate([y_tr_lin, y_val_lin])

            n_trainval = X_comb.shape[0]
            batch_size = max(1, int(0.1 * n_trainval))

            # 3) Compute 95% of max stable step size η₉₅
            _, S_comb, _ = svd(X_comb, full_matrices=False)
            eta_max = 2.0 / (S_comb[0]**2)
            eta_95  = 0.95 * eta_max

            start = time.perf_counter()
            sgd_model = sgd_training(X_comb, y_comb, num_epochs = 10000, criterion = nn.MSELoss(), batch_size = batch_size, lr = eta_95, tol=1e-8)
            end = time.perf_counter()
            sgd_time = end-start

            # Compute weight metrics on true weight vector
            true_m_gd = {'l2':l2_norm(true_w),'sparsity':sparsity_ratio(true_w),'kurtosis':weight_kurtosis(true_w)}
            true_weights.append(true_m_gd)

            # collect each parameter, detach from graph, move to CPU numpy, flatten
            weight_vectors = []
            for param in sgd_model.parameters():
                weight_vectors.append(param.detach().cpu().numpy().reshape(-1))
            w = np.concatenate(weight_vectors)
            # Compute your three metrics
            m_gd = {'l2':l2_norm(w),'sparsity':sparsity_ratio(w),'kurtosis':weight_kurtosis(w)}
            SGD_weight_properties.append(m_gd)

            SGD_loss = evaluate_model("SGD", sgd_model, X_te_lin, y_te_lin)

            SGD_losses.append(SGD_loss)

            print("Time Comparison for run:" + str(idx) + f":SGD {sgd_time:2f} sec")
        

        # SAVE LOSSES
        with open(sgd_losses_file, 'wb') as f:
            pickle.dump(SGD_losses, f)

        with open(sgd_losses_file, 'rb') as f:
            SGD_losses = pickle.load(f)
        print("Retrieved regular SGD losses")

        avg_SGD_loss = sum(SGD_losses)/len(SGD_losses)
        print("Average SGD loss =" + str(avg_SGD_loss))

        # SAVE WEIGHT METRICS/PROPERTIES
        with open(SGD_weight_properties_file, 'wb') as f:
            pickle.dump(SGD_weight_properties, f)

        # SAVE TRUE WEIGHT METRICS/PROPERTIES
        with open(true_weight_properties_file, 'wb') as f:
            pickle.dump(true_weights, f)

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
        
        #INIT/RETRIEVE STALENESS DISTR
        if os.path.exists(staleness_distr_file):
            with open(staleness_distr_file, 'rb') as f:
                ASGD_staleness_distributions = pickle.load(f)
            logging.info(f"Resuming staleness distr: {len(ASGD_staleness_distributions)}/{len(seeds)} done")
        else:
            if len(ASGD_losses) == 0:
                ASGD_staleness_distributions = [] 
            else: # In the case that you start tracking these distributions after some runs already have been computed
                ASGD_staleness_distributions = [None] * len(ASGD_losses)
            logging.info("Starting fresh on staleness distr")
        
        # INIT/RETRIEVE WEIGHT METRICS/PROPERTIES
        
        if os.path.exists(ASGD_weight_properties_file):
            with open(ASGD_weight_properties_file, 'rb') as f:
                ASGD_weight_properties = pickle.load(f)
            logging.info(f"Resuming weight properties: {len(ASGD_weight_properties)}/{len(seeds)} done")
        else:
            if len(ASGD_losses) == 0:
                ASGD_weight_properties  = [] 
            else: # In the case that you start tracking these distributions after some runs already have been computed
                ASGD_weight_properties  = [None] * len(ASGD_losses)
            logging.info("Starting fresh on ASGD weight properties")

        # Pick up where you left off
        start_idx = len(ASGD_losses)
        for idx in range(start_idx, len(seeds)):
            seed = seeds[idx]

            if RUNS_ASGD == 0:
                print("Performed the specified amount of runs for ASGD")
                break
            RUNS_ASGD = RUNS_ASGD - 1

            # full splits => Always the same when using the same seed
            X_tr_lin, y_tr_lin, X_val_lin, y_val_lin, X_te_lin, y_te_lin, true_weight = load_linear_data(n_samples= n_samples, n_features= n_features, noise=0.0, val_size=0.01,test_size=0.2, random_state=seed)

            X_comb = np.vstack([X_tr_lin, X_val_lin])
            y_comb = np.concatenate([y_tr_lin, y_val_lin])

            n_trainval = X_comb.shape[0]
            batch_size = max(1, int(0.1 * n_trainval))

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
                lr = eta_95,                          # DEPENDING ON ALGO THIS HAS TO BE CHANGED !
                local_steps = 10000,
                batch_size = batch_size,
                device = "cuda" if torch.cuda.is_available() else "cpu",
                log_level = logging.DEBUG,
                tol = 1e-8,                             # The tol for workers is currently set at tol = 1e-8
                Amplitude = 1                           # The max amplitude IN ASAP
            )

            # Run the SSP training and measure the time taken
            start = time.perf_counter()
            asgd_params, dim, stats, staleness_distr = run_training(dataset_builder, model, params_ssp, parameter_server=ParameterServerSAASGD)
            end = time.perf_counter()
            asgd_time = end - start
            ASGD_stats.append(stats)

            # Compute staleness distribution
            freq = np.array(staleness_distr) / sum(staleness_distr)  # normalize to probabilities
            ASGD_staleness_distributions.append(freq)

            # Evaluate the trained model on the test set
            asgd_model = build_model(asgd_params, model, dim)

            flat_parts = []
            for param in asgd_model.parameters():
                flat_parts.append(param.detach().cpu().numpy().reshape(-1))
            w_asgd = np.concatenate(flat_parts)
             # Compute weight metrics/properties
            m_asgd = {'l2':l2_norm(w_asgd),'sparsity': sparsity_ratio(w_asgd),'kurtosis': weight_kurtosis(w_asgd)}
            ASGD_weight_properties.append(m_asgd)

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

        # SAVE THE STALENESS DISTRIBUTIONS 
        with open(staleness_distr_file, 'wb') as f:
            pickle.dump(ASGD_staleness_distributions, f)
        
        # SAVE THE WEIGHT METRICS/PROPERTIES
        with open(ASGD_weight_properties_file, 'wb') as f:
            pickle.dump(ASGD_weight_properties, f)

    # COMPARE LOSSES FOR THE SEEDS THAT HAVE BEEN USED IN BOTH METHODS UNTIL NOW

    # Align lengths (in case one list is longer because of incomplete runs)
    n = min(len(SGD_losses), len(ASGD_losses))
    sgd_losses = SGD_losses[:n]
    asgd_losses = ASGD_losses[:n]

    # Compute difference: SGD_loss - ASGD_loss
    diffs = np.array(sgd_losses) - np.array(asgd_losses)

    # COMPUTE PAIRED T-TEST
    if n > 1:
        t_stat, p_value = stats_mod.ttest_rel(sgd_losses, asgd_losses, nan_policy='omit')

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

    # VISUALIZE THE STALENESS DISTRIBUTION OF THE LAST 3 RUNS
    #–– Extract the last three runs
    last3 = ASGD_staleness_distributions[-3:]   # list of length 3, each shape (S+1,)
    taus  = np.arange(last3[0].shape[0])        # 0 … max staleness
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, freq, run_idx in zip(
            axes, last3, range(len(ASGD_staleness_distributions)-3, len(ASGD_staleness_distributions))
        ):
        ax.bar(taus, freq, edgecolor='k', alpha=0.7)
        ax.set_title(f"Run {run_idx}")
        ax.set_xlabel("τ")
    axes[0].set_ylabel("P(τ)")
    fig.suptitle("Last 3 Runs: Staleness Distributions")
    plt.tight_layout()
    plt.show()

    # COMPARE THE WEIGHT METRICS/PROPERTIES
    
    # 1) Make a mask of valid runs
    M = min(len(SGD_weight_properties), len(ASGD_weight_properties), len(true_weights))
    mask = np.array([
        (SGD_weight_properties[i] is not None) and
        (ASGD_weight_properties[i] is not None) and
        (true_weights[i] is not None)
        for i in range(M)
    ])

    keys = ('l2','sparsity','kurtosis')

    # build the arrays of shape (N,3)
    sgd_arr   = np.vstack([ [SGD_weight_properties[i][k] for k in keys]
                            for i in range(M) if mask[i] ])
    asgd_arr  = np.vstack([ [ASGD_weight_properties[i][k] for k in keys]
                            for i in range(M) if mask[i] ])
    true_arr  = np.vstack([ [true_weights[i][k]           for k in keys]
                            for i in range(M) if mask[i] ])
    N = sgd_arr.shape[0]

    # 3) Paired differences
    diffs = sgd_arr - asgd_arr   # shape (N,3)
    
    # Descriptive summaries and confidence intervals
    for j,key in enumerate(keys):
        d = diffs[:,j]
        m, s = d.mean(), d.std(ddof=1)
        ci_low, ci_high = stats_mod.t.interval( 0.95, df=N-1, loc=m, scale=s/np.sqrt(N))
        print(f"{key}: mean diff = {m:.4f}, 95% CI = [{ci_low:.4f}, {ci_high:.4f}]")

    # Paired hypothesis testing and Effect-size (Cohen’s d for paired data)
    for j,key in enumerate(keys):
        d = diffs[:,j]
        d_mean, d_std = d.mean(), d.std(ddof=1)
        cohens_d = d_mean / d_std
        t_stat, p_t = stats_mod.ttest_rel(sgd_arr[:,j], asgd_arr[:,j])
        p_w = stats_mod.wilcoxon(d).pvalue
        print(f"{key}: Cohen’s d = {cohens_d:.3f}")
        print(f"{key}: paired t-test p = {p_t:.3e}, wilcoxon p = {p_w:.3e}")

    # Correlation with generalization gap
    sgd_sel = np.array(SGD_losses[:M])[mask]
    asgd_sel= np.array(ASGD_losses[:M])[mask]
    loss_diff = sgd_sel - asgd_sel
    for j,key in enumerate(keys):
        r, p = stats_mod.pearsonr(diffs[:,j], loss_diff)
        print(f"Corr(loss_diff, {key}_diff): r = {r:.3f}, p = {p:.3e}")

    # Boxplot
    fig, axes = plt.subplots(1,3,figsize=(12,4))
    for j,key in enumerate(keys):
        axes[j].boxplot([sgd_arr[:,j], asgd_arr[:,j]], labels=['SGD','ASGD'])
        axes[j].set_title(key)
    plt.tight_layout(); plt.show()

    for j,key in enumerate(keys):
        plt.figure()
        plt.scatter(sgd_arr[:,j], asgd_arr[:,j], alpha=0.7)
        lim = max(sgd_arr[:,j].max(), asgd_arr[:,j].max())
        plt.plot([0,lim],[0,lim], linestyle='--')
        plt.xlabel('SGD'); plt.ylabel('ASGD'); plt.title(key)
        plt.tight_layout(); plt.show()

    delta_sgd  = np.abs(sgd_arr  - true_arr)   # how far each run’s SGD metrics sit from its ground truth
    delta_asgd = np.abs(asgd_arr - true_arr)

    # — now compute distance-to-teacher for each method —
    # average signed difference in *distance* to teacher:
    for j,key in enumerate(keys):
        # negative means ASGD is *closer* (on average) to the teacher than SGD
        mean_dist_diff = delta_sgd[:,j].mean() - delta_asgd[:,j].mean()
        print(f"{key}: mean(|SGD-teacher| - |ASGD-teacher|) = {mean_dist_diff:.4f}")

    # you can also do a paired test on these distances:
    for j,key in enumerate(keys):
        d = delta_sgd[:,j] - delta_asgd[:,j]
        t_stat, pval = stats_mod.ttest_rel(delta_sgd[:,j], delta_asgd[:,j])
        print(f"{key}: paired t-test on dist-to-teacher p = {pval:.3e}")

    # — and finally, overlay the teacher’s *average* metric in your boxplots —
    teacher_means = true_arr.mean(axis=0)

    fig, axes = plt.subplots(1,3,figsize=(12,4))
    for j,key in enumerate(keys):
        axes[j].boxplot([sgd_arr[:,j], asgd_arr[:,j]], labels=['SGD','ASGD'])
        # horizontal line at the *average* teacher metric
        axes[j].axhline(teacher_means[j],
                        color='C2', linestyle='--', label='teacher')
        axes[j].set_title(key)
        axes[j].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
