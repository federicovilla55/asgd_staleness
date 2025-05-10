import pathlib
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

CHECKPOINT_DIR = pathlib.Path(__file__).with_suffix("").with_name("ckpt") / "ASGD_DROPOUT"

loss_files = {
    "SGD":              CHECKPOINT_DIR / "sgd_losses.pkl",
    "SGD_Dropout":     CHECKPOINT_DIR / "sgd_dropout_losses.pkl",
    "SGD_L2":          CHECKPOINT_DIR / "sgd_l2_losses.pkl",
    "SGD_Gaussian":    CHECKPOINT_DIR / "sgd_gauss_losses.pkl",
    "ASAP_SGD":        CHECKPOINT_DIR / "asap_losses.pkl",
}

loss_data = {}
for name, path in loss_files.items():
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    with open(path, 'rb') as f:
        loss_data[name] = pickle.load(f)

num_seeds = min(len(v) for v in loss_data.values())
seeds = np.arange(num_seeds)

plt.figure(figsize=(10, 6))
for name, losses in loss_data.items():
    aligned = losses[:num_seeds]
    plt.plot(seeds, aligned, marker='o', linestyle='-', label=name)

plt.title("Test Loss per Seed Across Training Methods")
plt.xlabel("Seed Index")
plt.ylabel("Test Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

os.makedirs("deliverables", exist_ok=True)
plt.savefig(f"deliverables/loss_comparison_plot.png", dpi=300)
print("Plot saved as `deliverables/loss_comparison_plot.png`")