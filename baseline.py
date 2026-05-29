import numpy as np
import matplotlib.pyplot as plt
from opendp.measurements import make_laplace
from opendp.domains import atom_domain, vector_domain
from opendp.metrics import l1_distance
from opendp.mod import enable_features

enable_features("contrib")

def dp_sum(data: np.ndarray, epsilon: float, beta: float = 0.01):
    if data.ndim != 2:
        raise ValueError("Data must be n x d array.")
    n, d = data.shape
    ts = np.sum(data, axis=0).astype(float)
    domain = vector_domain(
        atom_domain(T=float, bounds=(0., float(n)), nan=False),
        size=d
    )
    metric = l1_distance(float)
    scale = d / epsilon
    laplace_noise = make_laplace(domain, metric, scale)
    noisy = np.array(laplace_noise(ts))
    private = noisy
    return ts, private



def generate_sample_data(n=200, d=20, p=0.3):
    return np.random.binomial(1, p, size=(n, d))


def plot_true_vs_dp_with_envelope(true_sum, all_private_sums, n_runs=10):
    """
    Plot true sum (blue) vs multiple DP runs:
    - Orange line  = mean of all DP runs
    - Shaded band  = ±1 std deviation envelope
    - Light traces = individual DP runs (for transparency)
    """
    x = np.arange(len(true_sum))
    all_private_sums = np.array(all_private_sums)  # shape: (n_runs, d)

    mean_private = np.mean(all_private_sums, axis=0)
    std_private  = np.std(all_private_sums, axis=0)

    plt.figure(figsize=(12, 6))

    # Individual DP runs (faint)
    for i, run in enumerate(all_private_sums):
        plt.plot(x, run, color='orange', alpha=0.15, linewidth=0.8,
                 label='Individual DP runs' if i == 0 else None)

    # True sum
    plt.plot(x, true_sum, marker='o', color='steelblue', linewidth=2,
             markersize=5, label='True Sum', zorder=5)

    # Mean DP sum
    plt.plot(x, mean_private, marker='x', color='darkorange', linewidth=2,
             markersize=6, label=f'DP Mean ({n_runs} runs)', zorder=4)

    # ±1 std envelope
    plt.fill_between(x,
                     mean_private - std_private,
                     mean_private + std_private,
                     color='orange', alpha=0.25,
                     label='±1 Std Dev Envelope')

    plt.xlabel("Coordinate (dimension index)", fontsize=12)
    plt.ylabel("Sum", fontsize=12)
    plt.title(
        f"True vs Differentially Private Sum\n"
        f"(fixed dataset, {n_runs} independent DP runs, ε={epsilon})",
        fontsize=13
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # --- Summary stats ---
    errors = all_private_sums - true_sum  # shape: (n_runs, d)
    print(f"\n=== Summary over {n_runs} DP runs ===")
    print(f"Mean absolute error (per coord, averaged): "
          f"{np.abs(errors).mean():.3f}")
    print(f"Mean std dev across coordinates:           "
          f"{std_private.mean():.3f}")
    print(f"Max std dev (noisiest coordinate):         "
          f"{std_private.max():.3f}")


# ── Experiment ────────────────────────────────────────────────────────────────
# Fix ONE dataset instance; run the DP mechanism N_RUNS times on it.
# This isolates the randomness of the mechanism from data randomness.

np.random.seed(23)
N_RUNS   = 10
epsilon  = 1.0

data = generate_sample_data(n=200, d=20, p=0.3)   # fixed for all runs

all_private = []
for _ in range(N_RUNS):
    true_sum, private_sum = dp_sum(data, epsilon)
    all_private.append(private_sum)

print("True sums:   ", true_sum)
print("DP mean sums:", np.mean(all_private, axis=0).round(2))

plot_true_vs_dp_with_envelope(true_sum, all_private, n_runs=N_RUNS)