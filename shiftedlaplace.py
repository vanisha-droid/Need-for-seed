import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Primitives ────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np


def csv_to_binary_dataset(
    input_csv,
    output_csv=None,
    n=200,
    d=100,
    label_columns=None,
    threshold=127,
    random_state=42,
):
    """
    Convert a CSV dataset into a binary NumPy-style dataset with shape (n, d),
    matching the format returned by:

        generate_sample_data(n=200, d=20, p=0.3)

    Final output:
        - exactly n rows
        - exactly d features
        - values only 0 or 1

    Parameters
    ----------
    input_csv : str
        Path to input CSV file.

    output_csv : str or None
        Optional path to save processed dataset.

    n : int
        Number of rows.

    d : int
        Number of features.

    label_columns : list or None
        Columns to drop (e.g. labels such as 'label').

    threshold : int or float
        Threshold used to binarise values.
        Values > threshold become 1, otherwise 0.

    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        Binary dataset of shape (n, d).
    """

    # -----------------------------
    # Load CSV
    # -----------------------------
    df = pd.read_csv(input_csv)

    # -----------------------------
    # Remove label columns if needed
    # -----------------------------
    if label_columns is not None:
        df = df.drop(columns=label_columns, errors="ignore")

    # -----------------------------
    # Keep only numeric columns
    # -----------------------------
    df = df.select_dtypes(include=[np.number])

    # -----------------------------
    # Ensure enough rows
    # -----------------------------
    if len(df) < n:
        raise ValueError(f"Dataset only has {len(df)} rows, but n={n} requested.")

    # -----------------------------
    # Randomly sample n rows
    # -----------------------------
    df = df.sample(n=n, random_state=random_state).reset_index(drop=True)

    # -----------------------------
    # Ensure exactly d columns
    # -----------------------------
    if df.shape[1] < d:
        raise ValueError(
            f"Dataset only has {df.shape[1]} numeric columns, but d={d} requested."
        )

    # Use first d features
    df = df.iloc[:, :d]

    # -----------------------------
    # Convert to binary (0/1)
    # -----------------------------
    binary_df = (df > threshold).astype(int)

    # -----------------------------
    # Convert to NumPy array
    # -----------------------------
    data = binary_df.to_numpy(dtype=int)

    # -----------------------------
    # Save if requested
    # -----------------------------
    if output_csv is not None:
        pd.DataFrame(data).to_csv(output_csv, index=False)
        print(f"Saved processed dataset to: {output_csv}")

    return data


# -------------------------------------------------
# Example usage for MNIST 0/1 dataset
# -------------------------------------------------
def x21():

    # load raw .data file
    df = pd.read_csv(
        "agaricus-lepiota.data",
        header=None
    )

    print(df.head())

    binary_df = pd.get_dummies(df)

    data = (
    binary_df
    .sample(n=200, random_state=42)
    .iloc[:, :100]
    .to_numpy(dtype=int)
    )

    return data


    # Matches generate_sample_data output format:
    # array([[0,1,1,...],
    #        [1,0,0,...],
    #        ...])


def fair_coin():
    return random.randint(0, 1)


def count_flips_biased_coin(gamma):
    flips = 0
    while True:
        a = fair_coin(); flips += 1
        b = fair_coin(); flips += 1
        if a == 1:
            if b <= gamma:
                return b, flips
        else:
            return 0, flips


def sample_bernoulli(gamma):
    if 0 <= gamma <= 1:
        k = 1
        while True:
            A, flips = count_flips_biased_coin(gamma / k)
            if A == 0:
                break
            else:
                k += 1
        if k % 2 == 0:
            return 0, flips
        else:
            return 1, flips
    else:
        flips = 0
        for k in range(1, math.floor(gamma) + 1):
            B, f = sample_bernoulli(math.exp(-1))
            flips += f
            if B == 0:
                return 0, flips
        C, f = sample_bernoulli(math.exp(math.floor(gamma) - gamma))
        flips += f
        return C, flips


def sample_discrete_laplace(s, t):
    total_flips = 0
    while True:
        while True:
            U = random.randint(0, t - 1)
            total_flips += math.ceil(math.log2(t)) if t > 1 else 1
            D, f = sample_bernoulli(math.exp(-U / t))
            total_flips += f
            if D != 0:
                break
        V = 0
        while True:
            A, f = sample_bernoulli(math.exp(-1))
            total_flips += f
            if A == 0:
                break
            V += 1
        X = U + V * t
        Y = math.floor(X / s)
        B = fair_coin()
        total_flips += 1
        if B == 1 and Y == 0:
            continue
        Z = (1 - 2 * B) * Y
        return Z, total_flips


def dlap_batch_counted(n, s=1, t=20):
    samples, flip_counts = [], []
    for _ in range(n):
        z, f = sample_discrete_laplace(s, t)
        samples.append(z)
        flip_counts.append(f)
    return np.array(samples), np.array(flip_counts)


# ── Helpers ───────────────────────────────────────────────────────────────────

def floor_mod(v, m, s):
    ms = m * s
    return math.floor(v / ms) * ms

def sample_discrete_laplace_fast(scale):
    p = math.exp(-1 / scale)

    g1 = np.random.geometric(1 - p) - 1
    g2 = np.random.geometric(1 - p) - 1

    return g1 - g2


def sample_laplace_lt_m(scale, m):
    while True:
        eta = sample_discrete_laplace_fast(scale)
        if abs(eta) < m:
            return eta


def sample_laplace_geq_m(scale, m):
    sign = random.choice([-1, 1])

    p = math.exp(-1 / scale)

    tail = np.random.geometric(1 - p) - 1

    return sign * (m + tail)




# ── Algorithm 4.5 – Mechanism 5 ──────────────────────────────────────────────

def mechanism5(x, eps, m, s):
    x = np.asarray(x, dtype=int)
    n, d = x.shape

    eps_d = eps / d
    p = 2 * math.exp(-eps_d * (m - 1)) / (math.exp(eps_d) + 1)
    p = min(max(p, 0.0), 1.0)

    t_val = np.random.binomial(d, p)
    J = set(np.random.choice(d, size=t_val, replace=False).tolist())

    omega = random.randint(1, s) * m
    col_sum = x.sum(axis=0)

    lap_scale = max((d / eps), 1)

    y = []
    for i in range(d):
        si = int(col_sum[i])
        if i in J:
            eta = sample_laplace_geq_m(scale=lap_scale, m=m)
            yi  = floor_mod(si + omega + eta, m, s)
        else:
            lo = floor_mod(si + omega - m, m, s)
            hi = floor_mod(si + omega + m, m, s)
            if lo == hi:
                yi = lo
            else:
                eta = sample_laplace_lt_m(scale=lap_scale, m=m)
                yi  = floor_mod(si + omega + eta, m, s)
        y.append(yi)
    return y


# ── Data generator ────────────────────────────────────────────────────────────

def generate_sample_data(n, d, p=0.3, seed=None):
    rng = np.random.default_rng(seed)
    return rng.binomial(1, p, size=(n, d))


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_true_vs_dp_with_envelope(true_sum, all_private_sums,
                                  epsilon, n_runs=10):
    """
    Plot true sum vs multiple DP runs.

    - Blue line   : true sum
    - Orange line : mean DP estimate
    - Shaded band : ±1 std deviation
    - Faint lines : individual DP runs
    """

    x = np.arange(len(true_sum))
    all_private_sums = np.array(all_private_sums)

    mean_private = np.mean(all_private_sums, axis=0)
    std_private  = np.std(all_private_sums, axis=0)

    plt.figure(figsize=(12, 6))

    # ── Individual noisy runs ─────────────────────────────
    for i, run in enumerate(all_private_sums):
        plt.plot(
            x,
            run,
            color='orange',
            alpha=0.12,
            linewidth=0.8,
            label='Individual DP runs' if i == 0 else None
        )

    # ── True sum ──────────────────────────────────────────
    plt.plot(
        x,
        true_sum,
        marker='o',
        color='steelblue',
        linewidth=2,
        markersize=5,
        label='True Sum',
        zorder=5
    )

    # ── Mean DP estimate ─────────────────────────────────
    plt.plot(
        x,
        mean_private,
        marker='x',
        color='darkorange',
        linewidth=2,
        markersize=6,
        label=f'DP Mean ({n_runs} runs)',
        zorder=4
    )

    # ── Std deviation envelope ───────────────────────────
    plt.fill_between(
        x,
        mean_private - std_private,
        mean_private + std_private,
        color='orange',
        alpha=0.25,
        label='±1 Std Dev Envelope'
    )

    plt.xlabel("Coordinate index", fontsize=12)
    plt.ylabel("Sum", fontsize=12)

    plt.title(
        f"True vs Differentially Private Column Sum\n"
        f"({n_runs} independent runs, ε={epsilon})",
        fontsize=13,
        fontweight='bold'
    )

    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # ── Summary stats ────────────────────────────────────
    errors = all_private_sums - true_sum

    print(f"\n=== Summary over {n_runs} DP runs ===")
    print(f"Mean absolute error : {np.abs(errors).mean():.3f}")
    print(f"Mean std deviation  : {std_private.mean():.3f}")
    print(f"Max std deviation   : {std_private.max():.3f}")

# ── Randomness scaling experiment ────────────────────────────────────────────
def randomness_scaling_experiment():
    dims = [5, 10, 20, 40, 80, 160]
    avg_draws_scaled = []   # s scales with d  (paper's O(log d) regime)
    avg_draws_fixed  = []   # s = 25 constant  (your original, O(d) regime)

    N   = 50
    EPS = 1.0
    SEED = 42

    for D in dims:
        np.random.seed(SEED)
        random.seed(SEED)

        data = x21()
        x = np.asarray(data, dtype=int)
        col_sum = x.sum(axis=0)

        # ── helper: count draws for one (M, S) choice ──────────────────
        def count_draws(M, S):
            eps_d = EPS / D
            p = min(1.0, max(0.0,
                2 * math.exp(-eps_d * (M - 1)) / (math.exp(eps_d) + 1)))
            t_val = np.random.binomial(D, p)
            J = set(np.random.choice(D, size=t_val, replace=False).tolist())
            omega = random.randint(1, S) * M

            draws = len(J)
            for i in range(D):
                if i not in J:
                    lo = floor_mod(int(col_sum[i]) + omega - M, M, S)
                    hi = floor_mod(int(col_sum[i]) + omega + M, M, S)
                    if lo != hi:
                        draws += 1
            return draws

        # ── fixed s = 25, m = 5  (original) ────────────────────────────
        avg_draws_fixed.append(count_draws(M=5, S=25))

        # ── scaled s = d·polylog(d/ε),  m = polylog(d/ε) ───────────────
        polylog = max(1, int(math.log(max(D / EPS, 2)) ** 2))
        M_scaled = M
        S_scaled = S                        # s·m scales correctly
        avg_draws_scaled.append(count_draws(M=M_scaled, S=S_scaled))

    # ── reference curves ────────────────────────────────────────────────
    log_d   = [math.log2(d) for d in dims]
    linear_d = dims

    # ── plot ─────────────────────────────────────────────────────────────
    plt.figure(figsize=(9, 5))

    plt.plot(dims, avg_draws_scaled, marker='s', linewidth=2,
             color='#2563EB', label='Shifted Laplace (fixed params)')
    plt.plot(dims, log_d,            linestyle=':', linewidth=1.4,
             color='#6B7280', label='O(log d) reference')
    plt.plot(dims, linear_d,         linestyle='--', linewidth=1.4,
             color='#F59E0B', label='O(d) reference')

    plt.title("Laplace draws vs Dimension\n",
              fontsize=12)
    plt.xlabel("Dimension d")
    plt.ylabel("Laplace draws needed")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

def bits_comparison_three_way(
    flip_counts_dlap,
    d,
    epsilon=1.0
):
    """
    Randomness comparison:
        OpenDP baseline vs exact Discrete Laplace sampler.

    Parameters
    ----------
    flip_counts_dlap : np.ndarray
        Observed fair coin flips used by the exact DLap sampler.

    d : int
        Dimension count (for title only)

    epsilon : float
        Privacy parameter
    """

    opendp_bits_per_sample = 64

    dlap_mean = flip_counts_dlap.mean()

    # ── Print summary ───────────────────────────────────
    print("\n" + "=" * 60)
    print("Randomness comparison")
    print("=" * 60)

    print(f"OpenDP baseline (PRNG):      {opendp_bits_per_sample:.2f} bits/sample")
    print(f"Discrete Laplace (CKS):      {dlap_mean:.2f} bits/sample")

    print("\nRatios:")
    print(f"OpenDP / DLap:               "
          f"{opendp_bits_per_sample / dlap_mean:.2f}x")

    # ── Figure ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colours = {
        "OpenDP": "#DC2626",
        "DLap":   "#059669",
    }

    # ====================================================
    # Panel 1: distribution
    # ====================================================
    ax = axes[0]

    ax.hist(
        flip_counts_dlap,
        bins=range(
            int(flip_counts_dlap.min()),
            int(flip_counts_dlap.max()) + 2
        ),
        density=True,
        alpha=0.6,
        color=colours["DLap"],
        label="DLap samples"
    )

    ax.axvline(
        dlap_mean,
        color=colours["DLap"],
        linestyle="--",
        linewidth=2,
        label=f"DLap mean = {dlap_mean:.1f}"
    )

    ax.axvline(
        opendp_bits_per_sample,
        color=colours["OpenDP"],
        linestyle="-",
        linewidth=2,
        label=f"OpenDP baseline = {opendp_bits_per_sample}"
    )

    ax.set_title("Per-sample randomness cost")
    ax.set_xlabel("Fair bits consumed")
    ax.set_ylabel("Density")

    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    # ====================================================
    # Panel 2: bar comparison
    # ====================================================
    ax = axes[1]

    methods = ["OpenDP", "Discrete Laplace"]
    values  = [opendp_bits_per_sample, dlap_mean]

    bars = ax.bar(methods, values, alpha=0.8)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            val + 0.5,
            f"{val:.1f}",
            ha='center',
            fontsize=10
        )

    ax.set_ylabel("Average fair bits per sample")

    ax.set_title(
        f"Average randomness usage\n"
        f"(ε={epsilon}, d={d})"
    )

    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Parameters ──
    N   = 5000      # number of rows (individuals)
    D   = 20      # number of coordinates / columns
    EPS = 10.0     # privacy budget ε
    M   = 19      # threshold m  (controls noise split in Mechanism 5)
    S   = 25      # granularity s

    SEED = 42

    # ── Generate data and compute true sum ──
    data     = x21()
    true_sum = data.sum(axis=0)          # shape (D,)

    # ── Run multiple independent DP releases ──
    N_RUNS = 10

    all_private_sums = []

    for _ in range(N_RUNS):
        private_sum = mechanism5(data, eps=EPS, m=M, s=S)
        all_private_sums.append(private_sum)

    # ── Plot envelope ──
    plot_true_vs_dp_with_envelope(
        true_sum,
        all_private_sums,
        epsilon=EPS,
        n_runs=N_RUNS
    )

    # Run experiment
    randomness_scaling_experiment()

    # ── Randomness experiment ──────────────────────────────

    NUM_SAMPLES = 5000

    _, flip_counts_dlap = dlap_batch_counted(
        NUM_SAMPLES,
        s=1,
        t=20
    )

    bits_comparison_three_way(
        flip_counts_dlap=flip_counts_dlap,
        d=D,
        epsilon=EPS
    )