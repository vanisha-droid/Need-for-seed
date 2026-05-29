# =============================================================================
# SHARED-NOISE / LOW-RANDOMNESS MECHANISM
# =============================================================================
#
# Idea:
# -----
# Instead of sampling d independent Laplace noises,
# sample ONE Huffman noise value and reuse it across all dimensions.
#
# Standard mechanism:
#   Z_i iid ~ Laplace
#   randomness = O(d * H(p))
#
# Shared-noise mechanism:
#   sample B once
#   set Z_i = s_i * B
#   randomness ≈ O(H(p) + d sign bits)
#
# This is NOT guaranteed DP.
# It is an exploratory mechanism for studying:
#   - randomness reduction
#   - shared randomness
#   - utility degradation
#   - covariance structure
#
# =============================================================================

import heapq
import math
import random
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import opendp.prelude as dp
dp.enable_features("contrib", "floating-point")

from opendp.domains import vector_domain, atom_domain
from opendp.metrics import l1_distance
from opendp.measurements import make_laplace

from fractions import Fraction


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_sample_data(n=200, d=20, p=0.3):
    return np.random.binomial(1, p, size=(n, d))


# =============================================================================
# STANDARD OPENDP LAPLACE
# =============================================================================

def dp_sum(data: np.ndarray, epsilon: float):
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

    return ts, noisy


# =============================================================================
# FAIR COIN
# =============================================================================

def fair_coin():
    return random.randint(0, 1)


def count_flips_biased_coin(p, max_bits=64):

    p_frac = Fraction(p).limit_denominator(10**12)

    flips_used = 0

    for _ in range(max_bits):

        p_frac *= 2

        p_bit = 1 if p_frac >= 1 else 0

        p_frac -= p_bit

        flips_used += 1

        flip = fair_coin()

        if flip != p_bit:
            return p_bit, flips_used

    return random.randint(0, 1), flips_used


# =============================================================================
# DISCRETISE LAPLACE
# =============================================================================

def build_px(scale, grid_step=1.0, tail_prob=0.999):

    trunc = np.ceil(
        stats.laplace.ppf((1 + tail_prob) / 2, scale=scale)
    )

    xs = np.arange(-trunc, trunc + grid_step, grid_step)

    probs = (
        stats.laplace.cdf(xs + grid_step / 2, scale=scale)
        -
        stats.laplace.cdf(xs - grid_step / 2, scale=scale)
    )

    probs = probs / probs.sum()

    mask = probs > 0

    return xs[mask], probs[mask]


# =============================================================================
# TOP-K ROUNDING
# =============================================================================

def round_px_topk(xs, probs, k=51):

    idx = np.argsort(probs)[::-1][:k]

    idx = np.sort(idx)

    xs_r = xs[idx]

    probs_r = probs[idx]

    probs_r = probs_r / probs_r.sum()

    return xs_r, probs_r


# =============================================================================
# HUFFMAN TREE
# =============================================================================

class Node:

    def __init__(self, prob, symbol=None):

        self.prob = prob
        self.symbol = symbol

        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.prob < other.prob


def build_huffman_tree(xs, probs):

    heap = [Node(prob=p, symbol=x) for x, p in zip(xs, probs)]

    heapq.heapify(heap)

    while len(heap) > 1:

        lo = heapq.heappop(heap)

        hi = heapq.heappop(heap)

        parent = Node(prob=lo.prob + hi.prob)

        parent.left = lo
        parent.right = hi

        heapq.heappush(heap, parent)

    return heap[0]


# =============================================================================
# HUFFMAN SAMPLER
# =============================================================================

def huffman_sample_fair(root):

    node = root

    total_fair_flips = 0

    while node.symbol is None:

        p_left = (
            node.left.prob /
            (node.left.prob + node.right.prob)
        )

        outcome, flips_used = count_flips_biased_coin(p_left)

        total_fair_flips += flips_used

        node = node.left if outcome == 1 else node.right

    return node.symbol, total_fair_flips


# =============================================================================
# SHARED-NOISE MECHANISM
# =============================================================================

def shared_noise_dp_sum(
    data,
    root,
    epsilon,
    random_signs=True,
):
    """
    Sample ONE Huffman noise value and reuse it
    across all dimensions.

    Noise model:
        Z_i = s_i * B

    where:
        B = one shared Huffman sample
        s_i in {-1, +1}

    This dramatically reduces randomness usage.

    Returns:
        true_sum
        private_sum
        flips_used
    """

    n, d = data.shape

    true_sum = np.sum(data, axis=0).astype(float)

    # ONE shared noise sample
    base_noise, flips = huffman_sample_fair(root)

    # optional random signs
    if random_signs:

        signs = np.random.choice([-1, 1], size=d)

        noise = signs * base_noise

        flips += d

    else:

        noise = np.ones(d) * base_noise

    private = true_sum + noise

    return true_sum, private, flips


# =============================================================================
# PLOTTING
# =============================================================================

def plot_true_vs_dp_with_envelope(
    true_sum,
    all_private_sums,
    title,
    epsilon,
):

    x = np.arange(len(true_sum))

    all_private_sums = np.array(all_private_sums)

    mean_private = np.mean(all_private_sums, axis=0)

    std_private = np.std(all_private_sums, axis=0)

    plt.figure(figsize=(12, 6))

    # individual runs
    for i, run in enumerate(all_private_sums):

        plt.plot(
            x,
            run,
            color='orange',
            alpha=0.15,
            linewidth=0.8,
            label='Individual DP runs' if i == 0 else None
        )

    # true
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

    # mean
    plt.plot(
        x,
        mean_private,
        marker='x',
        color='darkorange',
        linewidth=2,
        markersize=6,
        label='DP Mean',
        zorder=4
    )

    # envelope
    plt.fill_between(
        x,
        mean_private - std_private,
        mean_private + std_private,
        color='orange',
        alpha=0.25,
        label='±1 Std Dev Envelope'
    )

    plt.xlabel("Coordinate")
    plt.ylabel("Sum")

    plt.title(
        f"{title}\nε={epsilon}",
        fontsize=13
    )

    plt.legend(fontsize=10)

    plt.grid(True, alpha=0.4)

    plt.tight_layout()

    plt.show()

    # summary stats
    errors = all_private_sums - true_sum

    print("\n===================================================")
    print(title)
    print("===================================================")

    print(f"Mean absolute error:      {np.abs(errors).mean():.3f}")

    print(f"Mean std across coords:   {std_private.mean():.3f}")

    print(f"Max std coordinate:       {std_private.max():.3f}")


# =============================================================================
# EXPERIMENT
# =============================================================================

if __name__ == "__main__":

    np.random.seed(23)
    random.seed(23)

    epsilon = 1.0

    N_RUNS = 100

    n = 200
    d = 20

    # -------------------------------------------------------------------------
    # FIX ONE DATASET
    # -------------------------------------------------------------------------

    data = generate_sample_data(
        n=n,
        d=d,
        p=0.3
    )

    # -------------------------------------------------------------------------
    # BUILD HUFFMAN DISTRIBUTION
    # -------------------------------------------------------------------------

    scale = d / epsilon

    xs, probs = build_px(scale=scale)

    # top-k entropy reduction
    xs, probs = round_px_topk(xs, probs, k=51)

    root = build_huffman_tree(xs, probs)

    # -------------------------------------------------------------------------
    # STANDARD OPENDP
    # -------------------------------------------------------------------------

    all_opendp = []

    for _ in range(N_RUNS):

        true_sum, private_sum = dp_sum(data, epsilon)

        all_opendp.append(private_sum)

    # -------------------------------------------------------------------------
    # SHARED-NOISE MECHANISM
    # -------------------------------------------------------------------------

    all_shared = []

    flip_counts = []

    for _ in range(N_RUNS):

        true_sum, private_sum, flips = shared_noise_dp_sum(
            data=data,
            root=root,
            epsilon=epsilon,
            random_signs=True,
        )

        all_shared.append(private_sum)

        flip_counts.append(flips)

    # -------------------------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------------------------

    plot_true_vs_dp_with_envelope(
        true_sum=true_sum,
        all_private_sums=all_opendp,
        title="OpenDP Laplace Mechanism",
        epsilon=epsilon,
    )

    plot_true_vs_dp_with_envelope(
        true_sum=true_sum,
        all_private_sums=all_shared,
        title="Shared-Noise Huffman Mechanism",
        epsilon=epsilon,
    )
  
    # -------------------------------------------------------------------------
    # RANDOMNESS COMPARISON
    # -------------------------------------------------------------------------

    H = -np.sum(probs * np.log2(probs))

    print("\n===================================================")
    print("Randomness Comparison")
    print("===================================================")

    print(f"Entropy H(p):                     {H:.4f} bits")

    print(f"Independent Huffman complexity:   O(d * H(p))")

    print(f"Shared-noise complexity:          O(H(p) + d sign bits)")

    print(f"Observed shared-noise mean bits:  {np.mean(flip_counts):.3f}")

    print(f"OpenDP baseline:                  ~64 * d bits")

    print(f"\nReduction vs independent sampling:")

    print(
        f"{(d * H) / np.mean(flip_counts):.2f}x fewer random bits"
    )