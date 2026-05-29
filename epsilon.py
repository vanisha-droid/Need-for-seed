"""
=============================================================================
EPSILON SCALING EXPERIMENT
=============================================================================

Compare four DP mechanisms across three privacy budgets:
  epsilon in {0.1, 1.0, 10.0}

Mechanisms:
  1. OpenDP standard Laplace (PRNG baseline, 64 bits/coord)
  2. Huffman-based Laplace sampler (fair coin flips, raw discretisation)
  3. Discrete Laplace / CKS sampler (fair coin flips)
  4. Shifted Laplace mechanism / Algorithm 4.5 (PRNG, log2(d)/d bits/coord)

Plots produced:
  (a) Trajectory envelope — rows = mechanisms, columns = epsilons
  (b) MAE vs epsilon — grouped bar chart
  (c) Fair bits per sample vs epsilon — grouped bar chart

Fixed dataset: n=200, d=20, Bernoulli(0.3), seed=42
n_runs = 100 independent runs per (mechanism, epsilon)
No subsampling (gamma = 1.0)
Huffman: raw discretised Laplace, no rounding
DLap:    s=1, t=round(scale), minimum t=2
Mech5:   m=25, s_param=20, randomness counted as ceil(log2(d))/d bits/coord
=============================================================================
"""

import heapq
import math
import random
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from fractions import Fraction

import opendp.prelude as dp
dp.enable_features("contrib", "floating-point")
from opendp.domains import vector_domain, atom_domain
from opendp.metrics import l1_distance
from opendp.measurements import make_laplace


# =============================================================================
# RANDOM PRIMITIVES
# =============================================================================

def fair_coin():
    return random.randint(0, 1)


def count_flips_biased_coin(p, max_bits=64):
    """
    Simulate biased coin(p) using fair coins via binary expansion.
    Expected cost: 2 fair flips regardless of p.
    """
    p_frac     = Fraction(p).limit_denominator(10**12)
    flips_used = 0
    for _ in range(max_bits):
        p_frac *= 2
        p_bit   = 1 if p_frac >= 1 else 0
        p_frac -= p_bit
        flips_used += 1
        if fair_coin() != p_bit:
            return p_bit, flips_used
    return random.randint(0, 1), flips_used


# =============================================================================
# DISCRETISE LAPLACE
# =============================================================================

def build_px(scale, grid_step=1.0, tail_prob=0.999):
    """
    Discretise Laplace(0, scale) onto a uniform integer grid.
    Integrates PDF over each bin [x - step/2, x + step/2].
    """
    trunc = np.ceil(stats.laplace.ppf((1 + tail_prob) / 2, scale=scale))
    xs    = np.arange(-trunc, trunc + grid_step, grid_step)
    probs = (
        stats.laplace.cdf(xs + grid_step / 2, scale=scale) -
        stats.laplace.cdf(xs - grid_step / 2, scale=scale)
    )
    probs = probs / probs.sum()
    mask  = probs > 0
    return xs[mask], probs[mask]


# =============================================================================
# HUFFMAN TREE + SAMPLER
# =============================================================================

class Node:
    def __init__(self, prob, symbol=None):
        self.prob   = prob
        self.symbol = symbol
        self.left   = None
        self.right  = None

    def __lt__(self, other):
        return self.prob < other.prob


def build_huffman_tree(xs, probs):
    heap = [Node(prob=p, symbol=x) for x, p in zip(xs, probs)]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo           = heapq.heappop(heap)
        hi           = heapq.heappop(heap)
        parent       = Node(prob=lo.prob + hi.prob)
        parent.left  = lo
        parent.right = hi
        heapq.heappush(heap, parent)
    return heap[0]


def huffman_sample_fair(root):
    """
    Traverse Huffman tree with fair coin flips.
    Returns (symbol, total_fair_flips).
    """
    node             = root
    total_fair_flips = 0
    while node.symbol is None:
        p_left              = node.left.prob / (node.left.prob + node.right.prob)
        outcome, flips_used = count_flips_biased_coin(p_left)
        total_fair_flips   += flips_used
        node                = node.left if outcome == 1 else node.right
    return node.symbol, total_fair_flips


# =============================================================================
# DISCRETE LAPLACE SAMPLER (CKS)
# =============================================================================

def sample_bernoulli_exp(gamma):
    """Sample Bernoulli(exp(-gamma)) using fair coins."""
    if 0 <= gamma <= 1:
        k     = 1
        flips = 0
        while True:
            A, f = count_flips_biased_coin(gamma / k)
            flips += f
            if A == 0:
                break
            k += 1
        return (1 if k % 2 != 0 else 0), flips
    else:
        flips = 0
        for k in range(1, math.floor(gamma) + 1):
            B, f = sample_bernoulli_exp(math.exp(-1))
            flips += f
            if B == 0:
                return 0, flips
        C, f = sample_bernoulli_exp(gamma - math.floor(gamma))
        flips += f
        return C, flips


def sample_discrete_laplace(s, t):
    """
    Sample from Discrete Laplace(s, t), scale ≈ t/s.
    Returns (sample, fair_flips_used).
    """
    total_flips = 0
    while True:
        while True:
            U            = random.randint(0, t - 1)
            total_flips += math.ceil(math.log2(t)) if t > 1 else 1
            D, f = sample_bernoulli_exp(U / t)
            total_flips += f
            if D != 0:
                break
        V = 0
        while True:
            A, f = sample_bernoulli_exp(1)
            total_flips += f
            if A == 0:
                break
            V += 1
        X = U + V * t
        Y = math.floor(X / s)
        B            = fair_coin()
        total_flips += 1
        if not (B == 1 and Y == 0):
            return (1 - 2 * B) * Y, total_flips


def dlap_batch(n, s, t):
    samples, flips = [], []
    for _ in range(n):
        z, f = sample_discrete_laplace(s, t)
        samples.append(z)
        flips.append(f)
    return np.array(samples), np.array(flips)


def dlap_params_for_scale(scale):
    """s=1, t=round(scale), minimum t=2."""
    return 1, max(2, round(scale))


def floor_mod(v, m, s):
    """Round v down to nearest multiple of m*s."""
    ms = m * s
    return math.floor(v / ms) * ms


def sample_discrete_laplace_fast(scale):
    """
    Fast geometric-based discrete Laplace sample (PRNG).
    Used inside mechanism5 — not fair-coin based.
    """
    p  = math.exp(-1.0 / scale)
    g1 = np.random.geometric(1 - p) - 1
    g2 = np.random.geometric(1 - p) - 1
    return int(g1 - g2)


def sample_laplace_lt_m(scale, m):
    """Sample discrete Laplace conditioned on |eta| < m."""
    while True:
        eta = sample_discrete_laplace_fast(scale)
        if abs(eta) < m:
            return eta


def sample_laplace_geq_m(scale, m):
    """Sample discrete Laplace conditioned on |eta| >= m."""
    p    = math.exp(-1.0 / scale)
    sign = random.choice([-1, 1])
    tail = int(np.random.geometric(1 - p) - 1)
    return sign * (m + tail)


def mechanism5(x, eps, m, s_param, d):
    """
    Shifted Laplace mechanism (Algorithm 4.5).

    Parameters
    ----------
    x       : np.ndarray, shape (n, d) — raw dataset
    eps     : float — privacy budget
    m       : int   — rounding / threshold parameter (fixed at 25)
    s_param : int   — shift scale parameter (fixed at 20)
    d       : int   — number of dimensions

    Returns
    -------
    y           : list of noisy sums, length d
    bits_used   : int — estimated fair bits consumed (ceil(log2(d)) per coord)
    """
    x       = np.asarray(x, dtype=int)
    n, d_   = x.shape
    eps_d   = eps / d_

    # probability that a coordinate lands in the heavy-tail set J
    p_J     = 2 * math.exp(-eps_d * (m - 1)) / (math.exp(eps_d) + 1)
    p_J     = min(max(p_J, 0.0), 1.0)

    # draw |J| ~ Binomial(d, p_J), then pick |J| coords uniformly
    t_val   = int(np.random.binomial(d_, p_J))
    J       = set(np.random.choice(d_, size=t_val, replace=False).tolist())

    # shared random shift — uniform over {m, 2m, ..., s_param * m}
    omega   = int(random.randint(1, s_param)) * m

    col_sum = x.sum(axis=0)

    # noise scale for Laplace draws
    lap_scale = max(d_ / eps, 1.0)

    y = []
    for i in range(d_):
        si = int(col_sum[i])
        if i in J:
            eta = sample_laplace_geq_m(scale=lap_scale, m=m)
            yi  = floor_mod(si + omega + eta, m, s_param)
        else:
            lo = floor_mod(si + omega - m, m, s_param)
            hi = floor_mod(si + omega + m, m, s_param)
            if lo == hi:
                yi = lo
            else:
                eta = sample_laplace_lt_m(scale=lap_scale, m=m)
                yi  = floor_mod(si + omega + eta, m, s_param)
        y.append(yi)

    bits_used = math.ceil(math.log2(max(d_, 2))) * d_ / d   # log2(d) bits total, averaged over d coordinates

    return np.array(y, dtype=float), bits_used


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_sample_data(n=200, d=20, p=0.3, seed=42):
    np.random.seed(seed)
    return np.random.binomial(1, p, size=(n, d))


# =============================================================================
# SINGLE-RUN WRAPPERS
# =============================================================================

def run_opendp(true_sums, scale, d, n):
    domain = vector_domain(
        atom_domain(T=float, bounds=(0., float(n)), nan=False), size=d
    )
    metric = l1_distance(float)
    meas   = make_laplace(domain, metric, scale)
    noisy  = np.array(meas(list(true_sums)))
    return noisy, 64 * d          # 64-bit PRNG per coordinate


def run_huffman(true_sums, root, d):
    noise       = []
    total_flips = 0
    for _ in range(d):
        s, f = huffman_sample_fair(root)
        noise.append(s)
        total_flips += f
    return true_sums + np.array(noise), total_flips


def run_dlap(true_sums, s_param, t_param, d):
    noise, flips = dlap_batch(d, s_param, t_param)
    return true_sums + noise.astype(float), int(flips.sum())


def run_mech5(data, eps, m, s_param, d):
    noisy, bits = mechanism5(data, eps, m, s_param, d)
    return noisy, bits


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_epsilon_experiment(
    epsilons  = [0.1, 1.0, 10.0],
    n         = 200,
    d         = 20,
    p_data    = 0.3,
    n_runs    = 100,
    seed      = 42,
    m_mech5   = 25,
    s_mech5   = 20,
):
    random.seed(seed)
    np.random.seed(seed)

    data      = generate_sample_data(n=n, d=d, p=p_data, seed=seed)
    true_sums = data.sum(axis=0).astype(float)

    results = {
        'epsilons'  : epsilons,
        'true_sums' : true_sums,
        'data'      : data,
    }

    for eps in epsilons:
        scale     = d / eps
        s_d, t_d  = dlap_params_for_scale(scale)
        xs, probs = build_px(scale)
        root      = build_huffman_tree(xs, probs)
        H         = -np.sum(probs * np.log2(probs + 1e-300))
        log2d     = math.ceil(math.log2(max(d, 2)))

        print(f'\n{"="*65}')
        print(f'  epsilon={eps}  scale={scale:.2f}  H={H:.3f}  '
              f'dlap s={s_d} t={t_d}  log2(d)={log2d}')
        print(f'{"="*65}')

        runs = {
            'opendp'        : [],
            'huffman'       : [],
            'dlap'          : [],
            'mech5'         : [],
            'flips_opendp'  : [],
            'flips_huffman' : [],
            'flips_dlap'    : [],
            'flips_mech5'   : [],
            'scale'         : scale,
            'H'             : H,
        }

        for i in range(n_runs):
            no, fo = run_opendp(true_sums, scale, d, n)
            nh, fh = run_huffman(true_sums, root, d)
            nd, fd = run_dlap(true_sums, s_d, t_d, d)
            nm, fm = run_mech5(data, eps, m_mech5, s_mech5, d)

            runs['opendp'].append(no)
            runs['huffman'].append(nh)
            runs['dlap'].append(nd)
            runs['mech5'].append(nm)
            runs['flips_opendp'].append(fo)
            runs['flips_huffman'].append(fh)
            runs['flips_dlap'].append(fd)
            runs['flips_mech5'].append(fm)

            if (i + 1) % 25 == 0:
                print(f'    run {i+1}/{n_runs} done')

        for k in ['opendp', 'huffman', 'dlap', 'mech5']:
            runs[k] = np.array(runs[k])
        for k in ['flips_opendp', 'flips_huffman', 'flips_dlap', 'flips_mech5']:
            runs[k] = np.array(runs[k], dtype=float)

        print(f'\n  {"Mechanism":<30}  {"MAE":>8}  {"bits/sample":>12}')
        print(f'  {"-"*30}  {"-"*8}  {"-"*12}')
        for mech, label in [
            ('opendp',  'OpenDP Laplace'),
            ('huffman', 'Huffman Laplace'),
            ('dlap',    'Discrete Laplace (CKS)'),
            ('mech5',   'Shifted Laplace (Mech 5)'),
        ]:
            mae  = float(np.abs(runs[mech] - true_sums).mean())
            bits = float(runs[f'flips_{mech}'].mean()) / d
            print(f'  {label:<30}  {mae:>8.3f}  {bits:>12.3f}')

        results[eps] = runs

    return results


# =============================================================================
# PLOTTING
# =============================================================================

COLOURS = {
    'opendp'  : '#E05C5C',
    'huffman' : '#378ADD',
    'dlap'    : '#1D9E75',
    'mech5'   : '#9B59B6',
}

LABELS = {
    'opendp'  : 'OpenDP Laplace',
    'huffman' : 'Huffman Laplace',
    'dlap'    : 'Discrete Laplace (CKS)',
    'mech5'   : 'Shifted Laplace',
}

MECHS = ['opendp', 'huffman', 'dlap', 'mech5']


# -----------------------------------------------------------------------------
# (a) Trajectory envelopes — 4 rows (mechanisms) x 3 cols (epsilons)
# -----------------------------------------------------------------------------

def plot_trajectories(results, d=20):
    epsilons  = results['epsilons']
    true_sums = results['true_sums']
    coords    = np.arange(d)

    fig, axes = plt.subplots(
        len(MECHS), len(epsilons),
        figsize=(5 * len(epsilons), 4 * len(MECHS)),
        sharey='row'
    )

    fig.suptitle(
        'Trajectory comparison — all mechanisms across ε ∈ {0.1, 1.0, 10.0}\n'
        f'n=200, d=20, 100 runs, fixed dataset',
        fontsize=12
    )

    for row, mech in enumerate(MECHS):
        for col, eps in enumerate(epsilons):
            ax   = axes[row, col]
            runs = results[eps][mech]
            mean = runs.mean(axis=0)
            std  = runs.std(axis=0)
            c    = COLOURS[mech]

            for i, run in enumerate(runs):
                ax.plot(coords, run, color=c, alpha=0.06,
                        linewidth=0.5,
                        label='runs' if i == 0 else None)

            ax.fill_between(coords, mean - std, mean + std,
                            alpha=0.18, color=c)

            ax.plot(coords, mean, color=c, linewidth=1.8,
                    linestyle='--', marker='^', markersize=3,
                    label='mean')

            ax.plot(coords, true_sums, color='steelblue',
                    linewidth=2, marker='o', markersize=3,
                    label='True Sum', zorder=10)

            ax.set_title(
                f'{LABELS[mech]}\nε={eps}  λ={results[eps]["scale"]:.1f}',
                fontsize=9
            )
            ax.set_xlabel('Coordinate', fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.tick_params(labelsize=7)

            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('epsilon_trajectories.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: epsilon_trajectories.png')


# -----------------------------------------------------------------------------
# (b) MAE vs epsilon
# -----------------------------------------------------------------------------

def plot_mae_vs_epsilon(results, d=20):
    epsilons  = results['epsilons']
    true_sums = results['true_sums']

    mae    = {mech: [] for mech in MECHS}
    for eps in epsilons:
        for mech in MECHS:
            mae[mech].append(
                float(np.abs(results[eps][mech] - true_sums).mean())
            )

    x      = np.arange(len(epsilons))
    n_mech = len(MECHS)
    width  = 0.18
    offsets = np.linspace(-(n_mech - 1) * width / 2,
                           (n_mech - 1) * width / 2,
                           n_mech)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, mech in enumerate(MECHS):
        bars = ax.bar(x + offsets[i], mae[mech], width,
                      alpha=0.82, color=COLOURS[mech],
                      label=LABELS[mech])
        for bar, val in zip(bars, mae[mech]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=7)
        ax.plot(x + offsets[i], mae[mech],
                color=COLOURS[mech], linewidth=1.4,
                linestyle='--', marker='o', markersize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f'ε = {e}' for e in epsilons], fontsize=10)
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=10)
    ax.set_title(
        'MAE vs privacy budget ε\n'
        'Lower ε = stronger privacy = more noise = higher MAE',
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('epsilon_mae.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: epsilon_mae.png')


# -----------------------------------------------------------------------------
# (c) Bits per sample vs epsilon
# -----------------------------------------------------------------------------

def plot_bits_vs_epsilon(results, d=20):
    epsilons = results['epsilons']

    bits   = {mech: [] for mech in MECHS}
    H_vals = []

    for eps in epsilons:
        r = results[eps]
        H_vals.append(r['H'])
        for mech in MECHS:
            bits[mech].append(
                float(r[f'flips_{mech}'].mean()) / d
            )

    x       = np.arange(len(epsilons))
    n_mech  = len(MECHS)
    width   = 0.18
    offsets = np.linspace(-(n_mech - 1) * width / 2,
                           (n_mech - 1) * width / 2,
                           n_mech)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, mech in enumerate(MECHS):
        bars = ax.bar(x + offsets[i], bits[mech], width,
                      alpha=0.82, color=COLOURS[mech],
                      label=LABELS[mech])
        for bar, val in zip(bars, bits[mech]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=7)

    # Huffman analytical bounds
    ax.plot(x, H_vals,
            color=COLOURS['huffman'], linewidth=2,
            linestyle=':', marker='D', markersize=6,
            label='H(p) lower bound (Huffman)')
    ax.plot(x, [2 * h for h in H_vals],
            color=COLOURS['huffman'], linewidth=1.5,
            linestyle=':', marker='D', markersize=6,
            alpha=0.5, label='2H(p) upper bound (Huffman)')

    # log2(d) reference line for Shifted Laplace
    log2d = math.ceil(math.log2(max(d, 2)))
    log2dbyd = log2d / d
    ax.axhline(log2dbyd, color=COLOURS['mech5'], linewidth=1.5,
               linestyle='--',
               label=f'log₂(d) = {log2dbyd:.3f} (Shifted Laplace target)')

    ax.set_xticks(x)
    ax.set_xticklabels([f'ε = {e}' for e in epsilons], fontsize=10)
    ax.set_ylabel('Mean fair bits per sample', fontsize=10)
    ax.set_title(
        'Randomness cost vs privacy budget ε\n'
        'Higher ε = smaller scale = lower entropy = fewer bits',
        fontsize=11
    )
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('epsilon_bits.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: epsilon_bits.png')


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_summary_table(results, d=20):
    true_sums = results['true_sums']
    log2d     = math.ceil(math.log2(max(d, 2)))
    log2dbyd  = log2d / d

    print('\n' + '=' * 75)
    print('  Epsilon scaling summary')
    print('=' * 75)
    print(f'  {"ε":>5}  {"λ":>7}  {"H(p)":>6}  '
          f'{"Mechanism":<28}  {"MAE":>8}  {"bits/coord":>11}')
    print('  ' + '-' * 71)

    for eps in results['epsilons']:
        r = results[eps]
        for mech in MECHS:
            mae  = float(np.abs(r[mech] - true_sums).mean())
            bits = float(r[f'flips_{mech}'].mean()) / d
            print(f'  {eps:>5}  {r["scale"]:>7.2f}  {r["H"]:>6.3f}  '
                  f'{LABELS[mech]:<28}  {mae:>8.3f}  {bits:>11.3f}')
        print('  ' + '-' * 71)

    print(f'\n  Note: Shifted Laplace randomness counted as '
          f'ceil(log2(d)) = {log2dbyd:.3f} bits/coord (PRNG-based).')
    print('=' * 75)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    EPSILONS = [0.1, 1.0, 10.0]
    D        = 20
    N        = 200
    N_RUNS   = 100
    SEED     = 42
    M_MECH5  = 25
    S_MECH5  = 20

    print('Epsilon scaling experiment')
    print(f'  epsilons={EPSILONS}  d={D}  n={N}  '
          f'n_runs={N_RUNS}  seed={SEED}')
    print(f'  Shifted Laplace: m={M_MECH5}, s={S_MECH5}')
    print(f'  Randomness accounting:')
    print(f'    OpenDP        : 64 bits/coord (PRNG)')
    print(f'    Huffman       : H(p) to 2H(p) fair bits/coord')
    print(f'    Discrete Lap  : fair bits/coord (CKS, counted)')
    print(f'    Shifted Lap   : ceil(log2(d))='
          f'{math.ceil(math.log2(max(D,2)))} bits/coord (PRNG, O(log d))')

    print('\nNoise scales:')
    for eps in EPSILONS:
        scale     = D / eps
        s_d, t_d  = dlap_params_for_scale(scale)
        xs, probs = build_px(scale)
        H         = -np.sum(probs * np.log2(probs + 1e-300))
        print(f'  eps={eps}  scale={scale:.2f}  H={H:.3f}  '
              f'dlap s={s_d} t={t_d}  symbols={len(xs)}')

    print('\nRunning...')
    results = run_epsilon_experiment(
        epsilons = EPSILONS,
        n        = N,
        d        = D,
        n_runs   = N_RUNS,
        seed     = SEED,
        m_mech5  = M_MECH5,
        s_mech5  = S_MECH5,
    )

    print_summary_table(results, d=D)

    print('\nPlotting trajectories...')
    plot_trajectories(results, d=D)

    print('\nPlotting MAE vs epsilon...')
    plot_mae_vs_epsilon(results, d=D)

    print('\nPlotting bits vs epsilon...')
    plot_bits_vs_epsilon(results, d=D)