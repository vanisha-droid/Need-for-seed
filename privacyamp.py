"""
=============================================================================
PRIVACY AMPLIFICATION BY SUBSAMPLING
=============================================================================

Three mechanisms compared:
  1. OpenDP standard Laplace
  2. Huffman-based Laplace sampler (fair coin flips)
  3. Discrete Laplace (CKS sampler)

Two experimental framings:
  A. Fixed noise, tighter privacy:
       keep scale lambda = d/eps, subsample at rate gamma
       effective eps' ≈ gamma * eps  (stronger privacy for free)

  B. Fixed privacy target (eps' = 1.0), less noise:
       subsample at rate gamma => only need noise at scale d / (eps'/gamma)
       = smaller scale => less noise => better accuracy, same privacy

Subsampling scheme: Poisson subsampling
  each record included independently with probability gamma

Privacy amplification bound used:
  eps'_poisson ≈ log(1 + gamma*(exp(eps) - 1))
  for small eps this is approximately gamma * eps

Randomness vs gamma:
  analytical H(p_gamma) curve for each mechanism
  showing how bits-per-sample falls as gamma decreases
  (under framing B where scale shrinks with gamma)

=============================================================================
"""

import heapq
import math
import os
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
    Simulate a biased coin with probability p using fair coins
    via binary expansion. Expected cost: 2 fair flips regardless of p.
    """
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


def entropy_of_scale(scale, grid_step=1.0, tail_prob=0.999):
    """Analytical entropy of discretised Laplace at given scale."""
    xs, probs = build_px(scale, grid_step, tail_prob)
    return -np.sum(probs * np.log2(probs + 1e-300))


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
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        parent       = Node(prob=lo.prob + hi.prob)
        parent.left  = lo
        parent.right = hi
        heapq.heappush(heap, parent)
    return heap[0]


def huffman_sample_fair(root):
    """
    Sample from the Huffman tree using fair coin flips.
    Returns (symbol, total_fair_flips).
    """
    node             = root
    total_fair_flips = 0
    while node.symbol is None:
        p_left = node.left.prob / (node.left.prob + node.right.prob)
        outcome, flips_used = count_flips_biased_coin(p_left)
        total_fair_flips   += flips_used
        node = node.left if outcome == 1 else node.right
    return node.symbol, total_fair_flips


# =============================================================================
# DISCRETE LAPLACE SAMPLER (CKS)
# =============================================================================

def sample_bernoulli(gamma):
    """Sample Bernoulli(exp(-gamma)) using fair coins."""
    if 0 <= gamma <= 1:
        k = 1
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
            B, f = sample_bernoulli(math.exp(-1))
            flips += f
            if B == 0:
                return 0, flips
        C, f = sample_bernoulli(math.exp(math.floor(gamma) - gamma))
        flips += f
        return C, flips


def sample_discrete_laplace(s, t):
    """
    Sample from Discrete Laplace with parameters s, t.
    Scale ≈ t/s. Returns (sample, fair_flips_used).
    """
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
        B  = fair_coin()
        total_flips += 1
        if not (B == 1 and Y == 0):
            Z = (1 - 2 * B) * Y
            return Z, total_flips


def dlap_batch(n, s, t):
    samples, flips = [], []
    for _ in range(n):
        z, f = sample_discrete_laplace(s, t)
        samples.append(z)
        flips.append(f)
    return np.array(samples), np.array(flips)


# =============================================================================
# PRIVACY AMPLIFICATION ACCOUNTING
# =============================================================================

def amplified_epsilon(eps_mechanism, gamma):
    """
    Poisson subsampling amplification:
      eps' = log(1 + gamma * (exp(eps) - 1))
    Approximates gamma * eps for small eps.
    """
    return math.log(1 + gamma * (math.exp(eps_mechanism) - 1))


def scale_for_target_epsilon(eps_target, gamma, d):
    """
    Under framing B (fixed privacy target eps_target):
    we want amplified_epsilon(eps_mechanism, gamma) = eps_target
    => eps_mechanism = log(1 + (exp(eps_target) - 1) / gamma)
    => scale = d / eps_mechanism
    """
    eps_mechanism = math.log(1 + (math.exp(eps_target) - 1) / gamma)
    return d / eps_mechanism


def dlap_params_for_scale(target_scale):
    """
    Choose s, t integers such that t/s ≈ target_scale.
    We fix s=1 and round t to nearest integer (minimum t=2).
    """
    s = 1
    t = max(2, round(target_scale))
    return s, t


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_data(n=200, d=20, p=0.3, seed=42):
    np.random.seed(seed)
    return np.random.binomial(1, p, size=(n, d))


# =============================================================================
# POISSON SUBSAMPLING
# =============================================================================

def poisson_subsample(data, gamma, rng=None):
    """
    Include each row independently with probability gamma.
    Returns subsampled data (variable size).
    """
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.random(len(data)) < gamma
    sub  = data[mask]
    # return at least one row to avoid empty sum edge case
    if len(sub) == 0:
        sub = data[[0]]
    return sub


# =============================================================================
# SINGLE-RUN DP SUM: all three mechanisms
# =============================================================================

def run_opendp(data, scale, d, n, gamma):
    ts     = np.sum(data, axis=0).astype(float) / gamma  # rescale
    domain = vector_domain(
        atom_domain(T=float, bounds=(0., float(n)), nan=False), size=d
    )
    metric = l1_distance(float)
    meas   = make_laplace(domain, metric, scale)
    noisy  = np.array(meas(ts))
    flips  = 64 * d
    return noisy, flips


def run_huffman(data, root, d, gamma):
    ts          = np.sum(data, axis=0).astype(float) / gamma  # rescale
    noise       = []
    total_flips = 0
    for _ in range(d):
        s, f = huffman_sample_fair(root)
        noise.append(s)
        total_flips += f
    return ts + np.array(noise), total_flips


def run_dlap(data, s_param, t_param, d, gamma):
    ts           = np.sum(data, axis=0).astype(float) / gamma  # rescale
    noise, flips = dlap_batch(d, s_param, t_param)
    return ts + noise.astype(float), int(flips.sum())


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(
    gamma_values  = [1.0, 0.5, 0.25, 0.1],
    eps_target    = 1.0,
    n             = 200,
    d             = 20,
    p_data        = 0.3,
    n_runs        = 100,
    seed          = 42,
):
    """
    Run both framings across multiple gamma values.

    Returns results dict for plotting.
    """
    rng  = np.random.default_rng(seed)
    data = generate_data(n=n, d=d, p=p_data, seed=seed)
    true_sums = np.sum(data, axis=0).astype(float)

    # ── storage ──────────────────────────────────────────────────────────────
    # framing_a: fixed noise (scale = d/eps_target), tighter privacy
    # framing_b: fixed privacy (eps' = eps_target), less noise

    results = {
        'gamma'      : gamma_values,
        'true_sums'  : true_sums,
        'framing_a'  : {},   # keyed by gamma
        'framing_b'  : {},
    }

    for gamma in gamma_values:
        print(f"\n{'='*60}")
        print(f"  gamma = {gamma}")
        print(f"{'='*60}")

        # ── Framing A: fixed noise scale ──────────────────────────────────
        scale_a     = d / eps_target           
        eps_eff_a   = amplified_epsilon(eps_target, gamma)
        xs_a, pr_a  = build_px(scale_a)
        root_a      = build_huffman_tree(xs_a, pr_a)
        s_a, t_a    = dlap_params_for_scale(scale_a)
        H_a         = -np.sum(pr_a * np.log2(pr_a + 1e-300))

        print(f"  [A] scale={scale_a:.2f}  eps_eff={eps_eff_a:.4f}  "
              f"H={H_a:.3f}  dlap s={s_a} t={t_a}")

        runs_a = {'opendp': [], 'huffman': [], 'dlap': [],
                  'flips_opendp': [], 'flips_huffman': [], 'flips_dlap': []}

        for _ in range(n_runs):
            sub = poisson_subsample(data, gamma, rng=rng)

            noisy_o, fl_o = run_opendp(sub, scale_a, d, n, gamma)
            noisy_h, fl_h = run_huffman(sub, root_a, d, gamma)
            noisy_d, fl_d = run_dlap(sub, s_a, t_a, d, gamma)

            runs_a['opendp'].append(noisy_o)
            runs_a['huffman'].append(noisy_h)
            runs_a['dlap'].append(noisy_d)
            runs_a['flips_opendp'].append(fl_o)
            runs_a['flips_huffman'].append(fl_h)
            runs_a['flips_dlap'].append(fl_d)

        for k in runs_a:
            runs_a[k] = np.array(runs_a[k])

        runs_a['scale']   = scale_a
        runs_a['eps_eff'] = eps_eff_a
        runs_a['H']       = H_a
        results['framing_a'][gamma] = runs_a

        # ── Framing B: fixed privacy target, adjusted scale ───────────────
        scale_b    = scale_for_target_epsilon(eps_target, gamma, d)
        xs_b, pr_b = build_px(scale_b)
        root_b     = build_huffman_tree(xs_b, pr_b)
        s_b, t_b   = dlap_params_for_scale(scale_b)
        H_b        = -np.sum(pr_b * np.log2(pr_b + 1e-300))

        print(f"  [B] scale={scale_b:.2f}  eps_eff={eps_target:.4f}  "
              f"H={H_b:.3f}  dlap s={s_b} t={t_b}")

        runs_b = {'opendp': [], 'huffman': [], 'dlap': [],
                  'flips_opendp': [], 'flips_huffman': [], 'flips_dlap': []}

        for _ in range(n_runs):
            sub = poisson_subsample(data, gamma, rng=rng)

            noisy_o, fl_o = run_opendp(sub, scale_b, d, n, gamma)
            noisy_h, fl_h = run_huffman(sub, root_b, d, gamma)
            noisy_d, fl_d = run_dlap(sub, s_b, t_b, d, gamma)

            runs_b['opendp'].append(noisy_o)
            runs_b['huffman'].append(noisy_h)
            runs_b['dlap'].append(noisy_d)
            runs_b['flips_opendp'].append(fl_o)
            runs_b['flips_huffman'].append(fl_h)
            runs_b['flips_dlap'].append(fl_d)

        for k in runs_b:
            runs_b[k] = np.array(runs_b[k])

        runs_b['scale']   = scale_b
        runs_b['eps_eff'] = eps_target
        runs_b['H']       = H_b
        results['framing_b'][gamma] = runs_b

    return results


# =============================================================================
# PLOTTING
# =============================================================================

COLOURS = {
    'opendp'  : '#E05C5C',
    'huffman' : '#378ADD',
    'dlap'    : '#1D9E75',
}

LABELS = {
    'opendp'  : 'OpenDP Laplace',
    'huffman' : 'Huffman Laplace',
    'dlap'    : 'Discrete Laplace (CKS)',
}


def _envelope_plot(ax, true_sums, runs_dict, gamma, framing_label, show_legend=True):
    """
    Draw trajectory envelope for all three mechanisms on a single axis.
    """
    coords = np.arange(len(true_sums))

    ax.plot(coords, true_sums,
            color='steelblue', linewidth=2.2, marker='o', markersize=4,
            label='True Sum', zorder=10)

    for mech in ['opendp', 'huffman', 'dlap']:
        runs  = runs_dict[mech]           # shape (n_runs, d)
        mean  = runs.mean(axis=0)
        std   = runs.std(axis=0)
        col   = COLOURS[mech]
        label = LABELS[mech]

        for i, run in enumerate(runs):
            ax.plot(coords, run, color=col, alpha=0.08, linewidth=0.6,
                    label=label + ' runs' if i == 0 else None)

        ax.plot(coords, mean, color=col, linewidth=1.8,
                linestyle='--', marker='^', markersize=3,
                label=label + ' mean')

        ax.fill_between(coords, mean - std, mean + std,
                        alpha=0.15, color=col)

    ax.set_title(f'{framing_label}  |  γ = {gamma}', fontsize=10)
    ax.set_xlabel('Coordinate', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    if show_legend:
        ax.legend(fontsize=7, ncol=2)


def plot_accuracy_comparison(results, framing='both'):
    """
    Trajectory envelope plots for each gamma, side-by-side framings A and B.
    """
    gammas    = results['gamma']
    true_sums = results['true_sums']
    n_gamma   = len(gammas)

    if framing == 'both':
        fig, axes = plt.subplots(n_gamma, 2,
                                 figsize=(14, 4 * n_gamma))
        fig.suptitle(
            'Accuracy comparison: Privacy Amplification by Subsampling\n'
            'Left: Framing A (fixed noise, tighter privacy) | '
            'Right: Framing B (fixed privacy, less noise)',
            fontsize=12
        )

        for row, gamma in enumerate(gammas):
            ra = results['framing_a'][gamma]
            rb = results['framing_b'][gamma]

            _envelope_plot(
                axes[row, 0], true_sums, ra, gamma,
                f'[A] scale={ra["scale"]:.1f}  ε_eff={ra["eps_eff"]:.3f}',
                show_legend=(row == 0)
            )
            _envelope_plot(
                axes[row, 1], true_sums, rb, gamma,
                f'[B] scale={rb["scale"]:.1f}  ε_eff={rb["eps_eff"]:.3f}',
                show_legend=(row == 0)
            )

    plt.tight_layout()
    plt.savefig('subsampling_accuracy.png', dpi=150, bbox_inches='tight')
    print(os.getcwd())
    plt.show()
    print('Figure saved: subsampling_accuracy.png')


def plot_randomness_vs_gamma(
    results,
    eps_target = 1.0,
    d          = 20,
    gamma_fine = None,
):
    """
    Per-sample flip cost vs gamma for each mechanism.
    Combines:
      - analytical H(p_gamma) curves across a fine gamma grid
      - empirical mean flips at the evaluated gamma points (dots)

    One panel only (as requested).
    """
    if gamma_fine is None:
        gamma_fine = np.linspace(0.05, 1.0, 80)

    # ── analytical curves ─────────────────────────────────────────────────────
    # Framing A: scale fixed, H fixed
    scale_a = d / eps_target
    H_a     = entropy_of_scale(scale_a)

    # Framing B: scale shrinks with gamma => H shrinks too
    H_b_analytical = []
    for g in gamma_fine:
        sc = scale_for_target_epsilon(eps_target, g, d)
        H_b_analytical.append(entropy_of_scale(sc))
    H_b_analytical = np.array(H_b_analytical)

    # ── empirical dots from sampled gamma values ──────────────────────────────
    gammas_eval = results['gamma']

    emp_a = {mech: [] for mech in ['opendp', 'huffman', 'dlap']}
    emp_b = {mech: [] for mech in ['opendp', 'huffman', 'dlap']}

    for gamma in gammas_eval:
        ra = results['framing_a'][gamma]
        rb = results['framing_b'][gamma]
        for mech in ['opendp', 'huffman', 'dlap']:
            # per-sample cost = total flips / d
            emp_a[mech].append(ra[f'flips_{mech}'].mean() / d)
            emp_b[mech].append(rb[f'flips_{mech}'].mean() / d)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Per-sample randomness vs subsampling rate γ  |  ε_target = {eps_target},  d = {d}',
        fontsize=12
    )

    panel_data = [
        (axes[0],
         'Framing A: fixed noise scale (λ = {:.0f})\nTighter privacy, same randomness'.format(scale_a),
         H_a * np.ones_like(gamma_fine),   # flat line: H doesn't change
         emp_a),
        (axes[1],
         'Framing B: fixed privacy target (ε\' = {:.1f})\nLess noise as γ decreases'.format(eps_target),
         H_b_analytical,
         emp_b),
    ]

    for ax, title, H_curve, emp in panel_data:

        # ── analytical entropy curve for Huffman (= H) ────────────────────
        ax.plot(gamma_fine, H_curve,
                color=COLOURS['huffman'], linewidth=2, linestyle='-',
                label='H(p) analytical (Huffman lower bound)')

        # ── analytical 2H upper bound ─────────────────────────────────────
        ax.plot(gamma_fine, 2 * H_curve,
                color=COLOURS['huffman'], linewidth=1.5, linestyle='--',
                alpha=0.5, label='2H(p) analytical (Huffman upper bound)')

        # ── OpenDP flat line (64 bits regardless of gamma) ────────────────
        ax.axhline(64, color=COLOURS['opendp'], linewidth=2,
                   linestyle='-', label='OpenDP (64 bits/sample, fixed)')

        # ── empirical dots ────────────────────────────────────────────────
        for mech in ['opendp', 'huffman', 'dlap']:
            ax.scatter(gammas_eval,
                       emp[mech],
                       color=COLOURS[mech], s=60, zorder=5,
                       marker='o' if mech == 'huffman' else
                               's' if mech == 'opendp' else 'D',
                       label=f'{LABELS[mech]} (observed)')

        ax.set_xlabel('Subsampling rate γ', fontsize=10)
        ax.set_ylabel('Mean fair bits per sample', fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(0, 1.05)

    plt.tight_layout()
    plt.savefig('subsampling_randomness.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Figure saved: subsampling_randomness.png')


def print_summary_table(results, d=20):
    """
    Print MAE and mean flips per sample for each mechanism,
    gamma, and framing.
    """
    true_sums = results['true_sums']

    print('\n' + '=' * 80)
    print('  Summary: MAE and mean fair bits per sample')
    print('=' * 80)
    print(f"  {'Framing':<4}  {'γ':>5}  {'scale':>7}  {'ε_eff':>7}  "
          f"{'Mech':<20}  {'MAE':>8}  {'bits/sample':>12}")
    print('  ' + '-' * 76)

    for framing_key, framing_label in [('framing_a', 'A'), ('framing_b', 'B')]:
        for gamma in results['gamma']:
            r = results[framing_key][gamma]
            for mech in ['opendp', 'huffman', 'dlap']:
                runs = r[mech]
                mae  = np.abs(runs - true_sums).mean()
                bits = r[f'flips_{mech}'].mean() / d
                print(f"  {framing_label:<4}  {gamma:>5.2f}  "
                      f"{r['scale']:>7.2f}  {r['eps_eff']:>7.4f}  "
                      f"{LABELS[mech]:<20}  {mae:>8.3f}  {bits:>12.3f}")
        print('  ' + '-' * 76)

    print('=' * 80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    # ── parameters ───────────────────────────────────────────────────────────
    EPS_TARGET   = 1.0
    D            = 20
    N            = 200
    P_DATA       = 0.3
    N_RUNS       = 100
    SEED         = 42
    GAMMA_VALUES = [1.0, 0.5, 0.25, 0.1]

    print(f'\nParameters:')
    print(f'  eps_target = {EPS_TARGET}')
    print(f'  d          = {D}')
    print(f'  n          = {N}')
    print(f'  n_runs     = {N_RUNS}')
    print(f'  gammas     = {GAMMA_VALUES}')
    print(f'\nPrivacy amplification (Poisson subsampling):')
    for g in GAMMA_VALUES:
        eps_eff  = amplified_epsilon(EPS_TARGET, g)
        scale_b  = scale_for_target_epsilon(EPS_TARGET, g, D)
        s_b, t_b = dlap_params_for_scale(scale_b)
        print(f'  gamma={g:.2f}  '
              f'[A] eps_eff={eps_eff:.4f}  '
              f'[B] scale={scale_b:.2f}  dlap s={s_b} t={t_b}')

    # ── run experiment ────────────────────────────────────────────────────────
    print('\nRunning experiment (this may take a few minutes)...')
    results = run_experiment(
        gamma_values = GAMMA_VALUES,
        eps_target   = EPS_TARGET,
        n            = N,
        d            = D,
        p_data       = P_DATA,
        n_runs       = N_RUNS,
        seed         = SEED,
    )

    # ── summary table ─────────────────────────────────────────────────────────
    print_summary_table(results, d=D)

    # ── accuracy plot ─────────────────────────────────────────────────────────
    print('\nPlotting accuracy comparison...')
    plot_accuracy_comparison(results, framing='both')

    # ── randomness vs gamma plot ──────────────────────────────────────────────
    print('\nPlotting randomness vs gamma...')
    plot_randomness_vs_gamma(
        results,
        eps_target = EPS_TARGET,
        d          = D,
    )