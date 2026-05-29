"""
=============================================================================
DIMENSIONALITY SCALING EXPERIMENT
=============================================================================

Sweeps d ∈ {5, 10, 20, 50, 100, 200, 500, 1000} with n = 10 * d.

Fixed settings:
  epsilon  = 1.0
  seed     = 42
  n_runs   = 10  (per mechanism × d point)
  p_data   = 0.3 (Bernoulli)

Mechanisms:
  1. OpenDP Laplace       (PRNG, 64 bits/coord)
  2. Huffman Laplace      (fair coins, raw discretisation)
  3. Discrete Laplace     (CKS fair coins)
  4. Shifted Laplace      (Algorithm 4.5, PRNG, ceil(log2(d)) bits/coord)

Metrics recorded per (mechanism, d) point:
  - mean_bits_per_coord   : mean fair bits consumed / d across runs
  - std_bits_per_coord    : std of above
  - mean_runtime_s        : mean wall-clock seconds per run
  - std_runtime_s         : std of above

WARNING — runtime expectations at n_runs=10:
  Huffman and DLap use fair-coin samplers; they are O(H·d) per run.
  At d=1000, eps=1.0: scale = d/eps = 1000, H ≈ large → very slow.
  Mech5 and OpenDP use PRNG and will be fast throughout.
  Consider skipping Huffman/DLap at d ≥ 500 if runtime is prohibitive
  (see SKIP_SLOW_AT_D constant below).

Output:
  scaling_results.csv  — one row per (mechanism, d) with all metrics
  Console              — progress + a formatted summary table

=============================================================================
"""

import heapq
import math
import random
import time
import csv
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from fractions import Fraction

import opendp.prelude as dp
dp.enable_features("contrib", "floating-point")
from opendp.domains import vector_domain, atom_domain
from opendp.metrics import l1_distance
from opendp.measurements import make_laplace


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

D_VALUES        = [5, 10, 20, 50, 100, 200, 500, 1000]
EPSILON         = 1.0
N_PER_D         = lambda d: 10 * d      # n grows with d
N_RUNS          = 10
SEED            = 42
P_DATA          = 0.3

# Shifted Laplace fixed hyperparameters (from epsilon experiment)
M_MECH5         = 25
S_MECH5         = 20

# Fair-coin mechanisms (Huffman, DLap) are slow at large d.
# Set to None to always run them, or set a cutoff e.g. 200 to skip above it.
SKIP_SLOW_AT_D  = None   # e.g. SKIP_SLOW_AT_D = 200

OUTPUT_CSV      = 'scaling_results.csv'


# =============================================================================
# RANDOM PRIMITIVES  
# =============================================================================

def fair_coin():
    return random.randint(0, 1)


def count_flips_biased_coin(p, max_bits=64):
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
    trunc = np.ceil(stats.laplace.ppf((1 + tail_prob) / 2, scale=scale))
    xs    = np.arange(-trunc, trunc + grid_step, grid_step)
    probs = (
        stats.laplace.cdf(xs + grid_step / 2, scale=scale) -
        stats.laplace.cdf(xs - grid_step / 2, scale=scale)
    )
    probs = probs / probs.sum()
    mask  = probs > 0

    idx   = np.argsort(probs)[::-1][:21]
    idx   = np.sort(idx)          # restore original order 
    xs_r  = xs[idx]
    pr_r  = probs[idx] / probs[idx].sum()
    return xs_r, pr_r


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
    return 1, max(2, round(scale))


# =============================================================================
# SHIFTED LAPLACE MECHANISM 
# =============================================================================

def floor_mod(v, m, s):
    ms = m * s
    return math.floor(v / ms) * ms


def sample_discrete_laplace_fast(scale):
    p  = math.exp(-1.0 / scale)
    g1 = np.random.geometric(1 - p) - 1
    g2 = np.random.geometric(1 - p) - 1
    return int(g1 - g2)


def sample_laplace_lt_m(scale, m):
    while True:
        eta = sample_discrete_laplace_fast(scale)
        if abs(eta) < m:
            return eta


def sample_laplace_geq_m(scale, m):
    p    = math.exp(-1.0 / scale)
    sign = random.choice([-1, 1])
    tail = int(np.random.geometric(1 - p) - 1)
    return sign * (m + tail)


def mechanism5(x, eps, m, s_param, d):
    x       = np.asarray(x, dtype=int)
    n, d_   = x.shape
    eps_d   = eps / d_

    p_J     = 2 * math.exp(-eps_d * (m - 1)) / (math.exp(eps_d) + 1)
    p_J     = min(max(p_J, 0.0), 1.0)

    t_val   = int(np.random.binomial(d_, p_J))
    J       = set(np.random.choice(d_, size=t_val, replace=False).tolist())

    omega   = int(random.randint(1, s_param)) * m
    col_sum = x.sum(axis=0)
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

    bits_used = math.ceil(math.log2(max(d_, 2))) * d_
    return np.array(y, dtype=float), bits_used


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_sample_data(n, d, p=0.3, seed=42):
    rng = np.random.RandomState(seed)
    return rng.binomial(1, p, size=(n, d))


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
    return noisy, 64 * d


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
# CORE SCALING LOOP
# =============================================================================

def run_scaling_experiment():
    random.seed(SEED)
    np.random.seed(SEED)

    # Determine which mechanisms run at each d
    # slow_mechs = Huffman + DLap; they use fair-coin loops and are O(H·d)
    slow_mechs = {'huffman', 'dlap'}

    all_rows = []   # collected for CSV

    print('=' * 70)
    print(f'  Dimensionality scaling  ε={EPSILON}  n=10·d  '
          f'n_runs={N_RUNS}  seed={SEED}')
    if SKIP_SLOW_AT_D is not None:
        print(f'  Huffman/DLap skipped for d > {SKIP_SLOW_AT_D}')
    print('=' * 70)

    for d in D_VALUES:
        n     = N_PER_D(d)
        scale = d / EPSILON

        # Pre-build shared structures for this d
        s_d, t_d  = dlap_params_for_scale(scale)
        xs, probs = build_px(scale)
        root      = build_huffman_tree(xs, probs)
        H         = -np.sum(probs * np.log2(probs + 1e-300))

        data       = generate_sample_data(n=n, d=d, p=P_DATA, seed=SEED)
        true_sums  = data.sum(axis=0).astype(float)

        skip_slow = (SKIP_SLOW_AT_D is not None and d > SKIP_SLOW_AT_D)

        print(f'\n  d={d:<5}  n={n:<6}  scale={scale:.1f}  '
              f'H={H:.3f}  dlap s={s_d} t={t_d}  '
              f'symbols={len(xs)}')
        if skip_slow:
            print(f'    [Huffman + DLap skipped at d={d}]')

        # Accumulate per-mechanism per-run metrics
        run_data = {
            'opendp'  : {'bits': [], 'time': []},
            'huffman' : {'bits': [], 'time': []},
            'dlap'    : {'bits': [], 'time': []},
            'mech5'   : {'bits': [], 'time': []},
        }

        for run_i in range(N_RUNS):
            # --- OpenDP ---
            t0 = time.perf_counter()
            _, fo = run_opendp(true_sums, scale, d, n)
            run_data['opendp']['time'].append(time.perf_counter() - t0)
            run_data['opendp']['bits'].append(fo / d)

            # --- Huffman ---
            if not skip_slow:
                t0 = time.perf_counter()
                _, fh = run_huffman(true_sums, root, d)
                run_data['huffman']['time'].append(time.perf_counter() - t0)
                run_data['huffman']['bits'].append(fh / d)

            # --- Discrete Laplace ---
            if not skip_slow:
                t0 = time.perf_counter()
                _, fd = run_dlap(true_sums, s_d, t_d, d)
                run_data['dlap']['time'].append(time.perf_counter() - t0)
                run_data['dlap']['bits'].append(fd / d)

            # --- Shifted Laplace (Mech 5) ---
            t0 = time.perf_counter()
            _, fm = run_mech5(data, EPSILON, M_MECH5, S_MECH5, d)
            run_data['mech5']['time'].append(time.perf_counter() - t0)
            run_data['mech5']['bits'].append(fm / d)

            if (run_i + 1) % 5 == 0:
                print(f'    run {run_i+1}/{N_RUNS} done')

        # --- Print per-d table ---
        print(f'\n    {"Mechanism":<28}  '
              f'{"bits/coord (mean±std)":>22}  '
              f'{"time/run s (mean±std)":>22}')
        print(f'    {"-"*28}  {"-"*22}  {"-"*22}')

        mech_labels = {
            'opendp'  : 'OpenDP Laplace',
            'huffman' : 'Huffman Laplace',
            'dlap'    : 'Discrete Laplace (CKS)',
            'mech5'   : 'Shifted Laplace',
        }

        for mech in ['opendp', 'huffman', 'dlap', 'mech5']:
            b = run_data[mech]['bits']
            t = run_data[mech]['time']

            if not b:   # skipped
                b_mean = b_std = t_mean = t_std = float('nan')
                marker = ' [skipped]'
            else:
                b_arr  = np.array(b)
                t_arr  = np.array(t)
                b_mean, b_std = b_arr.mean(), b_arr.std()
                t_mean, t_std = t_arr.mean(), t_arr.std()
                marker = ''

            bits_str = (f'{b_mean:.3f} ± {b_std:.3f}'
                        if not math.isnan(b_mean) else 'N/A')
            time_str = (f'{t_mean:.4f} ± {t_std:.4f}'
                        if not math.isnan(t_mean) else 'N/A')

            print(f'    {mech_labels[mech] + marker:<28}  '
                  f'{bits_str:>22}  {time_str:>22}')

            all_rows.append({
                'epsilon'          : EPSILON,
                'd'                : d,
                'n'                : n,
                'scale'            : scale,
                'H'                : H,
                'mechanism'        : mech,
                'mean_bits_coord'  : b_mean,
                'std_bits_coord'   : b_std,
                'mean_runtime_s'   : t_mean,
                'std_runtime_s'    : t_std,
                'skipped'          : (len(b) == 0),
            })

    return all_rows


# =============================================================================
# CSV WRITER
# =============================================================================

def save_csv(rows, path=OUTPUT_CSV):
    if not rows:
        print('No data to save.')
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nResults saved to: {path}')


# =============================================================================
# FINAL SUMMARY TABLE
# =============================================================================

def print_summary(rows):
    print('\n' + '=' * 90)
    print('  SCALING SUMMARY  (ε=1.0, n=10·d, 10 runs)')
    print('=' * 90)
    print(f'  {"d":>6}  {"n":>6}  {"scale":>7}  '
          f'{"Mechanism":<28}  {"bits/coord":>12}  {"time/run (s)":>14}')
    print('  ' + '-' * 86)

    prev_d = None
    for row in rows:
        if row['d'] != prev_d and prev_d is not None:
            print('  ' + '-' * 86)
        prev_d = row['d']

        if row['skipped']:
            bits_s = 'N/A'
            time_s = 'N/A'
        else:
            bits_s = f'{row["mean_bits_coord"]:.3f}'
            time_s = f'{row["mean_runtime_s"]:.4f}'

        mech_labels = {
            'opendp'  : 'OpenDP Laplace',
            'huffman' : 'Huffman Laplace',
            'dlap'    : 'Discrete Laplace (CKS)',
            'mech5'   : 'Shifted Laplace',
        }
        print(f'  {row["d"]:>6}  {row["n"]:>6}  {row["scale"]:>7.1f}  '
              f'{mech_labels[row["mechanism"]]:<28}  '
              f'{bits_s:>12}  {time_s:>14}')

    print('=' * 90)
    print(f'\n  Note: Shifted Laplace bits/coord = ceil(log2(d)) — grows as O(log d).')
    print(f'  Note: OpenDP bits/coord = 64 (flat, PRNG).')
    print(f'  Note: Huffman bits/coord ≈ H(p) to 2H(p); '
          f'grows with scale (= d/ε), so grows with d.')
    print(f'  Note: DLap bits/coord depends on t = round(d/ε); '
          f'also grows with d.')


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


def _rows_to_arrays(rows):
    """
    Convert flat list of row dicts into per-mechanism arrays indexed by d.
    Returns dict: mech -> {'d', 'bits_mean', 'bits_std', 'time_mean', 'time_std'}
    """
    out = {}
    for mech in MECHS:
        mech_rows = [r for r in rows if r['mechanism'] == mech]
        mech_rows.sort(key=lambda r: r['d'])
        out[mech] = {
            'd'         : np.array([r['d']               for r in mech_rows]),
            'bits_mean' : np.array([r['mean_bits_coord']  for r in mech_rows], dtype=float),
            'bits_std'  : np.array([r['std_bits_coord']   for r in mech_rows], dtype=float),
            'time_mean' : np.array([r['mean_runtime_s']   for r in mech_rows], dtype=float),
            'time_std'  : np.array([r['std_runtime_s']    for r in mech_rows], dtype=float),
            'skipped'   : np.array([r['skipped']          for r in mech_rows], dtype=bool),
        }
    return out


def _mask_valid(arr, skipped):
    """Return array with NaN wherever skipped=True."""
    out = arr.astype(float).copy()
    out[skipped] = np.nan
    return out


# -----------------------------------------------------------------------------
# (a) Bits/coord vs d  — log-log axes with theoretical reference lines
# -----------------------------------------------------------------------------

def plot_bits_vs_d(rows):
    data = _rows_to_arrays(rows)
    d_all = data['opendp']['d']

    fig, ax = plt.subplots(figsize=(9, 5))

    for mech in MECHS:
        d_arr   = data[mech]['d']
        bm      = _mask_valid(data[mech]['bits_mean'], data[mech]['skipped'])
        bs      = _mask_valid(data[mech]['bits_std'],  data[mech]['skipped'])
        valid   = ~np.isnan(bm)
        c       = COLOURS[mech]

        ax.plot(d_arr[valid], bm[valid],
                color=c, linewidth=2, marker='o', markersize=5,
                label=LABELS[mech])
        ax.fill_between(d_arr[valid],
                        np.maximum(bm[valid] - bs[valid], 1e-3),
                        bm[valid] + bs[valid],
                        color=c, alpha=0.12)

    # Theoretical reference lines
    d_ref  = np.array(sorted(set(d_all)))
    d_fine = np.logspace(np.log10(d_ref.min()), np.log10(d_ref.max()), 200)

    # Shifted Laplace: ceil(log2(d))
    log2d_ref = np.array([math.ceil(math.log2(max(int(d), 2))) for d in d_fine])
    ax.plot(d_fine, log2d_ref,
            color=COLOURS['mech5'], linewidth=1.2, linestyle=':',
            label='⌈log₂(d)⌉ (Shifted Lap theory)')

    # OpenDP: flat 64
    ax.axhline(64, color=COLOURS['opendp'], linewidth=1.2, linestyle=':',
               label='64 (OpenDP PRNG, flat)')

    # Entropy lower bound for Huffman: H grows roughly as log(scale) = log(d/ε)
    # Compute actual H for each d using build_px
    h_vals = []
    for d in d_fine:
        sc    = d / EPSILON
        xs_h, pr_h = build_px(sc)
        H     = -np.sum(pr_h * np.log2(pr_h + 1e-300))
        h_vals.append(H)
    h_vals = np.array(h_vals)
    ax.plot(d_fine, h_vals,
            color=COLOURS['huffman'], linewidth=1.2, linestyle=':',
            label='H(p) entropy lower bound (Huffman theory)')
    ax.plot(d_fine, 2 * h_vals,
            color=COLOURS['huffman'], linewidth=1.0, linestyle='-.',
            alpha=0.5, label='2H(p) upper bound (Huffman theory)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dimensionality d', fontsize=11)
    ax.set_ylabel('Mean fair bits per coordinate', fontsize=11)
    ax.set_title(
        'Randomness cost vs dimensionality  (ε=1.0, n=10·d)\n'
        'Log-log axes — slope reveals asymptotic scaling class',
        fontsize=11
    )
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(d_ref)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('scaling_bits_vs_d.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: scaling_bits_vs_d.png')


# -----------------------------------------------------------------------------
# (b) Wall-clock runtime vs d — log-log axes
# -----------------------------------------------------------------------------

def plot_runtime_vs_d(rows):
    data  = _rows_to_arrays(rows)
    d_all = data['opendp']['d']

    fig, ax = plt.subplots(figsize=(9, 5))

    for mech in MECHS:
        d_arr  = data[mech]['d']
        tm     = _mask_valid(data[mech]['time_mean'], data[mech]['skipped'])
        ts     = _mask_valid(data[mech]['time_std'],  data[mech]['skipped'])
        valid  = ~np.isnan(tm)
        c      = COLOURS[mech]

        ax.plot(d_arr[valid], tm[valid],
                color=c, linewidth=2, marker='o', markersize=5,
                label=LABELS[mech])
        ax.fill_between(d_arr[valid],
                        np.maximum(tm[valid] - ts[valid], 1e-9),
                        tm[valid] + ts[valid],
                        color=c, alpha=0.12)

    # O(d) and O(d log d) reference lines — anchored to OpenDP at d=20
    anchor_d  = 20
    anchor_t  = _mask_valid(
        data['opendp']['time_mean'],
        data['opendp']['skipped']
    )[data['opendp']['d'] == anchor_d][0]

    d_fine = np.logspace(
        np.log10(d_all.min()), np.log10(d_all.max()), 200
    )
    ax.plot(d_fine,
            anchor_t * (d_fine / anchor_d),
            color='grey', linewidth=1.0, linestyle=':',
            label='O(d) reference')
    ax.plot(d_fine,
            anchor_t * (d_fine / anchor_d) * np.log2(d_fine / anchor_d + 1),
            color='grey', linewidth=1.0, linestyle='-.',
            label='O(d log d) reference')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dimensionality d', fontsize=11)
    ax.set_ylabel('Mean runtime per run (in seconds)', fontsize=11)
    ax.set_title(
        'Runtime vs dimensionality  (ε=1.0, n=10·d)\n',
        fontsize=11
    )
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(d_all)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('scaling_runtime_vs_d.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: scaling_runtime_vs_d.png')


# -----------------------------------------------------------------------------
# (c) Bits/coord ratio: empirical / theoretical — deviation from ideal
# -----------------------------------------------------------------------------

def plot_bits_ratio(rows):
    """
    For each mechanism, plot empirical bits / theoretical target:
      OpenDP  : empirical / 64            (should be flat = 1)
      Huffman : empirical / H(p)          (should be in [1, 2])
      DLap    : empirical / H(p)          (fair-coin baseline)
      Mech5   : empirical / ceil(log2(d)) (should be flat = 1)
    """
    data  = _rows_to_arrays(rows)
    d_all = data['opendp']['d']

    fig, ax = plt.subplots(figsize=(9, 5))

    for mech in MECHS:
        d_arr  = data[mech]['d']
        bm     = _mask_valid(data[mech]['bits_mean'], data[mech]['skipped'])
        valid  = ~np.isnan(bm)
        c      = COLOURS[mech]

        ratios = []
        for d_val, b_val in zip(d_arr[valid], bm[valid]):
            if mech == 'opendp':
                theory = 64.0
            elif mech == 'mech5':
                theory = float(math.ceil(math.log2(max(int(d_val), 2))))
            else:
                # H(p) for this d
                sc       = d_val / EPSILON
                xs_h, pr_h = build_px(sc)
                theory   = -np.sum(pr_h * np.log2(pr_h + 1e-300))
            ratios.append(b_val / theory if theory > 0 else np.nan)

        ratios = np.array(ratios)
        ax.plot(d_arr[valid], ratios,
                color=c, linewidth=2, marker='o', markersize=5,
                label=LABELS[mech])

    ax.axhline(1.0, color='black', linewidth=1.2, linestyle='--',
               label='Ratio = 1 (exact theory)')
    ax.axhline(2.0, color='grey', linewidth=1.0, linestyle=':',
               label='Ratio = 2 (Huffman upper bound)')

    ax.set_xscale('log')
    ax.set_xlabel('Dimensionality d', fontsize=11)
    ax.set_ylabel('Empirical bits / theoretical target', fontsize=11)
    ax.set_title(
        'Efficiency ratio: empirical randomness cost vs theoretical target\n'
        'OpenDP & Shifted Lap → 1.0  |  Huffman/DLap → between 1 and 2',
        fontsize=11
    )
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(d_all)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('scaling_bits_ratio.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: scaling_bits_ratio.png')


# -----------------------------------------------------------------------------
# (d) Combined 2×2 dashboard
# -----------------------------------------------------------------------------

def plot_dashboard(rows):
    data  = _rows_to_arrays(rows)
    d_all = data['opendp']['d']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Dimensionality scaling — all mechanisms  (ε=1.0, n=10·d)',
        fontsize=13, y=1.01
    )

    # ── top-left: bits/coord (linear axes) ──────────────────────────────────
    ax = axes[0, 0]
    for mech in MECHS:
        d_arr  = data[mech]['d']
        bm     = _mask_valid(data[mech]['bits_mean'], data[mech]['skipped'])
        bs     = _mask_valid(data[mech]['bits_std'],  data[mech]['skipped'])
        valid  = ~np.isnan(bm)
        c      = COLOURS[mech]
        ax.plot(d_arr[valid], bm[valid], color=c, linewidth=2,
                marker='o', markersize=4, label=LABELS[mech])
        ax.fill_between(d_arr[valid],
                        np.maximum(bm[valid] - bs[valid], 0),
                        bm[valid] + bs[valid], color=c, alpha=0.10)
    ax.set_title('Bits/coord vs d  (linear)', fontsize=10)
    ax.set_xlabel('d'); ax.set_ylabel('bits/coord')
    ax.legend(fontsize=7); ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # ── top-right: bits/coord (log-log) ─────────────────────────────────────
    ax = axes[0, 1]
    for mech in MECHS:
        d_arr  = data[mech]['d']
        bm     = _mask_valid(data[mech]['bits_mean'], data[mech]['skipped'])
        valid  = ~np.isnan(bm)
        c      = COLOURS[mech]
        ax.plot(d_arr[valid], bm[valid], color=c, linewidth=2,
                marker='o', markersize=4, label=LABELS[mech])
    # reference lines
    d_fine = np.logspace(np.log10(d_all.min()), np.log10(d_all.max()), 200)
    log2d_ref = np.array([math.ceil(math.log2(max(int(d), 2))) for d in d_fine])
    h_ref = np.array([-np.sum(build_px(d / EPSILON)[1] *
                               np.log2(build_px(d / EPSILON)[1] + 1e-300))
                       for d in d_fine])
    ax.plot(d_fine, log2d_ref, color=COLOURS['mech5'],
            linewidth=1, linestyle=':', label='⌈log₂d⌉ theory')
    ax.plot(d_fine, h_ref, color=COLOURS['huffman'],
            linewidth=1, linestyle=':', label='H(p) theory')
    ax.axhline(64, color=COLOURS['opendp'], linewidth=1, linestyle=':',
               label='64 flat')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title('Bits/coord vs d  (log-log)', fontsize=10)
    ax.set_xlabel('d'); ax.set_ylabel('bits/coord')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks([5, 10, 20, 50, 100, 200, 500, 1000])
    ax.legend(fontsize=7); ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # ── bottom-left: runtime (log-log) ───────────────────────────────────────
    ax = axes[1, 0]
    for mech in MECHS:
        d_arr  = data[mech]['d']
        tm     = _mask_valid(data[mech]['time_mean'], data[mech]['skipped'])
        ts     = _mask_valid(data[mech]['time_std'],  data[mech]['skipped'])
        valid  = ~np.isnan(tm)
        c      = COLOURS[mech]
        ax.plot(d_arr[valid], tm[valid], color=c, linewidth=2,
                marker='o', markersize=4, label=LABELS[mech])
        ax.fill_between(d_arr[valid],
                        np.maximum(tm[valid] - ts[valid], 1e-9),
                        tm[valid] + ts[valid], color=c, alpha=0.10)
    # O(d) guide
    anchor_d = 20
    anchor_t = _mask_valid(data['opendp']['time_mean'],
                            data['opendp']['skipped']
                            )[data['opendp']['d'] == anchor_d][0]
    ax.plot(d_fine, anchor_t * (d_fine / anchor_d),
            color='grey', linewidth=1, linestyle=':', label='O(d) guide')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title('Runtime/run vs d  (log-log)', fontsize=10)
    ax.set_xlabel('d'); ax.set_ylabel('seconds')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks([5, 10, 20, 50, 100, 200, 500, 1000])
    ax.legend(fontsize=7); ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # ── bottom-right: efficiency ratio (empirical / theory) ──────────────────
    ax = axes[1, 1]
    for mech in MECHS:
        d_arr  = data[mech]['d']
        bm     = _mask_valid(data[mech]['bits_mean'], data[mech]['skipped'])
        valid  = ~np.isnan(bm)
        c      = COLOURS[mech]
        ratios = []
        for d_val, b_val in zip(d_arr[valid], bm[valid]):
            if mech == 'opendp':
                theory = 64.0
            elif mech == 'mech5':
                theory = float(math.ceil(math.log2(max(int(d_val), 2))))
            else:
                sc         = d_val / EPSILON
                xs_h, pr_h = build_px(sc)
                theory     = -np.sum(pr_h * np.log2(pr_h + 1e-300))
            ratios.append(b_val / theory if theory > 0 else np.nan)
        ax.plot(d_arr[valid], np.array(ratios), color=c, linewidth=2,
                marker='o', markersize=4, label=LABELS[mech])
    ax.axhline(1.0, color='black', linewidth=1.2, linestyle='--',
               label='= 1 (exact)')
    ax.axhline(2.0, color='grey',  linewidth=1.0, linestyle=':',
               label='= 2 (Huffman bound)')
    ax.set_xscale('log')
    ax.set_title('Efficiency ratio  (empirical / theory)', fontsize=10)
    ax.set_xlabel('d'); ax.set_ylabel('ratio')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks([5, 10, 20, 50, 100, 200, 500, 1000])
    ax.legend(fontsize=7); ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('scaling_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: scaling_dashboard.png')


# =============================================================================
# OVERLAID BIT-COST HISTOGRAM  (d=500, n=5000, 1000 samples per mechanism)
# =============================================================================

HIST_D       = 500
HIST_N       = 5000
HIST_SAMPLES = 1000   # independent single-query draws per mechanism


def _collect_bits_per_coord(d, n_samples, eps, seed):
    """
    Draw n_samples SINGLE-COORDINATE samples for each mechanism and record
    the fair bits used per individual draw.

    Plotting per-coord rather than summed-over-d exposes the true per-draw
    variance. Summing over d collapses everything to a spike via the CLT.

    Mechanisms:
      OpenDP   : always costs exactly 64 bits (PRNG word) — delta spike at 64
      Huffman  : variable; traverses Huffman tree with fair coins each draw
      DLap     : variable; CKS sampler fair-coin cost per draw
      Mech5    : always costs exactly ceil(log2(d)) bits — delta spike

    Returns dict: mech -> np.ndarray of shape (n_samples,), bits per coord.
    """
    random.seed(seed)
    np.random.seed(seed)

    scale     = d / eps
    s_d, t_d  = dlap_params_for_scale(scale)
    xs, probs = build_px(scale)
    root      = build_huffman_tree(xs, probs)
    log2d     = math.ceil(math.log2(max(d, 2)))
    log2dbyd = log2d / d

    bits = {mech: [] for mech in MECHS}

    print(f'  Collecting {n_samples} per-coord samples per mechanism '
          f'(d={d}, scale={scale:.1f}, ε={eps}) ...')

    for i in range(n_samples):
        # OpenDP: 64 bits per coordinate (PRNG word), always
        bits['opendp'].append(64)

        # Huffman: one fair-coin tree traversal per coordinate
        _, fh = huffman_sample_fair(root)
        bits['huffman'].append(fh)

        # Discrete Laplace: one CKS draw per coordinate
        _, fd = sample_discrete_laplace(s_d, t_d)
        bits['dlap'].append(fd)

        # Shifted Laplace: ceil(log2(d)) bits per coordinate (PRNG)
        bits['mech5'].append(log2dbyd)

        if (i + 1) % 200 == 0:
            print(f'    {i+1}/{n_samples} done')

    return {m: np.array(v, dtype=float) for m, v in bits.items()}


def plot_bits_histogram(d=HIST_D, n=HIST_N, n_samples=HIST_SAMPLES,
                        eps=1.0, seed=SEED):
    """
    Overlaid histogram of per-coordinate bit cost across all four mechanisms.

    OpenDP and Shifted Laplace are deterministic (delta spikes); Huffman and
    DLap are stochastic and show their true distributions.

    Plotting per-coord (not summed over d) so CLT doesn't wash out variance.
    Two panels:
      Left  — full x-range showing all four mechanisms inc. OpenDP at 64
      Right — zoomed in on the stochastic region (Huffman, DLap, Mech5)
               so the actual distributions are legible
    """
    bits = _collect_bits_per_coord(d, n_samples, eps, seed)

    scale     = d / eps
    xs, probs = build_px(scale)
    H         = -np.sum(probs * np.log2(probs + 1e-300))
    log2d     = math.ceil(math.log2(max(d, 2)))
    logdbyd    = log2d / d

    # theoretical per-coord reference values
    theo_H_lo   = H           # Huffman lower bound per coord
    theo_H_hi   = 2 * H       # Huffman upper bound per coord
    theo_opendp = 64.0        # OpenDP flat
    theo_mech5  = float(log2d/d) # Shifted Lap target

    def _draw_panel(ax, xlim=None, title_suffix=''):
        """Draw the overlaid histogram on ax, optionally clipping x-axis."""
        # stochastic mechanisms get histograms
        for mech in ['huffman', 'dlap']:
            b   = bits[mech]
            c   = COLOURS[mech]
            lo  = int(b.min())
            hi  = int(b.max()) + 2
            bins = range(lo, hi + 1, 1)
            ax.hist(b, bins=bins, density=True, color=c, alpha=0.40,
                    label=LABELS[mech])

        # deterministic mechanisms: tall narrow bars
        spike_width = 0.6
        for mech, xval in [('opendp', theo_opendp), ('mech5', theo_mech5)]:
            c = COLOURS[mech]
            ax.bar([xval], [1.05], width=spike_width, color=c, alpha=0.55,
                   label=f'{LABELS[mech]} (deterministic)')

        # mean verticals for stochastic mechs
        for mech in ['huffman', 'dlap']:
            b = bits[mech]
            c = COLOURS[mech]
            ax.axvline(b.mean(), color=c, linewidth=2.0, linestyle='--',
                       label=f'{LABELS[mech]} mean = {b.mean():.1f}')

        # mean verticals for deterministic mechs
        for mech, xval in [('opendp', theo_opendp), ('mech5', theo_mech5)]:
            c = COLOURS[mech]
            ax.axvline(xval, color=c, linewidth=2.0, linestyle='--',
                       label=f'{LABELS[mech]} = {xval:.0f}')

        # Shannon bound lines
        ax.axvline(theo_H_lo, color='orange', linewidth=1.5, linestyle=':',
                   label=f'H(p) = {theo_H_lo:.2f}  (lower bound)')
        ax.axvline(theo_H_hi, color='darkorange', linewidth=1.5, linestyle=':',
                   label=f'2H(p) = {theo_H_hi:.2f}  (upper bound)')

        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_xlabel('Fair bits per coordinate', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(title_suffix, fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(
        f'Per-coordinate bit-cost distribution — all mechanisms overlaid\n'
        f'd={d}, scale={scale:.0f}, ε={eps}, {n_samples} samples per mechanism',
        fontsize=12
    )

    # left: full view inc. OpenDP spike at 64
    _draw_panel(axes[0],
                xlim=(0, max(70, theo_opendp + 5)),
                title_suffix='Full range')

    # right: zoomed into stochastic region so Huffman/DLap shapes are legible
    stoch_vals = np.concatenate([bits['huffman'], bits['dlap']])
    zoom_lo = max(0, stoch_vals.min() - 2)
    zoom_hi = max(theo_mech5 + 5, stoch_vals.max() + 5)
    _draw_panel(axes[1],
                xlim=(zoom_lo, zoom_hi),
                title_suffix='Zoomed: stochastic region  (Huffman & DLap distributions)')

    plt.tight_layout()
    plt.savefig('bits_histogram_overlay.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: bits_histogram_overlay.png')

    # summary stats
    print(f'\n  {"Mechanism":<28}  {"mean":>8}  {"std":>7}  '
          f'{"min":>6}  {"max":>6}')
    print('  ' + '-' * 62)
    for mech in MECHS:
        b = bits[mech]
        print(f'  {LABELS[mech]:<28}  {b.mean():>8.2f}  {b.std():>7.3f}  '
              f'{b.min():>6.0f}  {b.max():>6.0f}')
    print(f'\n  H(p) = {theo_H_lo:.3f}   2H(p) = {theo_H_hi:.3f}   '
          f'OpenDP = 64   Shifted Lap = {logdbyd}')


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    log2d_vals = {d: math.ceil(math.log2(max(d, 2))) for d in D_VALUES}
    scales     = {d: d / EPSILON for d in D_VALUES}

    print('Dimensionality scaling experiment')
    print(f'  ε={EPSILON}  n=10·d  n_runs={N_RUNS}  seed={SEED}')
    print(f'  Shifted Laplace: m={M_MECH5}, s={S_MECH5}')
    print(f'  SKIP_SLOW_AT_D: {SKIP_SLOW_AT_D}')
    print()
    print(f'  {"d":>6}  {"n":>6}  {"scale":>8}  {"log2(d)":>8}')
    print('  ' + '-' * 36)
    for d in D_VALUES:
        print(f'  {d:>6}  {N_PER_D(d):>6}  {scales[d]:>8.1f}  '
              f'{log2d_vals[d]:>8}')

    print()
    rows = run_scaling_experiment()
    save_csv(rows, OUTPUT_CSV)
    print_summary(rows)

    print('\nPlotting bits/coord vs d ...')
    plot_bits_vs_d(rows)

    print('\nPlotting runtime vs d ...')
    plot_runtime_vs_d(rows)

    print('\nPlotting efficiency ratio ...')
    plot_bits_ratio(rows)

    print('\nPlotting combined dashboard ...')
    plot_dashboard(rows)

    print('\nPlotting overlaid bits-cost histogram (d=500, n=5000, 1000 samples)...')
    print('  WARNING: Huffman + DLap at d=500 scale=500 will be slow.')
    plot_bits_histogram(
        d=HIST_D, n_samples=HIST_SAMPLES, eps=EPSILON, seed=SEED
    )

    print('\nAll plots saved.')