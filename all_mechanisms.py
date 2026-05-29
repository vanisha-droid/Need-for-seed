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


# -----------------------------
# Fair coin + biased coin simulation
# our ONLY source of randomness
# biased coin with probability p simulated using fair coin flips via binary expansion method (von neumann's trick)
# exactly 2 expected fair flips per biased decision regardless of p
# -----------------------------

def fair_coin():
    return random.randint(0, 1)


def count_flips_biased_coin(p, max_bits=64):
    """
    same as biased_coin_fair_flips but also returns how many fair flips were used.
    uses Fraction arithmetic instead of floats to avoid binary expansion drift.
    basically:
      write p in binary: p = 0.b1 b2 b3 ...
      flip fair coins one at a time.
      stop when flip != b_n (disagreement).
      return b_n as the outcome.
    expected flips: 2, regardless of p.
    """
    p_frac = Fraction(p).limit_denominator(10**12)  # exact rational, no float drift
    flips_used = 0

    for _ in range(max_bits):
        p_frac *= 2
        p_bit = 1 if p_frac >= 1 else 0  # exact integer part
        p_frac -= p_bit                  # exact remainder

        flips_used += 1
        flip = fair_coin()

        if flip != p_bit:
            # disagreement — stop here, return p_bit (1 = success, 0 = failure)
            return p_bit, flips_used

    # extremely rare: only if p has > max_bits binary digits
    return random.randint(0, 1), flips_used


# -------------------------------------
# Discretise Laplace -> p(x)
# Huffman needs a finite list of (symbol, probability) pairs and can't work w a continuous curve
# so we slice the Laplace into buckets and ask: what fraction of noise lands in each bucket?
# each bucket = a symbol w a probability
# --------------------------------------

def build_px(scale, grid_step=1.0, tail_prob=0.999):
    """
    Discretize Laplace(0, scale) onto a uniform grid.
    grid_step: spacing between symbols (use 1.0 for unit grid)
    tail_prob: how much probability mass to cover (truncation point)
    """
    # find truncation radius: covers tail_prob of mass
    trunc = np.ceil(stats.laplace.ppf((1 + tail_prob) / 2, scale=scale))
    xs    = np.arange(-trunc, trunc + grid_step, grid_step)

    # integrate PDF over each grid cell [x - step/2, x + step/2]
    probs = (
        stats.laplace.cdf(xs + grid_step / 2, scale=scale) -
        stats.laplace.cdf(xs - grid_step / 2, scale=scale)
    )

    # renormalize to correct for truncation
    probs = probs / probs.sum()

    # filter out zero-probability symbols (numerical underflow at tails)
    mask = probs > 0

    return xs[mask], probs[mask]


# -----------------------------
# Huffman tree
# common symbols (noise near 0) sit close to root = short path = few flips
# rare tail symbols sit deep = long path = more flips, but rarely reached
# -----------------------------

class Node:
    def __init__(self, prob, symbol=None):
        self.prob   = prob
        self.symbol = symbol  # none for internal nodes, float for leaves
        self.left   = None
        self.right  = None

    def __lt__(self, other):
        return self.prob < other.prob


def build_huffman_tree(xs, probs):
    heap = [Node(prob=p, symbol=x) for x, p in zip(xs, probs)]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)  # lowest prob
        hi = heapq.heappop(heap)  # second lowest
        parent = Node(prob=lo.prob + hi.prob)
        parent.left  = lo
        parent.right = hi
        heapq.heappush(heap, parent)
    return heap[0]  # root


def build_codebook(root):
    # walk tree and assign binary codewords; left = '0', right = '1'
    codebook = {}

    def walk(node, code):
        if node.symbol is not None:
            codebook[node.symbol] = code if code else '0'
            return
        walk(node.left,  code + '0')
        walk(node.right, code + '1')

    walk(root, '')
    return codebook

# ---------------------------------------------
# Huffman sampler w fair coin flips
# traverse the tree using biased coin flips
# each biased flip is itself simulated w fair coins via binary expansion (2 expected fair flips each)
# at each internal node:
#   p_left = left.prob / (left.prob + right.prob)
#   flip biased coin(p_left) using fair coins
#   go left if 1, right if 0
# ---------------------------------------------

def huffman_sample_fair(root):
    node = root
    total_fair_flips = 0

    while node.symbol is None:
        p_left = node.left.prob / (node.left.prob + node.right.prob)
        outcome, flips_used = count_flips_biased_coin(p_left)
        total_fair_flips += flips_used
        node = node.left if outcome == 1 else node.right

    return node.symbol, total_fair_flips


def huffman_sample_batch_fair(root, n):
    # draw n independent noise samples
    samples, flip_counts = [], []
    for _ in range(n):
        s, f = huffman_sample_fair(root)
        samples.append(s)
        flip_counts.append(f)
    return np.array(samples), np.array(flip_counts)


# -----------------------------
# Sample data generator
# -----------------------------

def generate_sample_data(n=200, d=20, p=0.3):
    return np.random.binomial(1, p, size=(n, d))


# ------------------------------------------------------------------------------------------
# Probability rounding / coarsening
# Goal: reduce entropy of p(x) -> fewer expected fair flips per sample (H(p) bits of entropy = ~2*H(p) fair flips)
#
# Strategy: merge/zero-out small probability buckets into neighbours or a single "tail" bin
# before Huffman sampling. Lower entropy = shorter average codeword = fewer coin flips.
#
# Five strategies:
#   'none'       – identity, no rounding (baseline)
#   'threshold'  – zero out symbols below prob threshold, renormalise
#                  most aggressive; collapses tail into nothing
#   'topk'       – keep only the k highest-prob symbols, renormalise
#                  useful when you know roughly how many symbols you need
#   'quantile'   – zero out symbols below the q-th cumulative quantile of *mass*
#                  e.g. q=0.01 drops symbols covering the bottom 1% of mass
#   'merge_tail' – merge all sub-threshold symbols into one explicit "TAIL" bin
#                  preserves total mass; Huffman gets one deep leaf for all tails
#   'step'       – coarsen grid by grouping every k consecutive symbols, summing probs
#                  effectively widens the discretisation step without recomputing CDFs
# ------------------------------------------------------------------------------------------

TAIL_SYMBOL = np.inf  # sentinel value for the merged tail bin

def round_px(
    xs: np.ndarray,
    probs: np.ndarray,
    strategy: str = 'none',
    # --- strategy-specific knobs ---
    threshold: float = 1e-4,   # used by 'threshold' and 'merge_tail'
    topk: int       = 51,      # used by 'topk'
    quantile: float = 0.01,    # used by 'quantile' (fraction of mass to drop)
    step: int       = 2,       # used by 'step' (bucket width in original grid units)
    dyadic_mode: str = "nearest",  # used by 'dyadic' (rounding mode: nearest, floor, ceil)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coarsen/round a discrete distribution (xs, probs) before Huffman coding.

    Returns (xs_new, probs_new), always normalised to sum=1.

    Parameters
    ----------
    xs, probs   : output of build_px()
    strategy    : one of 'none', 'threshold', 'topk', 'quantile',
                  'merge_tail', 'step'
    threshold   : minimum probability to keep (strategies: threshold, merge_tail)
    topk        : number of symbols to retain    (strategy: topk)
    quantile    : tail mass fraction to drop     (strategy: quantile)
    step        : merge every `step` consecutive symbols (strategy: step)
    """
    if strategy == 'none':
        return xs.copy(), probs.copy()

    # ------------------------------------------------------------------
    elif strategy == 'threshold':
        # Drop every symbol whose probability is below `threshold`.
        # Residual mass is discarded and probs renormalised.
        # Effect: sharp truncation; can lose noticeable mass if threshold is high.
        mask  = probs >= threshold
        xs_r  = xs[mask]
        pr_r  = probs[mask] / probs[mask].sum()
        return xs_r, pr_r

    # ------------------------------------------------------------------
    elif strategy == 'topk':
        # Retain the k symbols with highest probability; drop the rest.
        # Equivalent to 'threshold' but controlled by count, not prob level.
        idx   = np.argsort(probs)[::-1][:topk]
        idx   = np.sort(idx)          # restore original order (useful for plotting)
        xs_r  = xs[idx]
        pr_r  = probs[idx] / probs[idx].sum()
        return xs_r, pr_r

    # ------------------------------------------------------------------
    elif strategy == 'quantile':
        # Drop symbols that together account for the bottom `quantile` fraction
        # of cumulative mass (sorted by prob ascending).
        # Gentler than 'threshold': always drops exactly the lowest-density tail.
        order     = np.argsort(probs)           # ascending
        cum_mass  = np.cumsum(probs[order])
        keep_mask = np.ones(len(probs), dtype=bool)
        keep_mask[order[cum_mass < quantile]] = False
        xs_r  = xs[keep_mask]
        pr_r  = probs[keep_mask] / probs[keep_mask].sum()
        return xs_r, pr_r

    # ------------------------------------------------------------------
    elif strategy == 'merge_tail':
        # Fold all sub-threshold symbols into a single explicit TAIL_SYMBOL bin.
        # Total probability mass is conserved (unlike 'threshold').
        # Huffman will assign one codeword to the entire tail.
        mask       = probs >= threshold
        tail_mass  = probs[~mask].sum()
        xs_r   = list(xs[mask])
        pr_r   = list(probs[mask])
        if tail_mass > 0:
            xs_r.append(TAIL_SYMBOL)
            pr_r.append(tail_mass)
        xs_r  = np.array(xs_r)
        pr_r  = np.array(pr_r) / np.array(pr_r).sum()   # should be ~1 already
        return xs_r, pr_r

    # ------------------------------------------------------------------
    elif strategy == 'step':
        # Coarsen the grid by summing every `step` consecutive buckets.
        # The representative symbol of each merged bucket is the weighted mean
        # of its constituent symbols.
        # Entropy reduction scales roughly as log2(step) bits in the limit.
        n        = len(probs)
        pad      = (-n) % step          # zero-pad to multiple of `step`
        xs_p     = np.pad(xs.astype(float), (0, pad), constant_values=0.0)
        pr_p     = np.pad(probs,            (0, pad), constant_values=0.0)
        xs_2d    = xs_p.reshape(-1, step)
        pr_2d    = pr_p.reshape(-1, step)
        pr_r     = pr_2d.sum(axis=1)
        # weighted mean position for each merged bucket (avoids arbitrary choice)
        with np.errstate(invalid='ignore'):
            xs_r = np.where(pr_r > 0,
                            (xs_2d * pr_2d).sum(axis=1) / pr_r,
                            0.0)
        mask = pr_r > 0
        xs_r = xs_r[mask]
        pr_r = pr_r[mask] / pr_r[mask].sum()
        return xs_r, pr_r
        # ------------------------------------------------------------------
    elif strategy == 'dyadic':
        """
        Round probabilities to nearby powers of two.

        Why?
        ----
        Dyadic probabilities align naturally with binary trees /
        fair-coin sampling and often reduce randomness complexity.

        Variants:
            mode='nearest' : nearest power of two
            mode='floor'   : largest dyadic <= p
            mode='ceil'    : smallest dyadic >= p

        After rounding we renormalise to sum=1.
        """

        mode = dyadic_mode

        # avoid log2(0)
        p_safe = np.maximum(probs, 1e-300)

        logs = np.log2(p_safe)

        if mode == "nearest":
            logs_r = np.round(logs)

        elif mode == "floor":
            logs_r = np.floor(logs)

        elif mode == "ceil":
            logs_r = np.ceil(logs)

        else:
            raise ValueError(
                f"Unknown dyadic mode: {mode!r}. "
                f"Choose from: nearest, floor, ceil"
            )

        probs_r = 2.0 ** logs_r

        # renormalise
        probs_r = probs_r / probs_r.sum()

        return xs.copy(), probs_r

    else:
        raise ValueError(f"Unknown rounding strategy: {strategy!r}. "
                         f"Choose from: none, threshold, topk, quantile, "
                         f"merge_tail, step")


# ------------------------------------------------------------------------------------------
# Convenience: entropy + expected-flip diagnostics
# To compare strategies.
# ------------------------------------------------------------------------------------------
def distribution_stats(xs, probs, label=''):
    """Print entropy and symbol count for a (xs, probs) pair."""
    H = -np.sum(probs * np.log2(probs + 1e-300))
    print(f"{label:20s}  symbols={len(probs):5d}  "
          f"H={H:.4f} bits  E[flips]≈{H:.2f}")
    return H

# -----------------------------
# OpenDP baseline (for comparison)
# -----------------------------

def opendp_dp_sum(data, epsilon):
    # standard dp sum using opendp laplace mechanism
    n, d = data.shape
    ts = np.sum(data, axis=0).astype(float)
    domain = vector_domain(
        atom_domain(T=float, bounds=(0., float(n)), nan=False), size=d
    )
    metric = l1_distance(float)
    scale  = d / epsilon
    meas   = make_laplace(domain, metric, scale)
    return ts, np.array(meas(ts))


# -----------------------------
# Entropy + codebook analysis
# huffman guarantees: H(p) <= E[code length] < H(p) + 1 bit
# w fair-coin biased flips: E[fair flips] <= 2 * H(p) + 2
# -----------------------------

def entropy_analysis(xs, probs, codebook, scale):
    lengths      = np.array([len(codebook[x]) for x in xs])
    H            = -np.sum(probs * np.log2(probs))       # entropy of p(x)
    H_laplace    = np.log2(2 * np.e * scale)             # differential entropy of Laplace
    expected_len = np.sum(probs * lengths)               # expected Huffman code length
    redundancy   = expected_len - H                      # should be < 1 bit (Huffman guarantee)

    print("=" * 55)
    print("Entropy + codebook analysis")
    print("=" * 55)
    print(f"  Alphabet size:              {len(xs)}")
    print(f"  Entropy H(p):               {H:.4f} bits")
    print(f"  Laplace differential H:     {H_laplace:.4f} bits")
    print(f"  Expected Huffman code len:  {expected_len:.4f} bits")
    print(f"  Redundancy E[L] - H(p):     {redundancy:.4f} bits  (< 1 guaranteed)")
    print(f"  Shortest codeword:          {lengths.min()} bits  "
          f"(symbol {xs[lengths.argmin()]:.1f})")
    print(f"  Longest codeword:           {lengths.max()} bits  "
          f"(symbol {xs[lengths.argmax()]:.1f})")

    print("\nSample codewords (10 symbols closest to 0):")
    center = sorted(codebook.keys(), key=lambda x: abs(x))[:10]
    for sym in sorted(center):
        print(f"  noise={sym:+7.1f}  ->  {codebook[sym]:<20}  "
              f"({len(codebook[sym])} bits)")

    return H, expected_len, redundancy

# -----------------------------------------------------------------------------------
#Discrete Laplace sampler (CKS)
# -----------------------------------------------------------------------------------

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


def dlap_batch_counted(n, s=1, t=1000):
    """
    Draw n samples from Discrete Laplace(s, t) and record fair bits per sample.
 
    Returns
    -------
    samples     : np.ndarray, shape (n,)
    flip_counts : np.ndarray, shape (n,)
    """
    samples, flip_counts = [], []
    for _ in range(n):
        z, f = sample_discrete_laplace(s, t)
        samples.append(z)
        flip_counts.append(f)
    return np.array(samples), np.array(flip_counts)


# ── Helpers + Faster approximations ───────────────────────────────────────────────────────────────────

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


def pairwise_ratios(
    opendp_bits,
    huffman_bits,
    dlap_bits,
    label="",
):
    """
    Given scalar or array-like average fair-bit costs for the three mechanisms,
    compute all pairwise ratios A/B and print a summary table.
 
    Parameters
    ----------
    opendp_bits  : float or array  — bits per sample for OpenDP Laplace
    huffman_bits : float or array  — bits per sample for Huffman (discretised Laplace)
    dlap_bits    : float or array  — bits per sample for Discrete Laplace sampler
    label        : str             — optional header label (e.g. epsilon value)
 
    Returns
    -------
    dict with keys:
        huffman_over_opendp   : Huffman / OpenDP
        dlap_over_opendp      : DLap    / OpenDP
        dlap_over_huffman     : DLap    / Huffman
        opendp_over_huffman   : OpenDP  / Huffman
        opendp_over_dlap      : OpenDP  / DLap
        huffman_over_dlap     : Huffman / DLap
    """
    o = np.asarray(opendp_bits,  dtype=float)
    h = np.asarray(huffman_bits, dtype=float)
    d = np.asarray(dlap_bits,    dtype=float)
 
    ratios = {
        "huffman_over_opendp" : h / o,
        "dlap_over_opendp"    : d / o,
        "dlap_over_huffman"   : d / h,
        "opendp_over_huffman" : o / h,
        "opendp_over_dlap"    : o / d,
        "huffman_over_dlap"   : h / d,
    }
 
    hdr = f"  Pairwise bit-cost ratios  {('(' + label + ')') if label else ''}"
    print("=" * 55)
    print(hdr)
    print("=" * 55)
    print(f"  {'Ratio':<30}  {'Value':>10}")
    print(f"  {'-'*30}  {'-'*10}")
    for name, val in ratios.items():
        scalar = float(np.mean(val))        # mean if array, identity if scalar
        a, b   = name.split("_over_")
        print(f"  {a:<14} / {b:<14}  {scalar:>10.4f}×")
    print("=" * 55)
 
    return ratios
 
 
# ── convenience: run a full comparison at a given epsilon ─────────────────────
 
def compare_at_epsilon(epsilon, n_samples=500, s=1, t=1000, huffman_root=None):
    """
    Draw n_samples from Discrete Laplace and (optionally) Huffman, then
    call pairwise_ratios and print results.
 
    Parameters
    ----------
    epsilon       : float  — privacy budget
    n_samples     : int    — number of samples to draw
    s, t          : int    — Discrete Laplace parameters
    huffman_root  : Node   — pre-built Huffman tree (from your build_huffman_tree).
                            If None, Huffman stats are estimated analytically.
 
    Returns
    -------
    dict of pairwise ratios (same as pairwise_ratios return value)
    """
    scale = 1.0 / epsilon
 
    # ── Discrete Laplace bits ────────────────────────────────────────────────
    _, dlap_flips = dlap_batch_counted(n_samples, s=s, t=t)
    dlap_mean = float(dlap_flips.mean())
 
    # ── Huffman bits (simulated if tree supplied, else analytical bound) ──────
    if huffman_root is not None:
        _, huff_flips = huffman_sample_batch_fair(huffman_root, n_samples)
        huff_mean = float(huff_flips.mean())
    else:
        # analytical: E[code length] in (H, H+1), each bit costs 2 fair flips
        import scipy.stats as stats
        xs   = np.arange(-200, 201, 1.0)
        probs = (stats.laplace.cdf(xs + 0.5, scale=scale) -
                 stats.laplace.cdf(xs - 0.5, scale=scale))
        probs = probs / probs.sum()
        H    = -np.sum(probs * np.log2(np.where(probs > 0, probs, 1)))
        huff_mean = (H + 0.5) * 2          # midpoint of Huffman guarantee × 2 fair/bit
 
    # ── OpenDP theoretical lower bound (= H(p)) ──────────────────────────────
    import scipy.stats as stats
    xs    = np.arange(-200, 201, 1.0)
    probs = (stats.laplace.cdf(xs + 0.5, scale=scale) -
             stats.laplace.cdf(xs - 0.5, scale=scale))
    probs = probs / probs.sum()
    opendp_mean = float(-np.sum(probs * np.log2(np.where(probs > 0, probs, 1))))
 
    print(f"\n  n={n_samples} samples | ε={epsilon} | scale={scale:.3f}")
    print(f"  OpenDP   (theoretical lower bound): {opendp_mean:.4f} bits/sample")
    print(f"  Huffman  (analytical estimate):     {huff_mean:.4f} bits/sample")
    print(f"  DLap     (simulated, s={s}, t={t}):      {dlap_mean:.4f} bits/sample")
 
    return pairwise_ratios(
        opendp_bits  = opendp_mean,
        huffman_bits = huff_mean,
        dlap_bits    = dlap_mean,
        label        = f"ε={epsilon}",
    )
 
 
# ── quick smoke-test ──────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print("Smoke test: single sample_discrete_laplace_counted")
    z, f = sample_discrete_laplace(s=1, t=1000)
    print(f"  sample={z}, fair flips used={f}")
 
    print("\nBatch of 10:")
    samples, flips = dlap_batch_counted(10, s=1, t=1000)
    print(f"  samples     : {samples}")
    print(f"  flip counts : {flips}")
    print(f"  mean flips  : {flips.mean():.2f}")
 
    print()
    compare_at_epsilon(epsilon=1.0, n_samples=300)
    compare_at_epsilon(epsilon=10.0, n_samples=300)
 


def analyse(samples, flip_counts, xs, probs, scale):
    H = -np.sum(probs * np.log2(probs))  # entropy of p(x)

    print("\n" + "=" * 55)
    print("Fair-flip Huffman sampler analysis")
    print("=" * 55)
    print(f"  Entropy H(p):              {H:.4f} bits")
    print(f"  Theoretical upper bound:   {2 * H:.4f} bits  (2 x H)")
    print(f"  Mean fair flips used:      {flip_counts.mean():.4f}")
    print(f"  Std fair flips:            {flip_counts.std():.4f}")
    print(f"  Sample mean (expect ~0):   {samples.mean():.4f}")
    print(f"  Sample std  (expect ~{np.sqrt(2)*scale:.1f}): {samples.std():.4f}")

    _plot(samples, flip_counts, xs, probs, scale, H)


def _plot(samples, flip_counts, xs, probs, scale, H):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Fair-flip Huffman sampler — Laplace noise", fontsize=13)

    # plot 1: sampled noise distribution vs theoretical -> does it look like laplace?
    ax = axes[0]
    ax.hist(samples, bins=80, density=True, alpha=0.6,
            color="#378ADD", label="Huffman samples")
    xplot = np.linspace(samples.min(), samples.max(), 400)
    ax.plot(xplot, stats.laplace.pdf(xplot, scale=scale),
            "r--", linewidth=1.5, label=f"Laplace(0, {scale})")
    ax.set_xlabel("Noise value")
    ax.set_ylabel("Density")
    ax.set_title("Sampled distribution vs Laplace")
    ax.legend(fontsize=9)

    # plot 2: fair flips per sample -> amt of flips being used: should be between H and 2(H)
    ax = axes[1]
    ax.hist(flip_counts,
            bins=range(int(flip_counts.min()), int(flip_counts.max()) + 2),
            density=True, alpha=0.7, color="#1D9E75")
    ax.axvline(flip_counts.mean(), color="r",     linestyle="--",
               linewidth=1.5, label=f"mean = {flip_counts.mean():.2f}")
    ax.axvline(H,                  color="orange", linestyle="--",
               linewidth=1.5, label=f"H(p) = {H:.2f}")
    ax.axvline(2 * H,              color="purple", linestyle="--",
               linewidth=1.5, label=f"2H(p) = {2*H:.2f}")
    ax.set_xlabel("Fair coin flips per sample")
    ax.set_ylabel("Density")
    ax.set_title("Flip cost  (want: H < mean < 2H)")
    ax.legend(fontsize=9)

    # plot 3: QQ -> are tails correct?
    ax = axes[2]
    (osm, osr), (slope, intercept, r) = stats.probplot(
        samples, dist=stats.laplace, sparams=(0, scale)
    )
    ax.scatter(osm, osr, s=2, alpha=0.3, color="#378ADD")
    ax.plot(osm, slope * np.array(osm) + intercept,
            "r--", linewidth=1.5, label=f"r = {r:.4f}")
    ax.set_xlabel("Theoretical Laplace quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.set_title("Q-Q vs Laplace(0, d/ε)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

# -----------------------------
# Residual distribution comparison
# both should look like Laplace(0, scale)
# -----------------------------

def residual_comparison(true_vals, dp_huffman, dp_opendp, scale):
    resid_h = dp_huffman - true_vals
    resid_o = dp_opendp  - true_vals

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Residual distributions — Huffman vs OpenDP", fontsize=13)

    xplot = np.linspace(-4 * scale, 4 * scale, 400)
    theo  = stats.laplace.pdf(xplot, scale=scale)

    for ax, resid, title, color in zip(
        axes,
        [resid_h,   resid_o],
        ["Huffman", "OpenDP"],
        ["#378ADD", "#1D9E75"]
    ):
        ax.hist(resid, bins=80, density=True,
                alpha=0.6, color=color, label=f"{title} residuals")
        ax.plot(xplot, theo, "r--", linewidth=1.5,
                label=f"Laplace(0, {scale:.0f}) theoretical")
        ax.set_xlim(-4 * scale, 4 * scale)
        ax.set_xlabel("Residual (DP - True)")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

# -----------------------------
# Random bits comparison: Huffman vs standard Laplace
# standard laplace via inverse CDF uses one float64 per sample = 53 bits (float64 mantissa)
# huffman uses ~H(p) to ~2H(p) fair bits per sample
# -----------------------------

def bits_comparison(flip_counts, probs, d, n_samples=5_000):
    H = -np.sum(probs * np.log2(probs))  # entropy of p(x)

    # standard laplace: one float64 per dimension per query
    # float64 has 53 bits of mantissa -> 53 bits per sample
    bits_per_sample_laplace = 64
    total_bits_laplace = bits_per_sample_laplace * d

    # huffman: actual observed mean from our samples
    mean_huffman_per_sample = flip_counts.mean()
    total_bits_huffman = mean_huffman_per_sample * d

    # theoretical bounds for huffman
    theoretical_lower = H        # can't do better than entropy
    theoretical_upper = 2 * H    # irpan binary expansion guarantee

    print("\n" + "=" * 55)
    print("Random bits comparison — per query (d dimensions)")
    print("=" * 55)
    print(f"\n  Standard Laplace (float64 inverse CDF):")
    print(f"    Bits per sample:       {bits_per_sample_laplace}  (float64 mantissa)")
    print(f"    Total bits per query:  {total_bits_laplace}  ({bits_per_sample_laplace} x d={d})")

    print(f"\n  Huffman sampler (fair coin flips):")
    print(f"    H(p) lower bound:      {theoretical_lower:.4f} bits per sample")
    print(f"    2H(p) upper bound:     {theoretical_upper:.4f} bits per sample")
    print(f"    Observed mean:         {mean_huffman_per_sample:.4f} bits per sample")
    print(f"    Total bits per query:  {mean_huffman_per_sample * d:.2f}  (mean x d={d})")

    print(f"\n  Savings:")
    saving_vs_laplace = total_bits_laplace - mean_huffman_per_sample * d
    saving_pct        = saving_vs_laplace / total_bits_laplace * 100
    print(f"    Bits saved per query:  {saving_vs_laplace:.2f}")
    print(f"    Reduction:             {saving_pct:.1f}%")
    print(f"    Ratio (Laplace/Huffman): {total_bits_laplace / (mean_huffman_per_sample * d):.2f}x")

    _plot_bits_comparison(
        flip_counts, H,
        bits_per_sample_laplace,
        theoretical_lower, theoretical_upper
    )

def accuracy_comparison_three_way(
    root,
    xs,
    probs,
    scale,
    epsilon=1.0,
    n=200,
    d=20,
    p=0.3,
    s=10,
    t=200,
    n_runs=10,
    seed=42
):
    """
    Compare:
      1. OpenDP Laplace baseline
      2. Huffman Laplace mechanism
      3. (Optional) Discrete Laplace mechanism

    Produces:
      - Mean trajectory across runs
      - ±1 std envelope
      - Individual faint traces
    """

    np.random.seed(seed)
    random.seed(seed)

    # ------------------------------------------------------------------
    # Fixed dataset across all runs
    # ------------------------------------------------------------------
    data = generate_sample_data(n=n, d=d, p=p)
    true_sums = np.sum(data, axis=0).astype(float)
    coords = np.arange(d)

    # ------------------------------------------------------------------
    # OpenDP baseline setup
    # ------------------------------------------------------------------
    domain = vector_domain(
        atom_domain(T=float, bounds=(0., float(n)), nan=False),
        size=d
    )

    metric = l1_distance(float)
    meas = make_laplace(domain, metric, scale)

    # ------------------------------------------------------------------
    # Collect runs
    # ------------------------------------------------------------------
    baseline_runs = []
    huffman_runs  = []
    mechanism5_runs = []

    # Uncomment if using discrete Laplace
    dlap_runs = []

    for _ in range(n_runs):

        # --------------------------------------------------------------
        # Baseline Laplace
        # --------------------------------------------------------------
        dp_base = np.array(meas(true_sums))
        baseline_runs.append(dp_base)

        # --------------------------------------------------------------
        # Huffman mechanism
        # --------------------------------------------------------------
        huff_noise = np.array([
            huffman_sample_fair(root)[0]
            for _ in range(d)
        ])

        dp_huff = true_sums + huff_noise
        huffman_runs.append(dp_huff)

        # # --------------------------------------------------------------
        # # Discrete Laplace (optional)
        # # --------------------------------------------------------------
        dlap_noise, _ = dlap_batch_counted(d, s=s, t=t)
        dp_dlap = true_sums + dlap_noise.astype(float)
        dlap_runs.append(dp_dlap)

        # --------------------------------------------------------------
        # Mechanism 5
        # --------------------------------------------------------------
        dp_mech5 = np.array(
            mechanism5(
                data,
                eps=epsilon,
                m=19,
                s=2
            ),
            dtype=float
        )

        mechanism5_runs.append(dp_mech5)


    baseline_runs = np.array(baseline_runs)
    huffman_runs  = np.array(huffman_runs)
    mechanism5_runs = np.array(mechanism5_runs)

    # Uncomment if using discrete Laplace
    dlap_runs = np.array(dlap_runs)

    # ------------------------------------------------------------------
    # Statistics helper
    # ------------------------------------------------------------------
    def summarize(name, runs):

        errors = runs - true_sums

        mean_run = np.mean(runs, axis=0)
        std_run  = np.std(runs, axis=0)

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        print(f"Mean absolute error:      {np.abs(errors).mean():.3f}")
        print(f"Mean std across coords:   {std_run.mean():.3f}")
        print(f"Max std coordinate:       {std_run.max():.3f}")

        return mean_run, std_run

    base_mean, base_std = summarize(
        "OpenDP Laplace Mechanism",
        baseline_runs
    )

    huff_mean, huff_std = summarize(
        "Huffman Mechanism",
        huffman_runs
    )

    # # Uncomment if using discrete Laplace
    dlap_mean, dlap_std = summarize(
        "Discrete Laplace Mechanism",
        dlap_runs
    )
    mech5_mean, mech5_std = summarize(
        "Shifted Laplace",
        mechanism5_runs
    )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))

    # ==============================================================
    # TRUE SUM
    # ==============================================================
    ax.plot(
        coords,
        true_sums,
        color="steelblue",
        linewidth=2.5,
        marker="o",
        markersize=5,
        label="True Sum",
        zorder=10
    )

    # ==============================================================
    # BASELINE LAPLACE
    # ==============================================================

    # faint runs
    for i, run in enumerate(baseline_runs):
        ax.plot(
            coords,
            run,
            color="#E05C5C",
            alpha=0.12,
            linewidth=0.8,
            label="Baseline runs" if i == 0 else None
        )

    # mean
    ax.plot(
        coords,
        base_mean,
        color="#C93C3C",
        linewidth=2,
        marker="s",
        markersize=5,
        linestyle="--",
        label="Baseline Mean"
    )

    # envelope
    ax.fill_between(
        coords,
        base_mean - base_std,
        base_mean + base_std,
        alpha=0.18,
        color="#E05C5C",
        label="Baseline ±1 Std"
    )

    # ==============================================================
    # HUFFMAN
    # ==============================================================

    for i, run in enumerate(huffman_runs):
        ax.plot(
            coords,
            run,
            color="#E8A020",
            alpha=0.12,
            linewidth=0.8,
            label="Huffman runs" if i == 0 else None
        )

    ax.plot(
        coords,
        huff_mean,
        color="#CC8400",
        linewidth=2,
        marker="^",
        markersize=5,
        linestyle="--",
        label="Huffman Mean"
    )

    ax.fill_between(
        coords,
        huff_mean - huff_std,
        huff_mean + huff_std,
        alpha=0.18,
        color="#E8A020",
        label="Huffman ±1 Std"
    )

    # # ==============================================================
    # # DISCRETE LAPLACE (OPTIONAL)
    # # ==============================================================

    
    for i, run in enumerate(dlap_runs):
        ax.plot(
            coords,
            run,
            color="#1D9E75",
            alpha=0.12,
            linewidth=0.8,
            label="DLap runs" if i == 0 else None
        )

    ax.plot(
        coords,
        dlap_mean,
        color="#147A5B",
        linewidth=2,
        marker="D",
        markersize=5,
        linestyle="--",
        label="DLap Mean"
    )

    ax.fill_between(
        coords,
        dlap_mean - dlap_std,
        dlap_mean + dlap_std,
        alpha=0.18,
        color="#1D9E75",
        label="DLap ±1 Std"
    )
        # ==============================================================
    # MECHANISM 5
    # ==============================================================

    for i, run in enumerate(mechanism5_runs):
        ax.plot(
            coords,
            run,
            color="#2563EB",
            alpha=0.10,
            linewidth=0.8,
            label="Shifted Laplace runs" if i == 0 else None
        )

    ax.plot(
        coords,
        mech5_mean,
        color="#1D4ED8",
        linewidth=2,
        marker="x",
        markersize=5,
        linestyle="--",
        label="Shifted Laplace Mean"
    )

    ax.fill_between(
        coords,
        mech5_mean - mech5_std,
        mech5_mean + mech5_std,
        alpha=0.18,
        color="#2563EB",
        label="Shifted Laplace ±1 Std"
    )
  
    # ------------------------------------------------------------------
    # Final formatting
    # ------------------------------------------------------------------
    ax.set_xlabel("Coordinate (dimension index)", fontsize=12)

    ax.set_ylabel("Count / Noisy Count", fontsize=12)

    ax.set_title(
        f"True vs Differentially Private Sums\n"
        f"({n_runs} independent runs on fixed dataset, ε={epsilon})",
        fontsize=13
    )

    ax.grid(True, linestyle="--", alpha=0.35)

    ax.legend(fontsize=9, ncol=2)

    plt.tight_layout()

    plt.savefig(
        "accuracy_comparison_envelope.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print("\nFigure saved: accuracy_comparison_envelope.png")

    return (
        true_sums,
        baseline_runs,
        huffman_runs,
        dlap_runs,
        mechanism5_runs
    )

def _plot_bits_comparison(flip_counts, H,
                           bits_laplace,
                           theoretical_lower, theoretical_upper):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Random bits: Huffman vs standard Laplace", fontsize=13)

    # plot 1: histogram of huffman flip counts w laplace line
    ax = axes[0]
    ax.hist(flip_counts,
            bins=range(int(flip_counts.min()), int(flip_counts.max()) + 2),
            density=True, alpha=0.7, color="#378ADD", label="Huffman flips per sample")
    ax.axvline(flip_counts.mean(),    color="r",      linestyle="--",
               linewidth=1.5, label=f"Huffman mean = {flip_counts.mean():.2f}")
    ax.axvline(theoretical_lower,     color="orange",  linestyle="--",
               linewidth=1.5, label=f"H(p) = {theoretical_lower:.2f}")
    ax.axvline(theoretical_upper,     color="purple",  linestyle="--",
               linewidth=1.5, label=f"2H(p) = {theoretical_upper:.2f}")
    ax.axvline(bits_laplace,          color="red",     linestyle="-",
               linewidth=2,   label=f"OpenDP= {bits_laplace}")
    ax.set_xlabel("Fair bits used per sample")
    ax.set_ylabel("Density")
    ax.set_title("Per-sample bit cost distribution")
    ax.legend(fontsize=8)

    # plot 2: bar chart comparing total bits per query across d dimensions
    ax = axes[1]
    d_range = np.arange(1, 41)  # d from 1 to 40
    laplace_total = bits_laplace * d_range
    huffman_total = flip_counts.mean() * d_range

    ax.plot(d_range, laplace_total, color="red",     linewidth=2,
            label=f"Laplace float64 (64 x d)")
    ax.plot(d_range, huffman_total, color="#378ADD", linewidth=2,
            label=f"Huffman mean ({flip_counts.mean():.1f} x d)")
    ax.fill_between(d_range,
                    theoretical_lower * d_range,
                    theoretical_upper * d_range,
                    alpha=0.15, color="#1D9E75",
                    label=f"Huffman [H, 2H] bounds")
    ax.set_xlabel("Number of dimensions (d)")
    ax.set_ylabel("Total fair bits per query")
    ax.set_title("Total bits per query vs d")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

def test_scale(s, t, n=100_000):
    samples, _ = dlap_batch_counted(n, s=s, t=t)
    print(f"s={s}, t={t} | mean={samples.mean():.4f} | std={samples.std():.4f} | var={samples.var():.4f}")

def bits_comparison_three_way(flip_counts_huffman, flip_counts_dlap, probs, d, epsilon=1.0):
    """
    Three-way randomness comparison: OpenDP baseline vs Huffman vs Discrete Laplace.
    All comparisons on a single figure with four panels.

    Parameters
    ----------
    flip_counts_huffman : np.ndarray — observed fair flips per sample from Huffman
    flip_counts_dlap    : np.ndarray — observed fair flips per sample from CKS DLap
    probs               : np.ndarray — discretised Laplace PMF (from build_px)
    d                   : int        — number of dimensions
    epsilon             : float      — privacy budget (for labelling)
    """
    H = -np.sum(probs * np.log2(probs))          # entropy of discretised Laplace
    opendp_bits_per_sample = 64                   # 64-bit float per coordinate via PRNG

    huff_mean  = flip_counts_huffman.mean()
    dlap_mean  = flip_counts_dlap.mean()

    # ── print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Three-way randomness comparison")
    print("=" * 60)
    print(f"  Entropy H(p):                     {H:.4f} bits")
    print(f"  OpenDP baseline (PRNG):           {opendp_bits_per_sample} bits/sample  (fixed)")
    print(f"  Huffman — observed mean:          {huff_mean:.4f} bits/sample")
    print(f"  Huffman — within [H, 2H]:         [{H:.2f}, {2*H:.2f}]")
    print(f"  DLap (CKS) — observed mean:       {dlap_mean:.4f} bits/sample")
    print(f"\n  Ratios:")
    print(f"    OpenDP / Huffman:               {opendp_bits_per_sample / huff_mean:.2f}x")
    print(f"    OpenDP / DLap:                  {opendp_bits_per_sample / dlap_mean:.2f}x")
    print(f"    DLap   / Huffman:               {dlap_mean / huff_mean:.2f}x")
    print(f"    Huffman / H(p):                 {huff_mean / H:.2f}x  (expect < 2)")

    # ── figure: 2x2 layout ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Randomness Comparison — OpenDP vs Huffman vs Discrete Laplace  |  ε={epsilon},  d={d}",
        fontsize=13
    )

    colours = {
        "OpenDP":   "#E05C5C",
        "Huffman":  "#378ADD",
        "DLap":     "#1D9E75",
    }

    # ── Panel 1 (top-left): per-sample flip distribution, Huffman vs DLap ────
    ax = axes[0, 0]

    # Huffman histogram
    ax.hist(flip_counts_huffman,
            bins=range(int(flip_counts_huffman.min()),
                       int(flip_counts_huffman.max()) + 2),
            density=True, alpha=0.55, color=colours["Huffman"],
            label="Huffman samples")

    # DLap histogram
    ax.hist(flip_counts_dlap,
            bins=range(int(flip_counts_dlap.min()),
                       int(flip_counts_dlap.max()) + 2),
            density=True, alpha=0.45, color=colours["DLap"],
            label="DLap (CKS) samples")

    # reference lines
    ax.axvline(huff_mean,             color=colours["Huffman"], linestyle="--",
               linewidth=1.8, label=f"Huffman mean = {huff_mean:.1f}")
    ax.axvline(dlap_mean,             color=colours["DLap"],    linestyle="--",
               linewidth=1.8, label=f"DLap mean = {dlap_mean:.1f}")
    ax.axvline(H,                     color="orange",            linestyle=":",
               linewidth=1.5, label=f"H(p) = {H:.1f}")
    ax.axvline(2 * H,                 color="purple",            linestyle=":",
               linewidth=1.5, label=f"2H(p) = {2*H:.1f}")
    ax.axvline(opendp_bits_per_sample, color=colours["OpenDP"],  linestyle="-",
               linewidth=2,   label=f"OpenDP = {opendp_bits_per_sample} bits")

    ax.set_xlabel("Fair bits per sample")
    ax.set_ylabel("Density")
    ax.set_title("Per-sample bit cost distribution")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)

    # ── Panel 2 (top-right): total bits per query vs d ────────────────────────
    ax = axes[0, 1]
    d_range = np.arange(1, 51)

    ax.plot(d_range, opendp_bits_per_sample * d_range,
            color=colours["OpenDP"],  linewidth=2,
            label=f"OpenDP ({opendp_bits_per_sample} × d)")
    ax.plot(d_range, dlap_mean * d_range,
            color=colours["DLap"],    linewidth=2,
            label=f"DLap ({dlap_mean:.1f} × d)")
    ax.plot(d_range, huff_mean * d_range,
            color=colours["Huffman"], linewidth=2,
            label=f"Huffman ({huff_mean:.1f} × d)")
    ax.fill_between(d_range,
                    H * d_range, 2 * H * d_range,
                    alpha=0.12, color=colours["Huffman"],
                    label=f"Huffman [H, 2H] band")

    # mark d=20 with a vertical reference
    ax.axvline(d, color="grey", linestyle=":", linewidth=1.2, label=f"d = {d}")

    ax.set_xlabel("Number of dimensions (d)")
    ax.set_ylabel("Total fair bits per query")
    ax.set_title("Total bits per query vs d")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)

    # ── Panel 3 (bottom-left): bar chart at fixed d ───────────────────────────
    ax = axes[1, 0]

    mechanisms = ["OpenDP\n(PRNG)", "Huffman\n(fair coins)", "DLap / CKS\n(fair coins)"]
    means      = [opendp_bits_per_sample, huff_mean, dlap_mean]
    cols       = [colours["OpenDP"], colours["Huffman"], colours["DLap"]]
    bars       = ax.bar(mechanisms, means, color=cols, alpha=0.8, width=0.5)

    # entropy lower bound line
    ax.axhline(H,     color="orange", linestyle="--", linewidth=1.5,
               label=f"H(p) = {H:.1f}  (lower bound)")
    ax.axhline(2 * H, color="purple", linestyle="--", linewidth=1.5,
               label=f"2H(p) = {2*H:.1f}  (Huffman upper bound)")

    # annotate bars with values
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Mean fair bits per sample")
    ax.set_title(f"Mean bits per sample at d={d}, ε={epsilon}")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # ── Panel 4 (bottom-right): bits saved relative to OpenDP ────────────────
    ax = axes[1, 1]
    d_range = np.arange(1, 51)

    saving_huffman = (opendp_bits_per_sample - huff_mean) * d_range
    saving_dlap    = (opendp_bits_per_sample - dlap_mean) * d_range

    ax.plot(d_range, saving_huffman,
            color=colours["Huffman"], linewidth=2,
            label=f"OpenDP − Huffman  ({opendp_bits_per_sample} − {huff_mean:.1f} per sample)")
    ax.plot(d_range, saving_dlap,
            color=colours["DLap"],    linewidth=2,
            label=f"OpenDP − DLap  ({opendp_bits_per_sample} − {dlap_mean:.1f} per sample)")

    ax.axhline(0, color="grey", linewidth=1)
    ax.fill_between(d_range, saving_huffman, alpha=0.1, color=colours["Huffman"])
    ax.fill_between(d_range, saving_dlap,    alpha=0.1, color=colours["DLap"])
    ax.axvline(d, color="grey", linestyle=":", linewidth=1.2, label=f"d = {d}")

    ax.set_xlabel("Number of dimensions (d)")
    ax.set_ylabel("Bits saved vs OpenDP baseline")
    ax.set_title("Bit savings relative to OpenDP")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("randomness_three_way.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved: randomness_three_way.png")

# ------------------------------------------------------------------------------------------
# Entropy after rounding / coarsening
#
# Goal:
#   Compare entropy H(Y) under different rounding strategies.
#
# Interpretation:
#   Lower entropy => fewer expected random bits for Huffman sampling.
#   But overly aggressive rounding distorts the Laplace distribution.
#
# We plot:
#   - entropy H(Y)
#   - alphabet size
#   - expected Huffman code length
#
# This mirrors the "randomness complexity vs approximation" idea
# from the Harvard randomness-in-DP paper.
# ------------------------------------------------------------------------------------------

def entropy_rounding_comparison(xs, probs, scale):
    """
    Compare entropy after applying different rounding/coarsening schemes.
    """

    configs = [
        ("none",        dict(strategy="none")),

        ("threshold\n1e-4",
            dict(strategy="threshold", threshold=1e-4)),

        ("threshold\n5e-4",
            dict(strategy="threshold", threshold=5e-4)),

        ("topk\n51",
            dict(strategy="topk", topk=51)),

        ("topk\n21",
            dict(strategy="topk", topk=21)),

        ("quantile\n1%",
            dict(strategy="quantile", quantile=0.01)),

        ("quantile\n5%",
            dict(strategy="quantile", quantile=0.05)),

        ("merge_tail\n1e-4",
            dict(strategy="merge_tail", threshold=1e-4)),

        ("step\n2",
            dict(strategy="step", step=2)),

        ("step\n4",
            dict(strategy="step", step=4)),
        
        ("dyadic\nnearest",
            dict(strategy="dyadic", dyadic_mode="nearest")),

        ("dyadic\nfloor",
            dict(strategy="dyadic", dyadic_mode="floor")),

        ("dyadic\nceil",
            dict(strategy="dyadic", dyadic_mode="ceil")),
    ]

    names            = []
    entropies        = []
    expected_lengths = []
    symbol_counts    = []

    print("\n" + "=" * 70)
    print("Entropy after rounding / coarsening")
    print("=" * 70)

    for label, kwargs in configs:

        xs_r, probs_r = round_px(xs, probs, **kwargs)

        # entropy
        H = -np.sum(probs_r * np.log2(probs_r + 1e-300))

        # build Huffman codebook
        root     = build_huffman_tree(xs_r, probs_r)
        codebook = build_codebook(root)

        lengths = np.array([len(codebook[x]) for x in xs_r])

        E_len = np.sum(probs_r * lengths)

        names.append(label)
        entropies.append(H)
        expected_lengths.append(E_len)
        symbol_counts.append(len(xs_r))

        print(f"{label:20s} | "
              f"symbols={len(xs_r):4d} | "
              f"H={H:7.4f} | "
              f"E[L]={E_len:7.4f}")

    order = np.argsort(entropies)

    sorted_names = [names[i] for i in order]
    sorted_entropies = [entropies[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 6))

    # horizontal bars
    bars = ax.barh(
        sorted_names,
        sorted_entropies,
        alpha=0.85
    )

    # labels and title
    ax.set_xlabel("Entropy H(Y) (bits)")
    ax.set_title("Entropy After Rounding")

    # cleaner style
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # annotate bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.03,
            bar.get_y() + bar.get_height()/2,
            f"{width:.2f}",
            va='center',
            fontsize=9
        )

    plt.tight_layout()
    plt.show()

    return {
        "names": names,
        "entropies": entropies,
        "expected_lengths": expected_lengths,
        "symbol_counts": symbol_counts,
    }

# ------------------------------------------------------------------------------------------
# Entropy vs rounding strategy comparison plot
#
# From Canonne, Su, Vadhan (2024): Huffman sampling costs H(Y) + O(1) expected fair bits.
# So H(rounded p(x)) is the direct quantity to minimise for randomness efficiency.
# This plot sweeps each strategy over its parameter range and shows:
#   - H(Y) after rounding (lower = fewer bits needed)
#   - symbol count (alphabet size, proxy for accuracy loss)
#   - mass retained (1 - probability discarded)
#
# Best strategy = lowest H with acceptable mass retention and symbol count.
# ------------------------------------------------------------------------------------------

def plot_entropy_vs_rounding(xs_base, probs_base, scale, figsize=(16, 10)):
    """
    Sweep all round_px strategies over their parameter ranges and plot:
      Panel 1 (top-left):  H(Y) vs parameter value, one line per strategy
      Panel 2 (top-right): symbol count vs parameter
      Panel 3 (bot-left):  mass retained vs parameter
      Panel 4 (bot-right): H(Y) vs mass retained — the efficiency frontier
                           (want: bottom-right = low H, high mass)

    Parameters
    ----------
    xs_base, probs_base : output of build_px() — unrounded baseline
    scale               : Laplace scale (for labelling)
    """
    H_base   = -np.sum(probs_base * np.log2(probs_base + 1e-300))
    n_base   = len(probs_base)

    # ── parameter grids for each strategy ────────────────────────────────────
    sweeps = {
        'threshold'  : ('threshold',  np.logspace(-5, -1, 40)),
        'topk'       : ('topk',       np.arange(3, min(n_base, 201), 4, dtype=int)),
        'quantile'   : ('quantile',   np.linspace(0.001, 0.20, 40)),
        'merge_tail' : ('threshold',  np.logspace(-5, -1, 40)),   # uses threshold param
        'step'       : ('step',       np.arange(1, 21, dtype=int)),
    }

    # colour per strategy
    colours = {
        'threshold'  : '#E05C5C',
        'topk'       : '#378ADD',
        'quantile'   : '#E8A020',
        'merge_tail' : '#1D9E75',
        'step'       : '#9B59B6',
    }

    # ── collect results ───────────────────────────────────────────────────────
        # replace these three lines at the top of the sweep loop initialisation:
    results = {}

    for name, (kwarg, param_grid) in sweeps.items():
        H_vals, sym_counts, mass_vals = [], [], []   # renamed from Hs, n_syms, masses
        for val in param_grid:
            kw = {kwarg: val}
            xr, pr = round_px(xs_base, probs_base, strategy=name, **kw)

            H_r = -np.sum(pr * np.log2(pr + 1e-300))

            if name == 'merge_tail':
                mass = 1.0
            else:
                if name == 'threshold':
                    raw_mask = probs_base >= kw['threshold']
                elif name == 'topk':
                    idx = np.argsort(probs_base)[::-1][:kw['topk']]
                    raw_mask = np.zeros(len(probs_base), dtype=bool)
                    raw_mask[idx] = True
                elif name == 'quantile':
                    order = np.argsort(probs_base)
                    cum   = np.cumsum(probs_base[order])
                    raw_mask = np.ones(len(probs_base), dtype=bool)
                    raw_mask[order[cum < kw['quantile']]] = False
                elif name == 'step':
                    raw_mask = np.ones(len(probs_base), dtype=bool)
                mass = float(probs_base[raw_mask].sum())

            n_sym = int(np.sum(xr != TAIL_SYMBOL))   # local scalar, not list

            H_vals.append(H_r)
            sym_counts.append(n_sym)       # append the local scalar
            mass_vals.append(mass)

        results[name] = {
            'params' : np.array(param_grid, dtype=float),
            'H'      : np.array(H_vals),
            'n_sym'  : np.array(sym_counts),
            'mass'   : np.array(mass_vals),
        }

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        f"Entropy vs rounding strategy  |  Laplace scale = {scale:.1f}  |  "
        f"baseline H = {H_base:.3f} bits,  {n_base} symbols",
        fontsize=13
    )

    # normalise x-axis to [0, 1] so all strategies appear on the same scale
    # raw param axes would be incommensurable (log-prob vs integer count vs fraction)

    # ── Panel 1: H(Y) vs normalised parameter ────────────────────────────────
    ax = axes[0, 0]
    ax.axhline(H_base, color='grey', linestyle=':', linewidth=1.5,
               label=f'baseline H = {H_base:.2f}')
    for name, r in results.items():
        p_norm = (r['params'] - r['params'].min()) / (
                  r['params'].max() - r['params'].min() + 1e-12)
        ax.plot(p_norm, r['H'], color=colours[name], linewidth=2, label=name)
    ax.set_xlabel("Parameter (normalised 0→1, increasing aggressiveness)")
    ax.set_ylabel("H(Y)  (bits)")
    ax.set_title("Entropy after rounding  ← lower = fewer bits needed")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)

    # ── Panel 2: symbol count vs normalised parameter ─────────────────────────
    ax = axes[0, 1]
    ax.axhline(n_base, color='grey', linestyle=':', linewidth=1.5,
               label=f'baseline n = {n_base}')
    for name, r in results.items():
        p_norm = (r['params'] - r['params'].min()) / (
                  r['params'].max() - r['params'].min() + 1e-12)
        ax.plot(p_norm, r['n_sym'], color=colours[name], linewidth=2, label=name)
    ax.set_xlabel("Parameter (normalised 0→1)")
    ax.set_ylabel("Alphabet size  (symbol count)")
    ax.set_title("Symbol count after rounding")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)

    # ── Panel 3: mass retained vs normalised parameter ────────────────────────
    ax = axes[1, 0]
    ax.axhline(1.0, color='grey', linestyle=':', linewidth=1.5, label='full mass')
    for name, r in results.items():
        p_norm = (r['params'] - r['params'].min()) / (
                  r['params'].max() - r['params'].min() + 1e-12)
        ax.plot(p_norm, r['mass'], color=colours[name], linewidth=2, label=name)
    ax.set_xlabel("Parameter (normalised 0→1)")
    ax.set_ylabel("Fraction of mass retained")
    ax.set_title("Mass retained after rounding ")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)

    # ── Panel 4: H(Y) vs mass retained — efficiency frontier ─────────────────
    # this is the key diagnostic: you want to be in the bottom-right corner
    # (low H = cheap sampling, high mass = low accuracy loss)
    ax = axes[1, 1]
    ax.axvline(1.0, color='grey', linestyle=':', linewidth=1)
    ax.axhline(H_base, color='grey', linestyle=':', linewidth=1.5,
               label=f'baseline H = {H_base:.2f}')

    for name, r in results.items():
        # scatter with plasma colourmap for aggressiveness — no label here
        ax.scatter(r['mass'], r['H'],
                   c=np.arange(len(r['H'])),
                   cmap='plasma',
                   s=18, alpha=0.7)
        # connecting line carries the legend label with the correct colour
        ax.plot(r['mass'], r['H'],
                color=colours[name], linewidth=1.8, alpha=0.8, label=name)
        # start dot
        ax.scatter(r['mass'][0], r['H'][0],
                   color=colours[name], s=60, zorder=5, marker='o')
        ax.annotate(name,
                    xy=(r['mass'][0], r['H'][0]),
                    xytext=(4, 2), textcoords='offset points',
                    fontsize=7, color=colours[name])

    ax.set_xlabel("Mass retained  (1 = no accuracy loss)")
    ax.set_ylabel("H(Y)  (bits)  ← lower = fewer fair flips")
    ax.set_title(
        "Efficiency frontier\n"
        "bottom-right = low entropy AND low mass loss  ← ideal"
    )
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig("entropy_vs_rounding.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Figure saved: entropy_vs_rounding.png")

    # ── print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  Entropy summary  (baseline: H={H_base:.4f}, symbols={n_base})")
    print("=" * 65)
    print(f"  {'strategy':<14}  {'min H':>8}  {'max ΔH':>10}  "
          f"{'min symbols':>12}  {'min mass':>10}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*10}")
    for name, r in results.items():
        print(f"  {name:<14}  {r['H'].min():>8.4f}  "
              f"{(H_base - r['H'].min()):>10.4f}  "
              f"{r['n_sym'].min():>12d}  "
              f"{r['mass'].min():>10.4f}")
    print("=" * 65)

    return results
# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    d       = 20
    epsilon = 1.0
    scale   = d / epsilon  # = 20.0

    print(f"\nParameters: d={d}, epsilon={epsilon}, scale={scale}")

    # step 1 — discretize Laplace into p(x)
    xs, probs = build_px(scale=scale, grid_step=1.0, tail_prob=0.999)
    # after build_px, before round_px / Huffman:
    rounding_results = plot_entropy_vs_rounding(xs, probs, scale)
    # step 1b — compare entropy under rounding schemes
    entropy_rounding_comparison(xs, probs, scale)
    xs,probs = round_px(xs, probs,
                          strategy='threshold', step=1.0, topk=51,
                          threshold=5e-4)
    print(f"\nbuild_px: {len(xs)} symbols, range [{xs.min():.0f}, {xs.max():.0f}]")

    # step 2 — build huffman tree + codebook
    root     = build_huffman_tree(xs, probs)
    codebook = build_codebook(root)

    # step 3 — entropy + codebook analysis
    H, E_len, redundancy = entropy_analysis(xs, probs, codebook, scale)

    # step 4 — validate sampler (5000 samples)
    print("\nGenerating 5000 Huffman samples for validation...")
    samples, flip_counts = huffman_sample_batch_fair(root, n=5_000)
    analyse(samples, flip_counts, xs, probs, scale)

    # step 5 — three-way accuracy comparison
    print("\nRunning three-way accuracy comparison...")
    true_sums, dp_base, dp_huff, discrete, dp_shifted_laplace = accuracy_comparison_three_way(
        root, xs, probs, scale,
        epsilon=epsilon, n=200, d=d, p=0.3,
        s=10, t=200, seed=42
    )

    # step 4b — random bits comparison
    bits_comparison(flip_counts, probs, d, n_samples=5_000)

    # step 6 — three-way randomness comparison
    print("\nRunning three-way randomness comparison...")
    _, flip_counts_dlap = dlap_batch_counted(5_000, s=10, t=200)
    bits_comparison_three_way(
        flip_counts_huffman=flip_counts,   # already computed in step 4
        flip_counts_dlap=flip_counts_dlap,
        probs=probs,
        d=d,
        epsilon=epsilon,
    )

    # Test several combinations where t/s ≈ 1
    test_scale(s=20, t=20)
    test_scale(s=30, t=30)
    test_scale(s=50, t=50)
    test_scale(s=100, t=100)

    # Also try slightly stronger decay
    test_scale(s=25, t=20)
    test_scale(s=40, t=30)