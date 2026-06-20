"""Compare the fast and reference shot counters.

Generates a set of synthetic rectangular masks of varying complexity, counts the
shots required by both the reference (adabox) counter and the fast (ILP) counter,
and plots their correlation together with the average speedup.

Usage:
    python examples/compare_shots.py --num 30 --size 256 --out shot_correlation.png

The fast counter is expected to (a) track the reference ranking closely
(high Spearman correlation) while (b) being far faster.
"""

import argparse
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shotcount import FastShotCounter, ShotCounter


def make_mask(size, n_rects, rng):
    """Build a random binary mask as the union of `n_rects` rectangles."""
    mask = torch.zeros((size, size), dtype=torch.float32)
    for _ in range(n_rects):
        w = int(rng.integers(size // 16, size // 4))
        h = int(rng.integers(size // 16, size // 4))
        x = int(rng.integers(0, size - w))
        y = int(rng.integers(0, size - h))
        mask[y:y + h, x:x + w] = 1.0
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=30, help="number of masks to evaluate")
    ap.add_argument("--size", type=int, default=256, help="mask resolution (NxN)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="shot_correlation.png")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    fast = FastShotCounter()
    ref = ShotCounter()

    # Vary the number of rectangles so shot counts span a wide range.
    n_rects = np.linspace(2, 40, args.num).round().astype(int)

    fast_counts, ref_counts = [], []
    fast_times, ref_times = [], []

    print(f"{'idx':>3} {'#rect':>5} {'ref':>5} {'fast':>5} {'ref_t':>8} {'fast_t':>8}")
    for i, k in enumerate(n_rects):
        mask = make_mask(args.size, int(k), rng)

        t0 = time.time()
        r = ref.run(mask, shape=(args.size, args.size))
        t1 = time.time()
        f = fast.run(mask, shape=(args.size, args.size))
        t2 = time.time()

        ref_counts.append(r)
        fast_counts.append(f)
        ref_times.append(t1 - t0)
        fast_times.append(t2 - t1)
        print(f"{i:>3} {k:>5} {r:>5} {f:>5} {t1 - t0:>8.3f} {t2 - t1:>8.3f}")

    ref_counts = np.array(ref_counts, dtype=float)
    fast_counts = np.array(fast_counts, dtype=float)

    # Ranking is what matters for the reward: Spearman rho. Pearson / R^2 of a
    # linear fit measure how linearly the proxy tracks the reference (slope is
    # irrelevant for a reward proxy, so we fit rather than assume y = x).
    spearman = np.corrcoef(
        ref_counts.argsort().argsort(), fast_counts.argsort().argsort())[0, 1]
    pearson = np.corrcoef(ref_counts, fast_counts)[0, 1]
    slope, intercept = np.polyfit(ref_counts, fast_counts, 1)
    fit = slope * ref_counts + intercept
    ss_res = np.sum((fast_counts - fit) ** 2)
    ss_tot = np.sum((fast_counts - fast_counts.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    speedup = np.mean(ref_times) / max(np.mean(fast_times), 1e-9)

    print(f"\nSpearman rho (ranking): {spearman:.4f}   <- key metric")
    print(f"Pearson r             : {pearson:.4f}")
    print(f"Linear-fit R^2        : {r2:.4f}")
    print(f"Avg ref time : {np.mean(ref_times):.4f}s")
    print(f"Avg fast time: {np.mean(fast_times):.4f}s")
    print(f"Mean speedup : {speedup:.1f}x")

    lo = min(ref_counts.min(), fast_counts.min())
    hi = max(ref_counts.max(), fast_counts.max())
    xs = np.array([lo, hi])

    plt.figure(figsize=(5.2, 5))
    plt.scatter(ref_counts, fast_counts, s=36, alpha=0.8, edgecolor="k", linewidth=0.4)
    plt.plot(xs, slope * xs + intercept, "-", color="C3", linewidth=1.5,
             label=f"linear fit ($R^2$ = {r2:.3f})")
    plt.plot([lo, hi], [lo, hi], ":", color="gray", linewidth=1, label="y = x")
    plt.xlabel("Reference shot count (adabox)")
    plt.ylabel("Fast shot count (ours)")
    plt.title(
        f"Shot-count ranking is preserved\n"
        f"Spearman $\\rho$ = {spearman:.3f}  ·  {speedup:.0f}× faster")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
