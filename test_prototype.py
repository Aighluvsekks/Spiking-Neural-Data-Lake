"""
Stress-test the spiking associative-memory prototype: how many patterns can it
hold, and how much cue corruption can it tolerate, before recall breaks?

Reuses the prototype's functions directly (no duplication). Run:
    python test_prototype.py
"""
from spiking_storage_prototype import make_patterns, train, recall, corrupt, overlap

N, K, STEPS = 256, 20, 12
TRIALS = 5  # average over a few random pattern sets for stable numbers


def mean_recall(num_patterns, drop, add, trials=TRIALS):
    """avg recall overlap across `trials` random codebooks, recalling every
    stored pattern from a cue with `drop` bits removed + `add` bits added."""
    total, count = 0.0, 0
    for t in range(trials):
        pats = make_patterns(num_patterns, N, K)
        W = train(pats, N, K)
        for p in pats:
            cue = corrupt(p, N, K, drop=drop, add=add)
            out, _, _ = recall(W, cue, N, K, STEPS)
            total += overlap(out, p, K)
            count += 1
    return total / count


def capacity_sweep():
    """Fix moderate noise (6 of 20 bits), grow the codebook until recall falls.
    The FACTORED memory stores the patterns themselves (O(P*k)), so it holds far more
    than the classic Hebbian 'few % of N' — recall stays high until inter-pattern
    crosstalk in the reconstructed field dominates. Swept well past N to find that knee."""
    print(f"CAPACITY SWEEP  (N={N}, K={K}, cue: drop 6 / add 6)")
    print(f"{'patterns':>9} | {'mean recall':>11} | bar")
    print("-" * 48)
    knee = None
    for num in (20, 40, 80, 128, 192, 256, 384, 512, 768):
        r = mean_recall(num, drop=6, add=6)
        bar = "#" * int(r * 20)
        print(f"{num:>9} | {r:>10.1%} | {bar}")
        if knee is None and r < 0.90:               # first load that drops below 90%
            knee = num
    if knee:
        print(f"knee (recall first < 90%): ~{knee} patterns = {knee / N:.1f}x N")
    else:
        print("knee: no cliff up to 3x N (768 patterns)")
    print()


def noise_sweep():
    """Fix a comfortable load (15 patterns), grow cue corruption until recall
    fails. drop=add keeps the cue the same size as a stored pattern."""
    print(f"NOISE SWEEP  (N={N}, K={K}, 15 patterns stored)")
    print(f"{'corruption':>10} | {'mean recall':>11} | bar")
    print("-" * 48)
    for d in (0, 2, 4, 6, 8, 10, 12, 14, 16):
        r = mean_recall(15, drop=d, add=d)
        pct = d / K  # fraction of active bits swapped
        bar = "#" * int(r * 20)
        print(f"{pct:>9.0%} | {r:>10.1%} | {bar}")
    print()


if __name__ == "__main__":
    print("=" * 48)
    print("SPIKING ASSOCIATIVE MEMORY — STRESS TEST")
    print("=" * 48)
    capacity_sweep()
    noise_sweep()
    # sanity floor: light load + light noise must still recall near-perfectly
    assert mean_recall(10, drop=4, add=4) >= 0.95, "regressed: easy case failed"
    print("self-check OK: 10 patterns @ 20% noise recalls >=95%")
