"""
Spiking associative-memory storage prototype  (stdlib-only, zero deps).

Project aim: reduce computational power and storage space with the experimental
SNN data-storing method described in the research docs. This prototype both
DEMONSTRATES the method and MEASURES those two metrics against a dense baseline.

Storage mechanism (straight from the docs):
  - long-term storage  == the stored patterns themselves. The Hopfield weight
    matrix W = Σ_p (ξ_p - a)(ξ_p - a)^T is rank-P, so we never materialise the
    N×N matrix: we keep the P sparse patterns as index lists and reconstruct the
    needed correlations on the fly (v0.7 — factored storage). The data IS the
    network fabric: storage and compute are co-located, no Von Neumann shuttle.
  - volatile recall    == attractor dynamics. Inject a NOISY / PARTIAL cue, then
    iterate sparse k-winners-take-all spiking until the network settles onto the
    stored pattern. Recall is content-addressable: a corrupted cue recovers the
    clean memory, AND denoises it — that is the value over a plain pattern list.

Where the savings come from:
  - storage : O(P·k) index bytes instead of O(N²) weights. For P,k ≪ N this is a
    real, large reduction — the factored memory is hundreds of times smaller than
    the dense W it represents (reported below). Fixed in v0.7.
  - compute : recall is event-driven and factored — O(P·k) work per step instead
    of the dense O(N²) matvec a materialised Hopfield net would do each step.

Recall math (so the factored form is exact, not an approximation):
  With W_ij = Σ_p (ξ_p,i-a)(ξ_p,j-a), zero diagonal, and cue active-set s,
  let d_p = |ξ_p ∩ s| - a·|s|. Then
      h_i = Σ_p (ξ_p,i - a)·d_p  -  diag_i  =  S_i - a·D - diag_i,
  where S_i = Σ_{p ∋ i} d_p, D = Σ_p d_p (constant in i, drops out of arg-top-k),
  and diag_i = cnt_i·(1-a)² + (P-cnt_i)·a²  for i ∈ s (else 0). Identical ranking
  to the dense matrix — same recall, far less memory.
"""
import random
from array import array

random.seed(0)  # ponytail: fixed seed -> deterministic self-check. Not a calibration knob.

# ---- problem size -----------------------------------------------------------
N = 256          # neurons (== data dimensionality, bits per pattern)
K = 20           # active neurons per pattern (sparse code, ~8% activity)
NUM_PATTERNS = 15
RECALL_STEPS = 8 # attractor settle steps cap


# ---- 1. write data into the (factored) memory -------------------------------
def make_patterns(num, n, k):
    """num random k-hot patterns; each is a sorted tuple of active indices."""
    return [tuple(sorted(random.sample(range(n), k))) for _ in range(num)]


def train(patterns, n, k):
    """Store patterns in FACTORED form: the rank-P Hopfield memory is just the
    patterns themselves (as frozensets) — no N×N matrix is ever built. Recall
    reconstructs the correlations on the fly. Signature kept for compatibility."""
    return [frozenset(p) for p in patterns]


# ---- 2. recall: noisy cue -> sparse attractor dynamics (kWTA) ----------------
def topk(values, k):
    """indices of the k largest values; deterministic tie-break by index."""
    return set(sorted(range(len(values)), key=lambda i: (-values[i], i))[:k])


def recall(memory, cue_active, n, k, steps):
    """Iterate event-driven kWTA to a fixed point, computing the field directly
    from the stored patterns (factored, never the N×N matrix). Returns
    (final_active_set, steps_used, synops). SynOps counts the spike-driven work:
    overlap probes (P·|active|) + scatter to pattern members (P·k) per step."""
    a = k / n
    P = len(memory)
    hi2, lo2 = (1.0 - a) * (1.0 - a), a * a
    cnt = [0] * n                          # cnt_i = #patterns containing neuron i
    for p in memory:
        for i in p:
            cnt[i] += 1
    active = set(cue_active)
    synops = 0
    used = 0
    for _ in range(steps):
        used += 1
        m = len(active)
        S = [0.0] * n
        for p in memory:
            d_p = sum(1 for j in active if j in p) - a * m   # |ξ_p ∩ s| - a|s|
            synops += m                                      # overlap probes
            if d_p == 0.0:
                continue
            for i in p:                                       # scatter to members
                S[i] += d_p
            synops += k
        for i in active:                                      # diagonal correction
            S[i] -= cnt[i] * hi2 + (P - cnt[i]) * lo2
        new_active = topk(S, k)
        if new_active == active:        # settled onto an attractor
            break
        active = new_active
    return active, used, synops


def corrupt(pattern, n, k, drop, add):
    """Make a partial/noisy cue: remove `drop` true bits, add `add` false bits."""
    active = list(pattern)
    random.shuffle(active)
    kept = set(active[drop:])                       # drop some real bits
    inactive = [i for i in range(n) if i not in pattern]
    kept.update(random.sample(inactive, add))       # add spurious bits
    return tuple(sorted(kept))


def overlap(a, b, k):
    """fraction of the stored pattern recovered, in [0,1]."""
    return len(set(a) & set(b)) / k


# ---- 3. run + measure -------------------------------------------------------
def main():
    patterns = make_patterns(NUM_PATTERNS, N, K)
    memory = train(patterns, N, K)

    # recall every stored pattern from a corrupted cue (drop 8 of 20, add 8 noise)
    overlaps, total_synops, total_steps = [], 0, 0
    for p in patterns:
        cue = corrupt(p, N, K, drop=8, add=8)
        out, steps_used, synops = recall(memory, cue, N, K, RECALL_STEPS)
        overlaps.append(overlap(out, p, K))
        total_synops += synops
        total_steps += steps_used

    mean_recall = sum(overlaps) / len(overlaps)
    perfect = sum(1 for o in overlaps if o == 1.0)

    # --- compute: factored spiking work vs a materialised dense Hopfield matvec ---
    dense_macs = total_steps * N * N            # dense recomputes full N x N each step
    compute_ratio = dense_macs / total_synops

    # --- storage: FACTORED memory vs the dense N x N weight matrix it represents ---
    factored_bytes = NUM_PATTERNS * K * 2       # P patterns x k indices x uint16
    dense_w_bytes = N * N * 8                    # the N x N float64 matrix (old cost)
    pattern_bitmap_bytes = NUM_PATTERNS * (-(-N // 8))  # patterns as dense 1-bit rows
    storage_ratio = dense_w_bytes / factored_bytes

    # --- storage of a single recalled state (AER events vs dense float32) ---
    dense_state_bytes = N * 4
    aer_bytes = K * 2

    print("=" * 60)
    print("SPIKING ASSOCIATIVE-MEMORY STORAGE PROTOTYPE  (v0.7 factored)")
    print("=" * 60)
    print(f"neurons N={N}  active K={K} ({K/N:.1%})  patterns stored={NUM_PATTERNS}")
    print()
    print("RECALL (content-addressable, from 40% corrupted cue):")
    print(f"  mean overlap with stored pattern : {mean_recall:.1%}")
    print(f"  perfectly recalled               : {perfect}/{NUM_PATTERNS}")
    print(f"  avg settle steps                 : {total_steps/NUM_PATTERNS:.1f}")
    print()
    print("COMPUTE  (lower = less power):")
    print(f"  factored SynOps  : {total_synops:,}")
    print(f"  dense Hopfield   : {dense_macs:,}  (N x N matvec each step)")
    print(f"  reduction        : {compute_ratio:.1f}x")
    print()
    print("STORAGE of the memory (lower = less space):")
    print(f"  factored (P x k indices) : {factored_bytes:,} B")
    print(f"  patterns as bitmaps      : {pattern_bitmap_bytes:,} B")
    print(f"  dense N x N matrix       : {dense_w_bytes:,} B")
    print(f"  reduction vs dense W     : {storage_ratio:.0f}x")
    print(f"  note: storage is O(P*k), NOT O(N^2). The memory is just the sparse")
    print(f"        patterns; recall reconstructs the correlations on the fly and")
    print(f"        denoises corrupted cues (content-addressable). That denoising")
    print(f"        recall is the value over a plain pattern list.")
    print("=" * 60)

    # ---- self-check (ponytail: one runnable check, assert-based) ----
    assert mean_recall >= 0.95, f"recall too weak: {mean_recall:.2f}"
    assert total_synops < dense_macs, "factored compute did not beat dense matvec"
    assert factored_bytes < dense_w_bytes, "factored storage not smaller than dense W"
    assert aer_bytes < dense_state_bytes, "events not smaller than dense state"
    print("self-check OK: recall>=95%, factored < dense on BOTH compute and storage")


if __name__ == "__main__":
    main()
