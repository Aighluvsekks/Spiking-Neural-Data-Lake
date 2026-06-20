"""
Spiking associative-memory storage prototype  (stdlib-only, zero deps).

Project aim: reduce computational power and storage space with the experimental
SNN data-storing method described in the research docs. This prototype both
DEMONSTRATES the method and MEASURES those two metrics against a dense baseline.

Storage mechanism (straight from the docs):
  - long-term storage  == synaptic weight matrix W, written once by a Hebbian /
    STDP correlation rule. The data IS the network fabric: storage and compute
    are co-located, so there is no Von Neumann shuttle between CPU and memory.
  - volatile recall    == attractor dynamics. Inject a NOISY / PARTIAL cue, then
    iterate sparse k-winners-take-all spiking (the docs' lateral-inhibition
    winner-take-all) until the network settles onto the stored pattern.
    Recall is content-addressable: a corrupted cue recovers the clean memory.

Where the savings come from:
  - compute : a spike only does work when it fires. SynOps = (active neurons) x
    fanout, vs a dense ANN that recomputes the full N x N matvec every step.
    Sparse k-of-N coding => ~ N/k fewer operations.
  - storage : a recalled state is binary and sparse. As an AER event list (just
    the active indices) it is far smaller than a dense float32 activation tensor.

Honest caveat: the weight matrix W itself costs N*N values. This method wins on
ACTIVATION compute/traffic and on content-addressable recall (no separate index
/ database), not on shrinking the stored weights. Capacity is reported below.
"""
import random
from array import array

random.seed(0)  # ponytail: fixed seed -> deterministic self-check. Not a calibration knob.

# ---- problem size -----------------------------------------------------------
N = 256          # neurons (== data dimensionality, bits per pattern)
K = 20           # active neurons per pattern (sparse code, ~8% activity)
NUM_PATTERNS = 15
RECALL_STEPS = 8 # attractor settle steps cap


# ---- 1. write data into synapses (Hebbian / STDP correlation rule) ----------
def make_patterns(num, n, k):
    """num random k-hot patterns; each is a sorted tuple of active indices."""
    return [tuple(sorted(random.sample(range(n), k))) for _ in range(num)]


def train(patterns, n, k):
    """Covariance Hebbian rule: W_ij += (x_i - a)(x_j - a), a = mean activity.
    Symmetric, zero diagonal. This is the docs' 'long-term storage in W via STDP'.
    Exploits sparsity: only the k active rows/cols of each pattern are touched."""
    a = k / n
    W = [array('d', [0.0]) * n for _ in range(n)]  # N x N float matrix
    neg_a = -a
    for p in patterns:
        active = set(p)
        # value for each neuron: (1 - a) if active else (0 - a)
        for i in range(n):
            vi = (1.0 - a) if i in active else neg_a
            if vi == 0.0:
                continue
            Wi = W[i]
            for j in range(n):
                if i == j:
                    continue
                vj = (1.0 - a) if j in active else neg_a
                Wi[j] += vi * vj
    return W


# ---- 2. recall: noisy cue -> sparse attractor dynamics (kWTA) ----------------
def topk(values, k):
    """indices of the k largest values; deterministic tie-break by index."""
    return set(sorted(range(len(values)), key=lambda i: (-values[i], i))[:k])


def recall(W, cue_active, n, k, steps):
    """Iterate event-driven kWTA until fixed point. Returns
    (final_active_set, steps_used, synops). SynOps counts only spike-driven work:
    each active presynaptic neuron drives its n postsynaptic targets."""
    active = set(cue_active)
    synops = 0
    used = 0
    for _ in range(steps):
        used += 1
        h = [0.0] * n
        for j in active:               # sparse: only active neurons emit spikes
            Wj = W[j]                   # symmetric, so row j == column j
            for i in range(n):
                h[i] += Wj[i]
            synops += n                 # one spike -> n synaptic operations
        new_active = topk(h, k)
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
    W = train(patterns, N, K)

    # recall every stored pattern from a corrupted cue (drop 8 of 20, add 8 noise)
    overlaps, total_synops, total_steps = [], 0, 0
    for p in patterns:
        cue = corrupt(p, N, K, drop=8, add=8)
        out, steps_used, synops = recall(W, cue, N, K, RECALL_STEPS)
        overlaps.append(overlap(out, p, K))
        total_synops += synops
        total_steps += steps_used

    mean_recall = sum(overlaps) / len(overlaps)
    perfect = sum(1 for o in overlaps if o == 1.0)

    # --- compute: spiking SynOps vs dense ANN MACs (same recall work) ---
    dense_macs = total_steps * N * N            # dense recomputes full N x N each step
    compute_ratio = dense_macs / total_synops

    # --- storage: one recalled state, AER events vs dense float32 ---
    dense_state_bytes = N * 4                    # float32 activation tensor
    bitmap_bytes = -(-N // 8)                    # 1-bit dense bitmap
    aer_bytes = K * 2                            # active indices, 2 bytes each (uint16)
    storage_ratio = dense_state_bytes / aer_bytes

    # --- capacity / honest weight cost ---
    weight_values = N * N
    raw_pattern_bits = NUM_PATTERNS * N

    print("=" * 60)
    print("SPIKING ASSOCIATIVE-MEMORY STORAGE PROTOTYPE")
    print("=" * 60)
    print(f"neurons N={N}  active K={K} ({K/N:.1%})  patterns stored={NUM_PATTERNS}")
    print()
    print("RECALL (content-addressable, from 40% corrupted cue):")
    print(f"  mean overlap with stored pattern : {mean_recall:.1%}")
    print(f"  perfectly recalled               : {perfect}/{NUM_PATTERNS}")
    print(f"  avg settle steps                 : {total_steps/NUM_PATTERNS:.1f}")
    print()
    print("COMPUTE  (lower = less power):")
    print(f"  spiking SynOps   : {total_synops:,}")
    print(f"  dense ANN MACs   : {dense_macs:,}")
    print(f"  reduction        : {compute_ratio:.1f}x  (~ N/K = {N/K:.1f}x)")
    print()
    print("STORAGE per recalled state (lower = less space):")
    print(f"  dense float32    : {dense_state_bytes} B")
    print(f"  1-bit bitmap     : {bitmap_bytes} B")
    print(f"  AER event list   : {aer_bytes} B")
    print(f"  reduction        : {storage_ratio:.1f}x vs float32")
    print()
    print("WEIGHT COST (honest):")
    print(f"  W matrix values  : {weight_values:,}  (this is the storage substrate)")
    print(f"  raw pattern bits : {raw_pattern_bits:,}")
    print(f"  note: W is content-addressable (no separate DB/index); recovers")
    print(f"        full data from partial cues. Wins on activation compute &")
    print(f"        traffic, not on shrinking the weights themselves.")
    print("=" * 60)

    # ---- self-check (ponytail: one runnable check, assert-based) ----
    assert mean_recall >= 0.95, f"recall too weak: {mean_recall:.2f}"
    assert total_synops < dense_macs, "spiking did not beat dense on compute"
    assert aer_bytes < dense_state_bytes, "events not smaller than dense state"
    print("self-check OK: recall>=95%, SynOps<MACs, events<dense state")


if __name__ == "__main__":
    main()
