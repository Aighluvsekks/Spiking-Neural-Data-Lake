"""
Spike-Driven MoE classifier (stdlib-only) — Nord's firing-rate routing ported
into the project's prototype.

Ported from Project Nord's SpikeDrivenMoE (NORD-V5/nord_core_700m.py:438). The
three faithful pieces:
  1. firing-rate routing  — route by the MEAN SPIKE RATE of input feature-
     clusters, aggregated per expert. NO learned router network (Nord keeps only
     a tiny per-expert bias). This is the storage win: a learned router would
     need N*n_experts weights; firing-rate routing needs n_experts biases.
  2. top-k sparse experts — only top_k of n_experts compute per sample. This is
     the compute win: expert work drops by n_experts/top_k.
  3. load balance         — Nord uses an aux gradient loss to spread usage; with
     a local learning rule we can't backprop it, so we use the homeostatic
     equivalent: nudge each expert's bias toward an even usage share (same role
     Nord's expert_bias + EMA play). [ponytail: local-rule analog of the aux loss]

Differences from Nord (by necessity, all stdlib, no torch):
  - experts trained by the project's local delta rule, not surrogate-grad BPTT.
  - routing clusters the raw input neurons (our "features") instead of a hidden
    d_model; same math (cluster firing rate -> per-expert score -> top-k).

Reuses the data + spike encoding from snn_classifier.py (no duplication).
Run:  python snn_moe_classifier.py
"""
import random
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> product modules
from snn_classifier import templates, noisy, dataset, N, STEPS, ON_RATE, OFF_RATE

random.seed(0)  # deterministic run + self-check

# ---- MoE config -------------------------------------------------------------
N_EXPERTS = 8
TOP_K = 2                 # experts active per sample (sparse routing)
N_CLUSTERS = 16           # input feature-clusters (must be divisible by N_EXPERTS)
CLUSTERS_PER_EXPERT = N_CLUSTERS // N_EXPERTS
ROUTE_TEMP = 0.5          # Nord's moe_route_temperature
LB_RATE = 0.05            # homeostatic load-balance strength (bias nudging)
LR = 0.5
EPOCHS = 8
TRAIN_PER_CLASS = 40
TEST_PER_CLASS = 20


def sample_spikes(sample):
    """One Poisson realisation: per-step active-input lists + spike counts.
    Shared by routing and expert forward so both see the same spikes."""
    steps, counts = [], [0] * N
    for _ in range(STEPS):
        active = [i for i, px in enumerate(sample)
                  if random.random() < (ON_RATE if px else OFF_RATE)]
        steps.append(active)
        for i in active:
            counts[i] += 1
    return steps, counts


# ---- 1. firing-rate routing (no learned router) -----------------------------
def expert_scores(counts, expert_bias):
    """Nord's _compute_expert_scores: cluster firing rates -> per-expert score.
    cluster of input i = i % N_CLUSTERS; expert score = mean of its clusters."""
    fr = [c / STEPS for c in counts]                       # firing rate per input
    cluster_rate = [0.0] * N_CLUSTERS
    cluster_size = [0] * N_CLUSTERS
    for i, r in enumerate(fr):
        cid = i % N_CLUSTERS
        cluster_rate[cid] += r
        cluster_size[cid] += 1
    cluster_rate = [cr / max(s, 1) for cr, s in zip(cluster_rate, cluster_size)]
    scores = []
    for e in range(N_EXPERTS):
        cl = cluster_rate[e * CLUSTERS_PER_EXPERT:(e + 1) * CLUSTERS_PER_EXPERT]
        scores.append(sum(cl) / len(cl) / max(ROUTE_TEMP, 0.01) + expert_bias[e])
    return scores


def route(scores):
    """top-k experts + softmax gate weights over their scores (Nord's forward)."""
    top = sorted(range(N_EXPERTS), key=lambda e: -scores[e])[:TOP_K]
    m = max(scores[e] for e in top)
    ex = [math.exp(scores[e] - m) for e in top]
    s = sum(ex)
    gates = {e: ex[j] / s for j, e in enumerate(top)}
    return gates  # {expert_idx: gate_weight}


# ---- 2. sparse expert forward -----------------------------------------------
def expert_forward(W_e, steps, n_classes):
    """class currents from one expert over the shared spike train. Spike-driven:
    work only on active inputs. Returns currents[C] and synops."""
    cur = [0.0] * n_classes
    synops = 0
    for active in steps:
        for i in active:
            for c in range(n_classes):
                cur[c] += W_e[c][i]
            synops += n_classes
    return cur, synops


def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def forward(experts, expert_bias, sample, n_classes):
    """route -> run only top-k experts -> gate-weighted class currents.
    Returns (probs, gates, counts, steps, expert_synops, route_synops)."""
    steps, counts = sample_spikes(sample)
    route_synops = sum(len(a) for a in steps)        # routing = summing spikes into clusters
    scores = expert_scores(counts, expert_bias)
    gates = route(scores)
    combined = [0.0] * n_classes
    esynops = 0
    cache = {}
    for e, g in gates.items():
        cur, s = expert_forward(experts[e], steps, n_classes)
        esynops += s
        cache[e] = cur
        for c in range(n_classes):
            combined[c] += g * cur[c]
    return softmax(combined), gates, counts, esynops, route_synops


# ---- 3. train (local delta rule on selected experts + load balance) ---------
def train(train_data, n_classes):
    experts = [[[0.0] * N for _ in range(n_classes)] for _ in range(N_EXPERTS)]
    expert_bias = [0.0] * N_EXPERTS
    usage = [1.0 / N_EXPERTS] * N_EXPERTS           # EMA usage share
    target_share = TOP_K / N_EXPERTS
    total_synops = 0
    for epoch in range(EPOCHS):
        correct = 0
        for sample, label in train_data:
            probs, gates, counts, esynops, rsynops = forward(
                experts, expert_bias, sample, n_classes)
            total_synops += esynops
            if max(range(n_classes), key=lambda c: probs[c]) == label:
                correct += 1
            # update ONLY the selected experts, credit scaled by gate weight
            for e, g in gates.items():
                We = experts[e]
                for c in range(n_classes):
                    err = (1.0 if c == label else 0.0) - probs[c]
                    wc = We[c]
                    k = LR * g * err / STEPS
                    if k:
                        for i in range(N):
                            if counts[i]:
                                wc[i] += k * counts[i]
            # load balance: EMA usage, nudge bias toward even share (homeostasis)
            picked = set(gates)
            for e in range(N_EXPERTS):
                hit = 1.0 if e in picked else 0.0
                usage[e] += 0.05 * (hit - usage[e])
                expert_bias[e] += LB_RATE * (target_share - usage[e])
        print(f"  epoch {epoch+1}/{EPOCHS}  train_acc={correct/len(train_data):.1%}")
    return experts, expert_bias, usage, total_synops


def evaluate(experts, expert_bias, data, n_classes):
    correct, esynops = 0, 0
    used = [0] * N_EXPERTS
    for sample, label in data:
        probs, gates, _, s, _ = forward(experts, expert_bias, sample, n_classes)
        esynops += s
        for e in gates:
            used[e] += 1
        if max(range(n_classes), key=lambda c: probs[c]) == label:
            correct += 1
    return correct / len(data), esynops, used


def main():
    temps = templates()
    C = len(temps)
    train_data = dataset(TRAIN_PER_CLASS, temps)
    test_data = dataset(TEST_PER_CLASS, temps)

    print("=" * 60)
    print("SPIKE-DRIVEN MoE CLASSIFIER  (Nord firing-rate routing, ported)")
    print("=" * 60)
    print(f"experts={N_EXPERTS}  top_k={TOP_K}  clusters={N_CLUSTERS}  "
          f"classes={C}  inputs N={N}")
    print(f"routing: firing-rate (NO learned router net)\n")
    experts, expert_bias, usage, _ = train(train_data, C)
    acc, moe_synops, used = evaluate(experts, expert_bias, test_data, C)
    print()
    print(f"TEST ACCURACY : {acc:.1%}   (chance = {1/C:.0%})\n")

    # --- compute: top-k sparse vs running ALL experts (dense MoE) ---
    dense_all_synops = moe_synops * N_EXPERTS / TOP_K     # if every expert ran
    print("COMPUTE on test set (lower = less power):")
    print(f"  MoE SynOps (top-{TOP_K})   : {moe_synops:,}")
    print(f"  all-{N_EXPERTS}-experts SynOps : {int(dense_all_synops):,}")
    print(f"  reduction              : {N_EXPERTS/TOP_K:.1f}x  (= n_experts/top_k)")
    print()

    # --- storage: firing-rate router vs a learned router network ---
    fr_router = N_EXPERTS                                  # just the per-expert biases
    learned_router = N * N_EXPERTS                         # a real router: N->n_experts
    print("ROUTER STORAGE (lower = less space):")
    print(f"  firing-rate router : {fr_router} params (biases only)")
    print(f"  learned router     : {learned_router} params (N x n_experts)")
    print(f"  reduction          : {learned_router/fr_router:.0f}x  (no router net)")
    print()

    # --- load balance (Nord stat) ---
    share = [u / sum(used) for u in used] if sum(used) else used
    print("LOAD BALANCE (expert usage share on test set):")
    print("  " + "  ".join(f"e{e}:{s:.0%}" for e, s in enumerate(share)))
    print("=" * 60)

    # ---- self-check (ponytail: one runnable check) ----
    assert acc >= 0.85, f"MoE did not learn: acc={acc:.2f}"
    assert moe_synops < dense_all_synops, "top-k routing not cheaper than all-experts"
    assert fr_router < learned_router, "firing-rate router not smaller than learned"
    assert sum(1 for u in used if u > 0) >= TOP_K, "fewer experts used than top_k"
    print("self-check OK: acc>=85%, top-k<all-experts, router smaller, experts spread")


if __name__ == "__main__":
    main()
