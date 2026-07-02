"""
v0.16 — precomputed deterministic-latency STDP, head-to-head vs rate coding.

Applies the v0.15 preprocessing recommendations to the real STDP model and COMPARES,
on the same MNIST subset, in one process:
  - RATE  (baseline) : the verified StdpNetwork — Poisson spikes generated on the fly
                       every timestep (stochastic; ~82% at M=300/6k tuned).
  - FAST  (this file): deterministic LATENCY coding, PRECOMPUTED once and reused — each
                       pixel emits ONE spike at a time set by its intensity (no RNG in
                       the loop). Far fewer input spikes per image -> fewer SynOps.

What we expect: FAST is deterministic, does less spike work, and trains faster; the
open question (the reason to compare) is whether accuracy holds vs rate coding.

Run:  python snn_mnist_stdp_fast.py
Config via env: NORD_M / NORD_TRAIN / NORD_TEST (shared with snn_mnist_stdp),
                FAST_THRESH (latency firing threshold), SKIP_RATE=1 (fast only).
"""
import os
import time
import torch
from snn_mnist_stdp import (load_mnist, StdpNetwork, assign_labels, evaluate,
                            M, T, TRAIN_N, TEST_N, N_IN, BETA, LR, W_NORM, WMAX,
                            TAU_PRE, THETA_PLUS, THETA_DECAY)

torch.manual_seed(0)
# v0.17 defaults = the gap-closer config (76.0% at M=300/6k, vs 68.8% pure latency).
# Set FAST_BURST=1 FAST_XTAR=0 FAST_THRESH=0.7 FAST_LR=0.10 to recover v0.16 pure latency.
FAST_THRESH = float(os.environ.get("FAST_THRESH", "2.0"))   # higher: burst raises membrane current
FAST_LR = float(os.environ.get("FAST_LR", "0.05"))
FAST_XTAR = float(os.environ.get("FAST_XTAR", "0.05"))      # LTD depression (Diehl&Cook x_tar) -> sharper prototypes
FAST_BURST = int(os.environ.get("FAST_BURST", "4"))         # spikes/pixel ∝ intensity: restores magnitude info + trace
# v0.18 pair-based STDP kernel: pre-triggered LTD via a post-synaptic trace.
# An input arriving AFTER the neuron fired (anti-causal) depresses that synapse,
# in proportion to how recently the neuron fired (x_post). Available as an opt-in,
# but DEFAULT 0: benchmarking showed it DEGRADES accuracy here (see CHANGELOG v0.18),
# so the v0.17 LTP+x_tar rule stays the default. Set FAST_AMINUS>0 to experiment.
FAST_AMINUS = float(os.environ.get("FAST_AMINUS", "0.0"))
TAU_POST = float(os.environ.get("TAU_POST", "20.0"))
FLOOR = 0.10
PRE_DECAY = float(torch.exp(torch.tensor(-1.0 / TAU_PRE)))
POST_DECAY = float(torch.exp(torch.tensor(-1.0 / TAU_POST)))


def encode_latency(img):
    """Deterministic latency encoding. FAST_BURST=1 -> one spike/pixel (brighter =
    earlier). FAST_BURST>1 -> graded burst: a bright pixel emits up to FAST_BURST
    spikes (count ∝ intensity) starting at its latency, restoring magnitude info and
    a stronger STDP trace while staying deterministic."""
    idx_all = (img >= FLOOR).nonzero(as_tuple=True)[0]
    base = ((1.0 - img[idx_all]) * (T - 1)).round().long()
    if FAST_BURST <= 1:
        return base, idx_all
    nb = (img[idx_all] * FAST_BURST).round().long().clamp(min=1)
    times, idx = [], []
    for j in range(idx_all.numel()):
        p, t0, k = int(idx_all[j]), int(base[j]), int(nb[j])
        for s in range(k):
            if t0 + s < T:
                times.append(t0 + s); idx.append(p)
    return torch.tensor(times, dtype=torch.long), torch.tensor(idx, dtype=torch.long)


class FastStdpNetwork:
    """Same LIF + adaptive-threshold + hard-WTA STDP as StdpNetwork, but driven by a
    precomputed latency spike tensor instead of on-the-fly Poisson."""

    def __init__(self):
        self.W = torch.rand(M, N_IN) * 0.3
        self._normalise()
        self.theta = torch.zeros(M)

    def _normalise(self):
        s = self.W.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.W *= (W_NORM / s)

    def run(self, times, idx, learn):
        mem = torch.zeros(M)
        x_pre = torch.zeros(N_IN)
        x_post = torch.zeros(M)                            # post-synaptic trace (pair STDP)
        counts = torch.zeros(M)
        synops = 0
        for t in range(T):
            x_pre = x_pre * PRE_DECAY
            x_post = x_post * POST_DECAY
            sel = times == t
            if bool(sel.any()):
                fi = idx[sel]
                if learn and FAST_AMINUS > 0:              # pair LTD: pre after post -> depress
                    self.W[:, fi] -= FAST_AMINUS * x_post.unsqueeze(1)
                    self.W[:, fi].clamp_(0.0, WMAX)
                x_pre[fi] += 1.0
                mem = mem * BETA + self.W[:, fi].sum(dim=1)   # spike-driven current
                synops += fi.numel() * M
            else:
                mem = mem * BETA
            eff = mem - self.theta
            w = int(eff.argmax().item())
            if eff[w] > FAST_THRESH:
                counts[w] += 1
                if learn:
                    # pair LTP: pre before post -> potentiate by causal pre-trace,
                    # minus x_tar baseline depression for sharper prototypes
                    self.W[w] += FAST_LR * (x_pre - FAST_XTAR)
                    self.W[w].clamp_(0.0, WMAX)
                    self.theta[w] += THETA_PLUS
                    x_post[w] += 1.0                       # mark this neuron just fired
                mem = torch.zeros(M)
            if learn:
                self.theta *= THETA_DECAY
        if learn:
            self._normalise()
        return counts, synops


def fast_assign(net, enc, Y):
    resp = torch.zeros(M, 10)
    for (tm, ix), y in zip(enc, Y):
        resp[:, int(y)] += net.run(tm, ix, learn=False)[0]
    return resp.argmax(dim=1)


def fast_eval(net, enc, Y, asg):
    correct, synops = 0, 0
    for (tm, ix), y in zip(enc, Y):
        c, s = net.run(tm, ix, learn=False)
        synops += s
        scores = torch.zeros(10)
        for cl in range(10):
            mask = asg == cl
            if mask.any():
                scores[cl] = c[mask].sum()
        correct += int(scores.argmax().item()) == int(y)
    return correct / len(Y), synops


def main():
    print("=" * 64)
    print("v0.16  PRECOMPUTED LATENCY STDP vs RATE STDP")
    print("=" * 64)
    print(f"M={M}  T={T}  train={TRAIN_N}  test={TEST_N}  FAST_THRESH={FAST_THRESH}\n")
    Xtr, Ytr, Xte, Yte = load_mnist(TRAIN_N, TEST_N)
    n_asg = min(TRAIN_N, 6000)

    rate = None
    if os.environ.get("SKIP_RATE") != "1":
        print("RATE baseline (Poisson, on-the-fly)...")
        net = StdpNetwork()
        t0 = time.perf_counter()
        tr_syn = 0
        for img in Xtr:
            tr_syn += net.run(img, learn=True)[1]
        rate_train = time.perf_counter() - t0
        asg = assign_labels(net, Xtr[:n_asg], Ytr[:n_asg])
        rate_acc, _ = evaluate(net, Xte, Yte, asg)
        rate = (rate_acc, rate_train, tr_syn)
        print(f"  acc={rate_acc:.1%}  train={rate_train:.1f}s  train SynOps={tr_syn:,}\n")

    print("FAST (deterministic latency, precomputed)...")
    t0 = time.perf_counter()
    enc_tr = [encode_latency(img) for img in Xtr]
    enc_te = [encode_latency(img) for img in Xte]
    pre_t = time.perf_counter() - t0
    fnet = FastStdpNetwork()
    t0 = time.perf_counter()
    fsyn = 0
    for tm, ix in enc_tr:
        fsyn += fnet.run(tm, ix, learn=True)[1]
    fast_train = time.perf_counter() - t0
    asg = fast_assign(fnet, enc_tr[:n_asg], Ytr[:n_asg])
    fast_acc, _ = fast_eval(fnet, enc_te, Yte, asg)
    print(f"  precompute={pre_t:.1f}s  acc={fast_acc:.1%}  train={fast_train:.1f}s  "
          f"train SynOps={fsyn:,}\n")

    print("=" * 64)
    print("COMPARISON")
    print("=" * 64)
    if rate:
        ra, rt, rs = rate
        print(f"  accuracy   : rate {ra:.1%}   vs   fast {fast_acc:.1%}   (Δ {100*(fast_acc-ra):+.1f} pts)")
        print(f"  train time : rate {rt:.1f}s  vs   fast {fast_train:.1f}s  ({rt/max(0.01,fast_train):.1f}x)")
        print(f"  train SynOps: rate {rs:,}  vs  fast {fsyn:,}  ({rs/max(1,fsyn):.1f}x fewer)")
    print(f"  fast is deterministic (no RNG) + cacheable; rate is stochastic.")
    print("=" * 64)

    assert fast_acc >= 0.55, f"fast latency STDP did not learn: {fast_acc:.2f}"
    print("self-check OK: fast latency STDP learns (>=55% vs 10% chance)")


if __name__ == "__main__":
    main()
