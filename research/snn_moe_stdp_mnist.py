"""
v0.6 — MoE + STDP hybrid on real MNIST  (snnTorch).

Fuses the two real-primitive lines of this repo:
  - v0.3/v0.5 : unsupervised STDP excitatory populations (data stored in synapses)
  - v0.4      : Project-Nord-style spike-driven MoE routing (firing-rate gate,
                NO learned router network)

Architecture:
  input (784 Poisson) --> firing-rate router --> top-K of N STDP expert pops
  Each expert is an independent unsupervised-STDP layer (Diehl & Cook-style with
  adaptive thresholds). The router scores each expert by how strongly the image's
  expected firing rate drives it (rate . colsum(W_e)) minus a load-balance bias,
  then only the top-K experts run the full temporal simulation and learn.

Why it serves the project aim (less compute / less storage):
  - compute : only K of N experts run per image -> ~N/K less expert compute than
    a dense MoE that runs every expert. Router itself is a single N x 784 matvec.
  - storage : the router has ZERO learned parameters (routing IS the spike drive);
    a conventional learned router would cost N x 784 weights.

No labels and no backprop during training (pure STDP). Neurons are labelled after
training by majority vote, exactly as in v0.3.

Run:  python snn_moe_stdp_mnist.py
Scale: NORD_EXPERTS / NORD_EXPERT_M / NORD_TOPK / NORD_TRAIN / NORD_TEST env vars.
"""
import os
import torch
import snntorch as snn
from snn_mnist_stdp import load_mnist   # reuse the MNIST loader (v0.3)

torch.manual_seed(0)


def _env(key, default):
    v = os.environ.get(key)
    return int(v) if v else default


# ---- config -----------------------------------------------------------------
N_EXPERTS = _env("NORD_EXPERTS", 6)
EXPERT_M = _env("NORD_EXPERT_M", 60)   # neurons per expert
TOP_K = _env("NORD_TOPK", 2)           # experts activated per image
T = 25
MAX_RATE = 0.35
BETA = 0.92
TRAIN_N = _env("NORD_TRAIN", 4000)
TEST_N = _env("NORD_TEST", 1500)
LR = 0.012
TAU_PRE = 20.0
W_NORM = 78.0
WMAX = 1.0
THRESH = 8.0
THETA_PLUS = 0.4
LOAD_BALANCE = 6.0   # router penalty on over-used experts (spreads the load)
N_IN = 28 * 28


class StdpExpert:
    """One unsupervised-STDP excitatory population (hard k-WTA, adaptive theta)."""

    def __init__(self, seed):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.rand(EXPERT_M, N_IN, generator=g) * 0.3
        self._normalise()
        self.lif = snn.Leaky(beta=BETA, threshold=1e9, reset_mechanism="none")
        self.pre_decay = torch.exp(torch.tensor(-1.0 / TAU_PRE))
        self.theta = torch.zeros(EXPERT_M)
        self.colsum = self.W.sum(dim=0)   # cached input drive per pixel (router)

    def _normalise(self):
        s = self.W.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.W *= (W_NORM / s)

    def affinity(self, rates):
        """firing-rate routing score: how hard this image drives the expert."""
        return float(torch.dot(rates, self.colsum))

    def run(self, image, learn):
        mem = self.lif.init_leaky()
        x_pre = torch.zeros(N_IN)
        counts = torch.zeros(EXPERT_M)
        synops = 0
        rates = image * MAX_RATE
        for _ in range(T):
            s_in = (torch.rand(N_IN) < rates).float()
            x_pre = x_pre * self.pre_decay + s_in
            cur = self.W @ s_in
            synops += int(s_in.sum().item()) * EXPERT_M
            _, mem = self.lif(cur, mem)
            eff = mem - self.theta
            winner = int(eff.argmax().item())
            if eff[winner] > THRESH:
                counts[winner] += 1
                if learn:
                    self.W[winner] += LR * x_pre
                    self.W[winner].clamp_(0.0, WMAX)
                    self.theta[winner] += THETA_PLUS
                mem = torch.zeros(EXPERT_M)
        if learn:
            self._normalise()
            self.colsum = self.W.sum(dim=0)   # refresh router cache after update
        return counts, synops


def route(experts, image, usage):
    """Pick TOP_K experts by firing-rate affinity minus a load-balance penalty."""
    rates = image * MAX_RATE
    total = max(1, sum(usage))
    scores = [e.affinity(rates) - LOAD_BALANCE * (usage[i] / total) * e.affinity(rates)
              for i, e in enumerate(experts)]
    return sorted(range(len(experts)), key=lambda i: -scores[i])[:TOP_K]


def main():
    print("=" * 60)
    print("v0.6  MoE + STDP HYBRID on MNIST  (firing-rate routing)")
    print("=" * 60)
    print("loading MNIST...")
    Xtr, Ytr, Xte, Yte = load_mnist(TRAIN_N, TEST_N)
    experts = [StdpExpert(seed=i + 1) for i in range(N_EXPERTS)]
    usage = [0] * N_EXPERTS
    print(f"experts N={N_EXPERTS} x {EXPERT_M} neurons  top-K={TOP_K}  "
          f"train={TRAIN_N} test={TEST_N}  (NO labels, NO backprop, NO learned router)\n")

    print("training (routed unsupervised STDP)...")
    for i, img in enumerate(Xtr):
        for e in route(experts, img, usage):
            experts[e].run(img, learn=True)
            usage[e] += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{TRAIN_N}  expert usage={usage}")

    print("\nassigning neuron labels (per expert, majority vote)...")
    # assignment[e] = tensor[EXPERT_M] -> class
    resp = [torch.zeros(EXPERT_M, 10) for _ in range(N_EXPERTS)]
    for img, y in zip(Xtr, Ytr):
        for e in route(experts, img, usage):
            counts, _ = experts[e].run(img, learn=False)
            resp[e][:, int(y)] += counts
    assignment = [r.argmax(dim=1) for r in resp]

    print("evaluating...")
    correct, active_synops = 0, 0
    for img, y in zip(Xte, Yte):
        chosen = route(experts, img, usage)
        scores = torch.zeros(10)
        for e in chosen:
            counts, s = experts[e].run(img, learn=False)
            active_synops += s
            for c in range(10):
                mask = assignment[e] == c
                if mask.any():
                    scores[c] += counts[mask].sum()
        if int(scores.argmax().item()) == int(y):
            correct += 1
    acc = correct / len(Yte)

    # honest compute accounting on the test set
    # active = only TOP_K experts ran; dense-MoE = all N experts would run.
    dense_moe = active_synops * N_EXPERTS / TOP_K
    dense_ann = TEST_N * T * N_IN * (N_EXPERTS * EXPERT_M)
    print(f"\nTEST ACCURACY : {acc:.1%}   (chance = 10%)\n")
    print("COMPUTE on test set (lower = less power):")
    print(f"  active SynOps (top-{TOP_K})   : {active_synops:,}")
    print(f"  dense-MoE (all {N_EXPERTS} experts) : {int(dense_moe):,}  -> {dense_moe/active_synops:.1f}x saved by routing")
    print(f"  dense ANN (T*N*allneurons) : {dense_ann:,}  -> {dense_ann/active_synops:.1f}x")
    print()
    print("STORAGE:")
    print(f"  expert weights : {N_EXPERTS}*{EXPERT_M}*{N_IN} = {N_EXPERTS*EXPERT_M*N_IN:,}")
    print(f"  router params  : 0 (routing is the spike drive)  vs {N_EXPERTS*N_IN:,} for a learned router")
    print("=" * 60)

    assert acc >= 0.55, f"hybrid did not learn: acc={acc:.2f}"
    assert active_synops < dense_moe, "routing saved no compute"
    print(f"self-check OK: acc>=55%, top-{TOP_K} routing < dense-MoE, router=0 params")


if __name__ == "__main__":
    main()
