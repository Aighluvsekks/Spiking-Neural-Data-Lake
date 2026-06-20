"""
Unsupervised STDP MNIST classifier  (real MNIST + snnTorch) — Diehl & Cook 2015
style, the recipe the research docs describe ("Two-Layer MNIST STDP, Brian2").

Combines the two requested directions:
  (1) real MNIST data + snnTorch LIF neurons
  (3) TRUE unsupervised STDP: no labels and no backprop during training. Input->
      excitatory weights are written purely by spike-timing correlation; a
      winner-take-all competition forces each excitatory neuron to specialise on
      a digit prototype. Labels are attached only AFTER training, by majority
      vote (each neuron is assigned the class it fires most for).

How it embodies the project's storage method:
  - long-term storage : the STDP-written weight matrix W[exc, 784]. Each row is a
    learned digit prototype — the data lives in the synapses.
  - compute           : rate-coded sparse binary spikes; a synaptic op happens
    only when an input pixel fires. Reported against the dense same-architecture
    cost (T * 784 * M), which the sparse version beats by ~1/spike_rate.

Run (after deps):  ../.venv/Scripts/python snn_mnist_stdp.py
Deps: numpy torch torchvision snntorch   (CPU build is fine)

Honest scope: simplified vs full Diehl&Cook (hard k-WTA instead of an explicit
inhibitory population; fixed thresholds, no adaptive homeostasis). Upgrade path
noted at THRESH/INHIB below. Expect ~70-85% with 100 neurons on a subset, not
the paper's 95% (which needs the full machinery + all 60k images).
"""
import torch
import snntorch as snn
from torchvision import datasets, transforms

torch.manual_seed(0)  # deterministic run + self-check

# ---- config (tune here) -----------------------------------------------------
M = 100            # excitatory neurons (= number of learned prototypes)
T = 30             # timesteps per image
MAX_RATE = 0.35    # Poisson spike prob for a fully-lit pixel, per step
BETA = 0.92        # LIF membrane decay (snnTorch Leaky)
TRAIN_N = 3000     # training images (unsupervised)
TEST_N = 1500      # test images
LR = 0.012         # STDP learning rate
TAU_PRE = 20.0     # pre-synaptic trace time constant (steps)
W_NORM = 78.0      # target sum of each neuron's incoming weights (homeostasis)
WMAX = 1.0         # weight clip
THRESH = 8.0       # base membrane firing threshold
THETA_PLUS = 0.4   # adaptive-threshold bump per spike (homeostasis - the fix
                   #   that forces neurons to specialise instead of one winning all)
THETA_DECAY = 1.0  # theta persistence during training (1.0 = no decay)
N_IN = 28 * 28
# ponytail upgrade path: replace hard k-WTA with an explicit snn.Leaky inhibitory
# population for the last few % of Diehl&Cook accuracy.


def load_mnist(n_train, n_test):
    tf = transforms.Compose([transforms.ToTensor()])
    tr = datasets.MNIST(root="./data", train=True, download=True, transform=tf)
    te = datasets.MNIST(root="./data", train=False, download=True, transform=tf)
    Xtr = tr.data[:n_train].float().view(n_train, -1) / 255.0
    Ytr = tr.targets[:n_train]
    Xte = te.data[:n_test].float().view(n_test, -1) / 255.0
    Yte = te.targets[:n_test]
    return Xtr, Ytr, Xte, Yte


class StdpNetwork:
    """Input(784 Poisson) -> Excitatory(M, snnTorch LIF) with STDP + hard k-WTA."""

    def __init__(self):
        # weights init: small random positive, then normalised
        self.W = torch.rand(M, N_IN) * 0.3
        self._normalise()
        # snn.Leaky with an unreachable threshold: we use it purely for the leaky
        # membrane integration, then apply our own adaptive-threshold + WTA.
        self.lif = snn.Leaky(beta=BETA, threshold=1e9, reset_mechanism="none")
        self.pre_decay = torch.exp(torch.tensor(-1.0 / TAU_PRE))
        self.theta = torch.zeros(M)   # per-neuron adaptive threshold (homeostasis)

    def _normalise(self):
        # homeostasis: keep each neuron's total incoming weight constant.
        s = self.W.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.W *= (W_NORM / s)

    def run(self, image, learn):
        """Simulate one image for T steps. Returns exc spike-count vector [M] and
        the number of synaptic operations performed (spike-driven)."""
        mem = self.lif.init_leaky()
        x_pre = torch.zeros(N_IN)
        counts = torch.zeros(M)
        synops = 0
        rates = image * MAX_RATE
        for _ in range(T):
            s_in = (torch.rand(N_IN) < rates).float()      # Poisson input spikes
            x_pre = x_pre * self.pre_decay + s_in          # pre-synaptic trace
            cur = self.W @ s_in                            # sparse: drives exc
            synops += int(s_in.sum().item()) * M
            _, mem = self.lif(cur, mem)                    # snnTorch leaky integration
            # adaptive-threshold hard WTA: the single strongest neuron, AFTER its
            # own homeostatic threshold, fires if it clears the base threshold.
            eff = mem - self.theta
            winner = int(eff.argmax().item())
            if eff[winner] > THRESH:
                counts[winner] += 1
                if learn:
                    # STDP potentiation: move winner's weights toward recently
                    # active inputs (pre-before-post), then renormalise.
                    self.W[winner] += LR * x_pre
                    self.W[winner].clamp_(0.0, WMAX)
                    self.theta[winner] += THETA_PLUS       # raise its own bar
                mem = torch.zeros(M)                        # WTA reset all
            if learn:
                self.theta *= THETA_DECAY
        if learn:
            self._normalise()
        return counts, synops


def assign_labels(net, X, Y):
    """After unsupervised training, label each neuron by the class it responds
    to most strongly (averaged spike count per class)."""
    resp = torch.zeros(M, 10)
    per_class = torch.zeros(10)
    for img, y in zip(X, Y):
        counts, _ = net.run(img, learn=False)
        resp[:, int(y)] += counts
        per_class[int(y)] += 1
    resp /= per_class.clamp_min(1)
    return resp.argmax(dim=1)          # neuron -> assigned class


def evaluate(net, X, Y, assignment):
    correct, synops = 0, 0
    for img, y in zip(X, Y):
        counts, s = net.run(img, learn=False)
        synops += s
        # predicted class = class whose assigned neurons fired most
        scores = torch.zeros(10)
        for c in range(10):
            mask = assignment == c
            if mask.any():
                scores[c] = counts[mask].sum()
        if int(scores.argmax().item()) == int(y):
            correct += 1
    return correct / len(Y), synops


def main():
    print("=" * 58)
    print("UNSUPERVISED STDP MNIST  (snnTorch, Diehl&Cook style)")
    print("=" * 58)
    print("loading MNIST...")
    Xtr, Ytr, Xte, Yte = load_mnist(TRAIN_N, TEST_N)
    net = StdpNetwork()
    print(f"neurons M={M}  steps T={T}  train={TRAIN_N}  test={TEST_N}")
    print(f"max spike rate={MAX_RATE}  lr={LR}  (NO labels, NO backprop)\n")

    print("training (unsupervised STDP)...")
    for i, img in enumerate(Xtr):
        net.run(img, learn=True)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{TRAIN_N} images")

    print("\nassigning neuron labels by majority vote...")
    assignment = assign_labels(net, Xtr, Ytr)
    counts_per_class = [int((assignment == c).sum().item()) for c in range(10)]
    print(f"  neurons per class: {counts_per_class}")

    print("\nevaluating...")
    acc, test_synops = evaluate(net, Xte, Yte, assignment)
    print(f"\nTEST ACCURACY : {acc:.1%}   (chance = 10%)\n")

    dense_macs = TEST_N * T * N_IN * M     # dense same-arch over T steps
    ratio = dense_macs / test_synops if test_synops else float("inf")
    print("COMPUTE on test set (spiking vs dense same-architecture):")
    print(f"  spiking SynOps : {test_synops:,}")
    print(f"  dense T*N*M    : {dense_macs:,}")
    print(f"  reduction      : {ratio:.1f}x  (from input-spike sparsity)")
    print()
    print("STORAGE:")
    print(f"  prototypes in W : {M} x {N_IN} = {M*N_IN:,} weights (the stored data)")
    print(f"  activations     : 1-bit spikes vs 32-bit floats (32x)")
    print("=" * 58)

    # self-check: unsupervised STDP must beat chance by a wide margin
    assert acc >= 0.55, f"STDP did not learn usable prototypes: acc={acc:.2f}"
    assert test_synops < dense_macs, "spiking did not beat dense same-arch"
    print("self-check OK: acc>=55% (vs 10% chance), SynOps<dense")


if __name__ == "__main__":
    main()
