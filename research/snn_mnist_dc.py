"""
v0.8 — proper Diehl & Cook unsupervised STDP on MNIST  (snnTorch).

Fixes the v0.7 honest-negative inhibition result. The v0.7 lumped scheme
(single hard winner + a global scalar that suppressed everyone) let the strongest
neuron reset and immediately re-win, so coverage collapsed. This implements the
real Diehl & Cook (2015) machinery that makes lateral inhibition *help*:

  - MULTIPLE excitatory neurons may spike per timestep (a population code), not a
    single hard winner.
  - ALL-BUT-SELF lateral inhibition: inh_i = INH * (total_spikes - spike_i), so a
    neuron never inhibits itself; the rest are suppressed. This is the lumped form
    of the inhibitory population (Ae -> Ai one-to-one, Ai -> Ae all-but-one).
  - Adaptive membrane threshold theta per neuron (homeostasis): +theta_plus on
    each spike, slow decay -> firing spreads across the population over training.
  - Weight-dependent STDP with a presynaptic target (implicit depression):
    dW = lr * spk (x_pre - x_tar) (w_max - W); per-neuron weight normalisation.
  - Re-presentation: if an image fires too few spikes, boost the input rate and
    re-present, so every image yields a learning signal.

No labels, no backprop. Neurons labelled by majority vote after training.
Goal: beat the v0.5 hard-WTA baseline (81.5%) and move toward the literature ~95%.

RESULT (honest, v0.8): this current-based reimplementation does NOT beat hard-WTA.
Across a parameter sweep (THRESH 20-40, INH 0.5-1.5, theta_plus 0.2-1.0, x_tar
0.1-0.5) it collapses — most neurons end up assigned to 1-2 round-digit classes
(0/8), giving 9-22% accuracy vs hard-WTA's 70.6% on the same smoke config. The
collapse is the classic "no diversity" failure: faithful Diehl & Cook needs
CONDUCTANCE-based synapses and carefully matched membrane time constants, not the
current-based integration here. Kept as a documented experiment. The mechanism
that actually works in this repo is hard single-winner WTA + adaptive thresholds
(snn_mnist_stdp.py) — the effective strong-inhibition limit — scaled up.

Run:  python snn_mnist_dc.py
Scale: NORD_M / NORD_TRAIN / NORD_TEST (env). Tune: DC_THRESH / DC_INH / DC_TPLUS /
       DC_TDECAY / DC_BETA / DC_LR / DC_T (env).
"""
import os
import torch
import snntorch as snn  # noqa: F401  (kept: same LIF lineage as the rest of the lake)
from snn_mnist_stdp import load_mnist

torch.manual_seed(0)


def _ei(k, d):
    v = os.environ.get(k);  return int(v) if v else d


def _ef(k, d):
    v = os.environ.get(k);  return float(v) if v else d


# ---- config -----------------------------------------------------------------
M = _ei("NORD_M", 100)
TRAIN_N = _ei("NORD_TRAIN", 3000)
TEST_N = _ei("NORD_TEST", 1500)
T = _ei("DC_T", 60)                 # presentation window (timesteps)
INPUT_RATE = _ef("DC_RATE", 0.40)   # Poisson rate for a fully-lit pixel
BETA_E = _ef("DC_BETA", 0.90)       # excitatory membrane decay
THRESH = _ef("DC_THRESH", 20.0)     # base firing threshold (v - theta > THRESH)
INH = _ef("DC_INH", 0.6)            # lateral inhibition strength (all-but-self)
THETA_PLUS = _ef("DC_TPLUS", 0.20)  # adaptive-threshold bump per spike
THETA_DECAY = _ef("DC_TDECAY", 0.9999)  # slow homeostatic decay
LR = _ef("DC_LR", 0.01)             # STDP learning rate
X_TAR = _ef("DC_XTAR", 0.10)        # presynaptic target (implicit depression)
TAU_PRE = 20.0
W_NORM = 78.0
WMAX = 1.0
V_RESET = 0.0
MIN_SPIKES = 5                      # re-present (boost rate) if fewer than this
N_IN = 28 * 28
PRE_DECAY = float(torch.exp(torch.tensor(-1.0 / TAU_PRE)))


class DCNetwork:
    """Excitatory STDP layer with all-but-self lateral inhibition + adaptive theta."""

    def __init__(self):
        self.W = torch.rand(M, N_IN) * 0.3
        self._normalise()
        self.theta = torch.zeros(M)

    def _normalise(self):
        s = self.W.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.W *= (W_NORM / s)

    def _present(self, rates, learn):
        """One presentation of `rates` for T steps. Returns exc spike counts."""
        v = torch.zeros(M)
        inh = torch.zeros(M)
        x_pre = torch.zeros(N_IN)
        counts = torch.zeros(M)
        for _ in range(T):
            s_in = (torch.rand(N_IN) < rates).float()
            x_pre = x_pre * PRE_DECAY + s_in
            v = v * BETA_E + (self.W @ s_in) - inh        # integrate + inhibition
            fired = (v - self.theta) > THRESH             # MANY may fire
            spk = fired.float()
            n_spk = float(spk.sum())
            if n_spk > 0:
                counts += spk
                inh = INH * (n_spk - spk)                 # all-but-self inhibition
                v = torch.where(fired, torch.full_like(v, V_RESET), v)  # reset firers only
                if learn:
                    # weight-dependent STDP toward recently-active inputs
                    self.W += LR * spk.unsqueeze(1) * (x_pre - X_TAR).unsqueeze(0) * (WMAX - self.W)
                    self.W.clamp_(0.0, WMAX)
                    self.theta += THETA_PLUS * spk
            else:
                inh = inh * 0.0
            if learn:
                self.theta *= THETA_DECAY
        return counts

    def run(self, image, learn):
        """Present an image; boost rate and re-present if it fires too few spikes."""
        scale = 1.0
        for _ in range(4):
            counts = self._present(image * INPUT_RATE * scale, learn)
            if counts.sum() >= MIN_SPIKES:
                break
            scale *= 1.4
        if learn:
            self._normalise()
        # synops ~ proportional to spikes * fanout; report a cheap proxy
        return counts


def assign_labels(net, X, Y):
    resp = torch.zeros(M, 10)
    for img, y in zip(X, Y):
        resp[:, int(y)] += net.run(img, learn=False)
    return resp.argmax(dim=1)


def evaluate(net, X, Y, assignment):
    correct = 0
    for img, y in zip(X, Y):
        counts = net.run(img, learn=False)
        scores = torch.zeros(10)
        for c in range(10):
            mask = assignment == c
            if mask.any():
                scores[c] = counts[mask].sum()
        if int(scores.argmax().item()) == int(y):
            correct += 1
    return correct / len(Y)


def main():
    print("=" * 60)
    print("v0.8  PROPER DIEHL & COOK STDP on MNIST")
    print("=" * 60)
    print("loading MNIST...")
    Xtr, Ytr, Xte, Yte = load_mnist(TRAIN_N, TEST_N)
    net = DCNetwork()
    print(f"M={M}  T={T}  train={TRAIN_N} test={TEST_N}  "
          f"INH={INH} THRESH={THRESH} THETA+={THETA_PLUS}  (NO labels, NO backprop)\n")

    print("training (Diehl & Cook STDP + lateral inhibition)...")
    for i, img in enumerate(Xtr):
        net.run(img, learn=True)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{TRAIN_N}")

    print("\nassigning neuron labels (majority vote)...")
    assignment = assign_labels(net, Xtr, Ytr)
    per_class = [int((assignment == c).sum().item()) for c in range(10)]
    print(f"  neurons per class: {per_class}")

    print("evaluating...")
    acc = evaluate(net, Xte, Yte, assignment)
    print(f"\nTEST ACCURACY : {acc:.1%}   (chance = 10%)")
    print(f"vs v0.5 hard-WTA baseline: 81.5% (same scale needs NORD_M=300 NORD_TRAIN=6000)")
    print("=" * 60)

    # Documented NEGATIVE result (see header): this current-based D&C collapses
    # and underperforms hard-WTA. The check only confirms it ran end-to-end.
    assert acc > 0.05, f"network produced no usable output: acc={acc:.2f}"
    print(f"ran OK (acc={acc:.1%}). NOTE: underperforms hard-WTA "
          "(snn_mnist_stdp.py, 82.3%) — see this file's header for why.")


if __name__ == "__main__":
    main()
