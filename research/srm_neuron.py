"""
v0.44 — Spike Response Model (SRM) neuron: kernels + soft reset, vs hard-reset IF.

The repo's stdlib neurons are simple Integrate-and-Fire: hard reset to 0 on firing. SRM is a
richer abstraction — the membrane is a sum of KERNELS, not a state with a hard reset:

    u(t) = rest + Σ_inputs  w · eps(t - t_in)        # eps = PSP kernel (rise then decay)
                + Σ_own      eta(t - t_fire)          # eta = reset/refractory kernel (negative)

`eps` gives each input spike a temporal post-synaptic potential; `eta` suppresses the neuron
after its OWN spike (soft reset + refractoriness) instead of zeroing the state. This retains
recent temporal context, prevents runaway spike cascades, and yields more diverse, time-coded
firing — the temporal richness hard-reset IF lacks. stdlib + self-checked.

  python srm_neuron.py
"""
import math


class SRMNeuron:
    def __init__(self, tau_m=8.0, tau_s=2.0, tau_r=6.0, eta0=2.0, thresh=1.0, rest=0.0):
        self.tau_m, self.tau_s, self.tau_r = tau_m, tau_s, tau_r
        self.eta0, self.thresh, self.rest = eta0, thresh, rest
        self.inputs = []        # (t_in, weight)
        self.fires = []         # t_fire

    def eps(self, s):
        """PSP kernel: 0 at s<=0, rises to a peak, decays back to 0 (double-exponential)."""
        if s <= 0:
            return 0.0
        return math.exp(-s / self.tau_m) - math.exp(-s / self.tau_s)

    def eta(self, s):
        """Reset/refractory kernel: strong negative right after a spike, decaying to 0."""
        if s < 0:
            return 0.0
        return -self.eta0 * math.exp(-s / self.tau_r)

    def potential(self, t):
        u = self.rest
        u += sum(w * self.eps(t - ti) for ti, w in self.inputs if ti < t)
        u += sum(self.eta(t - tf) for tf in self.fires if tf < t)
        return u

    def step(self, t, in_weights=()):
        """Advance to time t with incoming weighted spikes; fire if potential >= threshold."""
        for w in in_weights:
            self.inputs.append((t, w))
        u = self.potential(t)
        if u >= self.thresh:
            self.fires.append(t)
            return True
        return False


class IFNeuron:
    """Plain integrate-and-fire with a HARD reset to 0 (the repo's current style)."""

    def __init__(self, thresh=1.0):
        self.thresh, self.u, self.fires = thresh, 0.0, []

    def step(self, t, in_weights=()):
        self.u += sum(in_weights)
        if self.u >= self.thresh:
            self.fires.append(t)
            self.u = 0.0                 # hard reset
            return True
        return False


def run(neuron, T, drive):
    """drive(t) -> tuple of incoming weights at step t. Returns list of fire times."""
    for t in range(1, T + 1):
        neuron.step(t, drive(t))
    return list(neuron.fires)


def main():
    srm = SRMNeuron()

    # 1. eps PSP kernel: zero at 0, rises to a peak, decays away
    e = [srm.eps(s) for s in range(0, 30)]
    peak = max(range(len(e)), key=lambda i: e[i])
    assert e[0] == 0.0 and 0 < peak < 20 and e[peak] > e[29], "eps is not a rise-then-decay PSP"

    # 2. eta refractory kernel: most negative right after a spike, decays to 0
    assert srm.eta(0) < srm.eta(5) < srm.eta(50) <= 0.0, "eta is not a decaying reset kernel"

    # 3. strong constant drive: SRM's refractory regulates firing; IF (hard reset) fires more
    def strong(t):
        return (0.6,)                    # steady input every step
    srm2, iff = SRMNeuron(), IFNeuron()
    srm_fires = run(srm2, 40, strong)
    if_fires = run(iff, 40, strong)
    # no two consecutive SRM spikes (eta refractoriness), and fewer total than hard-reset IF
    min_gap = min((b - a for a, b in zip(srm_fires, srm_fires[1:])), default=99)
    assert min_gap >= 2, f"SRM fired with no refractory gap (gap={min_gap})"
    assert len(srm_fires) < len(if_fires), "SRM refractory should regulate vs hard-reset IF"

    print("=" * 60)
    print("SRM NEURON (kernels + soft reset) vs hard-reset IF")
    print("=" * 60)
    print(f"eps PSP kernel : peak at s={peak}, decays to {e[29]:+.3f} (temporal PSP)")
    print(f"eta refractory : eta(0)={srm.eta(0):+.2f} -> eta(50)={srm.eta(50):+.2f}")
    print(f"strong drive   : SRM {len(srm_fires)} spikes (min gap {min_gap}) vs "
          f"IF {len(if_fires)} spikes (no refractory)")
    print("=" * 60)
    print("self-check OK: eps PSP rises+decays, eta enforces refractoriness (soft reset), "
          "SRM regulates firing where hard-reset IF runs away")


if __name__ == "__main__":
    main()
