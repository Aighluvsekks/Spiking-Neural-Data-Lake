"""
v0.35 — cortisol stress-state: slow tonic neuromodulation.

The slow counterpart to fast phasic dopamine (valence_stdp.py) and the millisecond reflex
(reflex.py). A single global `stress` scalar integrates aversive events (reflex fires,
negative dopamine/RPE) with a LONG time constant and decays slowly = recovery — the
HPA-axis cortisol abstraction (minutes-hours), not the fight-or-flight reflex (milliseconds).

High stress globally reshapes the other systems:
  - hypervigilance  : LOWERS the reflex threshold  -> protective actions fire sooner
  - stress learning : RAISES the aversive learning rate -> bad events stick harder
  - caution bias    : shifts valence toward AVOID -> risk-averse while stressed

Together: phasic dopamine (fast reward learning) + tonic cortisol (slow stress state) =
the fast/slow neuromodulator pair the brain actually uses.

  python cortisol.py
"""
import os
import math

TAU = float(os.environ.get("CORT_TAU", 200.0))      # slow decay (steps) — the "minutes" scale
GAIN = float(os.environ.get("CORT_GAIN", 0.30))     # how much one aversive event raises stress
MAXMOD = float(os.environ.get("CORT_MAXMOD", 0.60)) # cap on how strongly stress modulates


class Cortisol:
    def __init__(self, tau=TAU, gain=GAIN, maxmod=MAXMOD):
        self.decay = math.exp(-1.0 / tau)
        self.gain = gain
        self.maxmod = maxmod
        self.level = 0.0

    def step(self, aversive=0.0):
        """Advance one tick. `aversive` in [0,1] = severity of a bad event this tick
        (reflex fired, |negative dopamine|, ...). Leaky integrator -> stress in [0,1]."""
        self.level = self.level * self.decay + self.gain * max(0.0, aversive)
        self.level = min(1.0, self.level)
        return self.level

    # ---- how stress reshapes the other systems ------------------------------
    def reflex_threshold_scale(self):
        """< 1 under stress -> lower reflex threshold -> hypervigilance."""
        return 1.0 - self.maxmod * self.level

    def learn_rate_scale(self, signed):
        """> 1 for aversive signals (signed < 0) under stress -> bad memories stick;
        ~1 for appetitive. `signed` = reward or dopamine sign."""
        return 1.0 + self.maxmod * self.level if signed < 0 else 1.0

    def caution_bias(self):
        """Subtract from valence in act() -> risk-averse (more AVOID) when stressed."""
        return self.maxmod * self.level


def demo_episode():
    """Reflex + valence + cortisol coupled through a stress episode:
    quiet -> danger burst (stress rises, reflex sharpens, aversive learning amplified)
    -> quiet (recovery)."""
    from snn_data_lake.reflex import Reflex
    from snn_data_lake.valence_stdp import ValenceLearner, _pattern, _jitter
    import random

    rng = random.Random(0)
    cort = Cortisol(tau=40.0)            # short tau so the demo runs quickly
    base_threshold = 0.10
    danger = [500, 500, 500, 500, 500, 500, 950, 0]   # collision on ch6

    levels = []
    # phase A — 20 quiet ticks: stress stays near zero
    for _ in range(20):
        levels.append(cort.step(0.0))
    quiet = cort.level

    # phase B — 30 danger ticks: reflex fires, stress climbs, threshold sharpens
    reflex = Reflex()
    fires = 0
    for _ in range(30):
        reflex.threshold = base_threshold * cort.reflex_threshold_scale()   # hypervigilance
        fired = reflex.step(danger)
        if fired:
            fires += 1
        levels.append(cort.step(1.0 if fired else 0.0))
    stressed = cort.level

    # phase C — 120 quiet ticks: recovery
    for _ in range(120):
        levels.append(cort.step(0.0))
    recovered = cort.level

    return {"quiet": quiet, "stressed": stressed, "recovered": recovered,
            "fires": fires, "levels": levels}


def main():
    ep = demo_episode()
    stressed = Cortisol()
    for _ in range(40):                  # drive a separate cortisol to high stress
        stressed.step(1.0)

    print("=" * 60)
    print("CORTISOL STRESS-STATE (slow tonic modulation)")
    print("=" * 60)
    print(f"episode  quiet={ep['quiet']:.2f} -> danger burst -> stressed={ep['stressed']:.2f}"
          f" -> recovered={ep['recovered']:.2f}  ({ep['fires']} reflex fires)")
    print(f"under stress (level={stressed.level:.2f}):")
    print(f"  reflex threshold scale : {stressed.reflex_threshold_scale():.2f}  (<1 = fires sooner)")
    print(f"  learn-rate scale  bad  : {stressed.learn_rate_scale(-1):.2f}  "
          f"good : {stressed.learn_rate_scale(+1):.2f}")
    print(f"  caution bias           : {stressed.caution_bias():.2f}  (subtracted from valence)")
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    assert ep["quiet"] < 0.2, "stress should stay low when safe"
    assert ep["stressed"] > 0.5, "danger burst should raise stress"
    assert ep["recovered"] < ep["stressed"] * 0.5, "stress should decay (recover) when safe again"
    assert stressed.reflex_threshold_scale() < 1.0, "stress should lower the reflex threshold"
    assert stressed.learn_rate_scale(-1) > stressed.learn_rate_scale(+1), \
        "stress should amplify aversive learning specifically"
    assert stressed.caution_bias() > 0.0, "stress should add a caution (AVOID) bias"
    print("self-check OK: stress rises on danger, recovers when safe, and under stress "
          "sharpens reflex + amplifies aversive learning + biases toward caution")


if __name__ == "__main__":
    main()
