"""
v0.35 — reward-PREDICTION-ERROR STDP: dopamine-driven learned valence  (#3, upgraded).

The reflex (reflex.py) is hardwired. This LEARNS which signals are good or bad from
outcomes, the way dopamine does: a single valence neuron's output is the predicted value
of a signal (in [-1, +1]); learning is driven by the reward PREDICTION ERROR, not the raw
reward —

    dopamine = reward - predicted_value          (the RPE / TD error)
    w += lr * dopamine * eligibility             (three-factor plasticity)

This buys the real dopamine behaviours that raw-reward R-STDP could not:
  - acquisition : dopamine is large on the first (unexpected) reward, then SHRINKS toward
                  zero as the value is learned — exactly Schultz's dopamine-fires-on-surprise.
  - extinction  : a once-rewarded signal that stops paying out gives a NEGATIVE dopamine
                  dip (reward - high_prediction < 0) -> the value decays back to neutral.
  - omission    : expected reward withheld -> dopamine dip below baseline.

act() -> APPROACH / AVOID / neutral (neutral defers to the matcher); a `bias` arg lets
cortisol (cortisol.py) push the decision toward AVOID under stress, and `lr_scale` lets it
amplify aversive learning. Phasic dopamine here + tonic cortisol there = the fast/slow
neuromodulator pair.

  python valence_stdp.py
"""
import os
import math
import random

from spike_preprocessing import encode_latency, N

LR = float(os.environ.get("VAL_LR", 0.05))
THETA = float(os.environ.get("VAL_THETA", 0.30))   # |valence| above this -> act


class ValenceLearner:
    def __init__(self, n=N, lr=LR):
        self.w = [0.0] * n
        self.n = n
        self.lr = lr

    def _active(self, window):
        """Which input features fired (deterministic latency encode -> 1 spike/feature)."""
        c = [0] * self.n
        for _, i in encode_latency(window):
            if i < self.n:
                c[i] = 1
        return c

    def valence(self, window):
        c = self._active(window)
        s = sum(self.w[i] * c[i] for i in range(self.n))
        return math.tanh(s)

    def act(self, window, bias=0.0):
        """bias < 0 (e.g. cortisol caution under stress) shifts the decision toward AVOID."""
        v = self.valence(window) - bias
        if v >= THETA:
            return "APPROACH", v
        if v <= -THETA:
            return "AVOID", v
        return None, v          # neutral: let the recognition matcher decide

    def dopamine(self, window, reward):
        """Reward PREDICTION ERROR: delta = reward - predicted value. This is the dopamine
        signal — fires positive on better-than-expected, dips negative on omission/worse,
        and goes to ~0 once the value is learned (no surprise -> no dopamine)."""
        return reward - self.valence(window)

    def learn(self, window, reward, lr_scale=1.0):
        """Three-factor plasticity driven by the RPE (not raw reward): converges when the
        prediction matches reward (delta -> 0) and EXTINGUISHES when a once-rewarded signal
        stops paying out (delta < 0 -> weights decay). lr_scale lets cortisol amplify
        aversive learning under stress. Returns the dopamine signal (delta)."""
        delta = self.dopamine(window, reward)
        c = self._active(window)
        norm = sum(c) or 1
        for i in range(self.n):
            if c[i]:
                self.w[i] += self.lr * lr_scale * delta * (c[i] / norm)
        return delta


def _pattern(seed):
    r = random.Random(seed)
    return [r.random() if r.random() < 0.4 else 0.0 for _ in range(N)]


def _jitter(vec, rng, noise=0.08):
    return [min(1.0, max(0.0, v + rng.uniform(-noise, noise))) for v in vec]


def main():
    rng = random.Random(0)
    good, bad = _pattern(1), _pattern(2)
    vl = ValenceLearner()

    print("=" * 64)
    print("RPE DOPAMINE — prediction error drives learning (not raw reward)")
    print("=" * 64)
    print(f"untrained value : good {vl.valence(good):+.2f}  bad {vl.valence(bad):+.2f}  (neutral)\n")

    # 1. ACQUISITION — dopamine is big on the first reward, shrinks as the value is learned
    print("acquisition (good, reward=+1): dopamine should SHRINK as value is learned")
    acq = []
    for t in range(1, 41):
        d = vl.learn(_jitter(good, rng), +1.0)
        acq.append(d)
        if t in (1, 2, 5, 10, 20, 40):
            print(f"  trial {t:>2}: dopamine={d:+.3f}  value={vl.valence(good):+.3f}")
    for _ in range(40):
        vl.learn(_jitter(bad, rng), -1.0)             # also learn the bad signal
    v_good = vl.valence(good)
    act_good_acq = vl.act(good)[0]          # action AFTER acquisition (before extinction below)
    print(f"learned: good -> {act_good_acq} ({v_good:+.2f})   "
          f"bad -> {vl.act(bad)[0]} ({vl.valence(bad):+.2f})\n")

    # 2. EXTINCTION — good now pays nothing: dopamine dips, value decays toward neutral
    print("extinction (good, reward=0 now): dopamine dips, value decays")
    dips = []
    for t in range(1, 41):
        d = vl.learn(_jitter(good, rng), 0.0)
        dips.append(d)
        if t in (1, 10, 40):
            print(f"  trial {t:>2}: dopamine={d:+.3f}  value={vl.valence(good):+.3f}")
    v_ext = vl.valence(good)
    print(f"good value {v_good:+.2f} -> {v_ext:+.2f}; action now {vl.act(good)[0]}")
    print("=" * 64)

    # ---- self-checks --------------------------------------------------------
    assert acq[0] > 0.5, "first reward should be a big positive surprise (dopamine)"
    assert acq[-1] < acq[0] * 0.6, "dopamine must shrink as the value is learned (RPE)"
    assert v_good > THETA and act_good_acq == "APPROACH", "good not learned as APPROACH"
    assert vl.act(bad)[0] == "AVOID", "bad not learned as AVOID"
    assert all(d <= 0 for d in dips[:3]), "omitted reward must produce a dopamine dip (<=0)"
    assert v_ext < v_good and v_ext < THETA, "extinction should decay the value below APPROACH"
    assert vl.act(good)[0] is None, "good should be neutral after extinction"
    print("self-check OK: dopamine shrinks with learning, dips on omission, value extinguishes")


if __name__ == "__main__":
    main()
