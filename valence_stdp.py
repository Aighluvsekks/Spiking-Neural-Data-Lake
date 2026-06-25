"""
v0.34 — reward-modulated STDP: LEARNED valence  (#3).

The reflex (reflex.py) is hardwired. This LEARNS which signals are good or bad from
outcomes — a dopamine-like reward — then acts instinctively on what it learned.

A single valence neuron sums weighted spike-counts of the encoded signal into a scalar
valence in [-1, +1]. After each action an outcome reward r in [-1, +1] modulates the
weights along an eligibility trace (R-STDP): inputs active when the reward arrived are
strengthened (r > 0) or weakened (r < 0). Over trials, signals that precede bad outcomes
drive valence negative -> AVOID; good ones positive -> APPROACH; unlearned/ambiguous
stays near zero -> neutral (defer to the matcher).

This is the learned complement to the reflex: instinct you are born with vs instinct you
acquire from consequences.

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

    def act(self, window):
        v = self.valence(window)
        if v >= THETA:
            return "APPROACH", v
        if v <= -THETA:
            return "AVOID", v
        return None, v          # neutral: let the recognition matcher decide

    def learn(self, window, reward):
        """R-STDP: w[i] += lr * reward * eligibility[i] (active inputs at reward time)."""
        c = self._active(window)
        norm = sum(c) or 1
        for i in range(self.n):
            if c[i]:
                self.w[i] += self.lr * reward * (c[i] / norm)


def _pattern(seed):
    r = random.Random(seed)
    return [r.random() if r.random() < 0.4 else 0.0 for _ in range(N)]


def _jitter(vec, rng, noise=0.08):
    return [min(1.0, max(0.0, v + rng.uniform(-noise, noise))) for v in vec]


def main():
    rng = random.Random(0)
    good, bad = _pattern(1), _pattern(2)              # two distinct signals
    vl = ValenceLearner()

    vg0, vb0 = vl.valence(good), vl.valence(bad)      # untrained -> neutral

    print("=" * 60)
    print("REWARD-MODULATED STDP — learned valence (good vs bad)")
    print("=" * 60)
    print(f"untrained valence : good {vg0:+.2f}  bad {vb0:+.2f}  (both ~neutral)")
    print(f"{'trials':>6} | {'good valence':>12} | {'bad valence':>11}")
    print("-" * 40)
    for trial in range(1, 201):
        vl.learn(_jitter(good, rng), +1.0)            # good signal -> reward
        vl.learn(_jitter(bad, rng), -1.0)             # bad signal  -> punishment
        if trial % 50 == 0:
            print(f"{trial:>6} | {vl.valence(good):>+12.2f} | {vl.valence(bad):>+11.2f}")

    vg, vb = vl.valence(good), vl.valence(bad)
    ag, ab = vl.act(good)[0], vl.act(bad)[0]
    print("-" * 40)
    print(f"learned actions   : good -> {ag}   bad -> {ab}")
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    assert abs(vg0) < THETA and abs(vb0) < THETA, "untrained valence should be neutral"
    assert vg > THETA and vb < -THETA, f"valence did not separate good/bad: {vg:+.2f}/{vb:+.2f}"
    assert ag == "APPROACH" and ab == "AVOID", f"wrong learned actions: {ag}/{ab}"
    # the good signal is now clearly more positive than the bad one (learning happened)
    assert vg - vb > 1.0, "good vs bad valence gap too small"
    print("self-check OK: started neutral, learned good->APPROACH / bad->AVOID from reward")


if __name__ == "__main__":
    main()
