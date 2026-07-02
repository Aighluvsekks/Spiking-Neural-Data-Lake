"""
v0.51 — RBF population encoding of a continuous scalar (Gemini brief: "Population Rate
Encoding via Radial Basis Functions").

Single-neuron rate/latency coding (one value -> one firing rate) is brittle: noise on the
one channel corrupts the whole reading. A *population* spreads one scalar across N neurons
with overlapping Gaussian receptive fields — the value is read from WHICH neurons fire and
how hard, so noise on any one neuron barely moves the decoded estimate. This is the standard
neuromorphic way to feed continuous signals (joint angle, distance, IR) into an SNN.

  x  -> [g_0(x), g_1(x), ... g_{N-1}(x)]   (Gaussian activations, g_i peaks at center c_i)
       -> spike counts over a window       (activation * max_spikes, rounded)
  counts -> activation-weighted centroid of the centers  -> x_hat

Pure stdlib (math only). Does NOT replace the verified distance-bin path in arm_config /
gesture_recognition — it's an additional, noise-robust front-end for downstream SNN consumers.

  python population_encoding.py        # self-check
"""
import math
import random


class PopulationEncoder:
    """N Gaussian receptive fields tiling [lo, hi]; encode/decode a scalar."""

    def __init__(self, n=8, lo=0.0, hi=2.0, max_spikes=10, overlap=1.0):
        if n < 2 or hi <= lo:
            raise ValueError("need n>=2 and hi>lo")
        self.n, self.lo, self.hi, self.max_spikes = n, lo, hi, max_spikes
        step = (hi - lo) / (n - 1)
        self.centers = [lo + i * step for i in range(n)]
        # width tied to spacing so adjacent fields overlap (smooth, gap-free coverage)
        self.sigma = step * overlap

    def activations(self, x):
        """Gaussian response of every neuron to x (continuous, in [0,1])."""
        s2 = 2.0 * self.sigma * self.sigma
        return [math.exp(-((x - c) ** 2) / s2) for c in self.centers]

    def encode(self, x):
        """Scalar -> per-neuron spike counts over the window (the population code)."""
        return [int(round(a * self.max_spikes)) for a in self.activations(x)]

    def decode(self, counts):
        """Spike counts -> scalar estimate (activation-weighted centroid of centers)."""
        total = sum(counts)
        if total == 0:                      # nothing fired -> out of range, no estimate
            return None
        return sum(c * k for c, k in zip(self.centers, counts)) / total


def _add_noise(counts, jitter, rng):
    """Per-neuron count jitter (+/- jitter spikes), clamped at 0 — models spike noise."""
    return [max(0, k + rng.randint(-jitter, jitter)) for k in counts]


def main():
    enc = PopulationEncoder(n=8, lo=0.0, hi=2.0, max_spikes=12)
    rng = random.Random(0)

    # 1) clean encode->decode round-trips across the range
    xs = [0.1 * i for i in range(1, 20)]            # 0.1 .. 1.9 (inside range)
    errs = [abs(enc.decode(enc.encode(x)) - x) for x in xs]
    mean_err = sum(errs) / len(errs)

    # 2) noise robustness: population vs single-neuron threshold encoding
    #    single-neuron: one rate = x/hi * max_spikes; decode = rate/max_spikes*hi
    def single_err(x):
        rate = int(round(x / enc.hi * enc.max_spikes))
        rate_n = max(0, rate + rng.randint(-3, 3))
        return abs(rate_n / enc.max_spikes * enc.hi - x)

    pop_noisy = [abs(enc.decode(_add_noise(enc.encode(x), 3, rng)) - x) for x in xs]
    sng_noisy = [single_err(x) for x in xs]
    pop_mean, sng_mean = sum(pop_noisy) / len(xs), sum(sng_noisy) / len(xs)

    print("=" * 60)
    print("RBF POPULATION ENCODING (continuous scalar -> spikes -> scalar)")
    print("=" * 60)
    print(f"neurons={enc.n} centers[{enc.lo},{enc.hi}] sigma={enc.sigma:.3f} max_spikes={enc.max_spikes}")
    print(f"clean round-trip mean abs error : {mean_err:.4f}  (range span {enc.hi-enc.lo})")
    print(f"under noise: population {pop_mean:.4f}  vs single-neuron {sng_mean:.4f}")
    print(f"out-of-range x=5.0 -> counts {enc.encode(5.0)} decode {enc.decode(enc.encode(5.0))}")
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    assert mean_err < 0.05, f"clean round-trip too lossy: {mean_err:.4f}"
    assert pop_mean < sng_mean, "population code should beat single-neuron under noise"
    assert enc.decode(enc.encode(5.0)) is None or enc.decode([0] * enc.n) is None, \
        "all-silent population must decode to None (out of range)"
    assert enc.encode(1.0)[enc.n // 2 - 1:enc.n // 2 + 1], "mid-range must excite middle neurons"
    print(f"self-check OK: round-trip {mean_err:.4f}, noise-robust ({pop_mean:.4f}<{sng_mean:.4f})")


if __name__ == "__main__":
    main()
