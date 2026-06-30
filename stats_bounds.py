"""
v0.44 — FASER-style statistical assertion bounds for non-deterministic tests.

FASER (ICSE'23) is a research artifact, not a package. Its INSIGHT is buildable: a stochastic
metric (SNN accuracy over seeds, STDP convergence) should be asserted within a statistically
derived band (mean ± k·sigma) rather than a hand-picked floor. Hand-picked floors are set
loose to avoid flaky failures — and a loose floor silently misses subtle regressions. A tight
statistical band flags genuine regressions while tolerating expected stochastic variance.

  python stats_bounds.py
"""
import math


def mean_std(samples):
    n = len(samples)
    if n == 0:
        raise ValueError("no samples")
    m = sum(samples) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in samples) / (n - 1)      # sample variance
    return m, math.sqrt(var)


def bound(samples, k=3.0):
    """Statistical acceptance band [mean - k*sigma, mean + k*sigma] for a metric."""
    m, s = mean_std(samples)
    return (m - k * s, m + k * s)


def within(value, samples, k=3.0):
    lo, hi = bound(samples, k)
    return lo <= value <= hi


def assert_within(value, samples, k=3.0, name="metric"):
    lo, hi = bound(samples, k)
    if not (lo <= value <= hi):
        raise AssertionError(f"{name}={value:.4f} outside statistical band "
                             f"[{lo:.4f}, {hi:.4f}] (mean±{k}sigma over {len(samples)} runs)")
    return True


def floor_misses_regression(samples, hand_floor, regressed_value, k=3.0):
    """FASER's point: does a hand-picked floor MISS a regression the statistical band catches?
    Returns True when floor passes the regression but the band rejects it."""
    lo, _ = bound(samples, k)
    floor_passes = regressed_value >= hand_floor
    band_rejects = regressed_value < lo
    return floor_passes and band_rejects


def main():
    # a stochastic metric: accuracy ~ 0.80 with small run-to-run noise (deterministic generator)
    base = [0.80, 0.81, 0.79, 0.82, 0.78, 0.80, 0.81, 0.79, 0.80, 0.82]
    m, s = mean_std(base)
    lo, hi = bound(base, k=3.0)

    print("=" * 60)
    print("STATISTICAL ASSERTION BOUNDS (FASER-style)")
    print("=" * 60)
    print(f"metric over {len(base)} runs : mean={m:.3f} sigma={s:.3f}")
    print(f"acceptance band (3sigma) : [{lo:.3f}, {hi:.3f}]")

    # a normal new run passes; a genuine regression is rejected
    ok_run, regressed = 0.805, 0.62
    print(f"new run {ok_run}  -> {'PASS' if within(ok_run, base) else 'FAIL'}")
    print(f"regressed {regressed} -> {'PASS' if within(regressed, base) else 'REJECTED'}")

    # the FASER point: a loose hand-picked floor (0.5) would have MISSED this regression
    hand_floor = 0.50
    missed = floor_misses_regression(base, hand_floor, regressed)
    print(f"hand floor {hand_floor}: regression 0.62 passes the floor but the band rejects it "
          f"-> floor would MISS the bug = {missed}")
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    assert within(ok_run, base), "a normal run should pass the band"
    assert not within(regressed, base), "a real regression should be rejected"
    assert_within(ok_run, base, name="accuracy")
    assert missed, "loose hand-floor should miss a regression the statistical band catches"
    try:
        assert_within(regressed, base, name="accuracy")
        raise AssertionError("assert_within should have raised on the regression")
    except AssertionError as e:
        assert "outside statistical band" in str(e)
    print("self-check OK: band passes normal runs, rejects regressions, beats a loose floor")


if __name__ == "__main__":
    main()
