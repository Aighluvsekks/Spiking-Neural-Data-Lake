"""
v0.50 — robot-arm calibration in ONE place.

Every distance/IR threshold the arm pipeline depends on was hardcoded across
gesture_recognition.py and live_arm.py, tuned to the builder's specific sensor mount.
Move the rig + the 95% recognition collapses. This centralizes them so a re-mount is a
one-file edit, and `bins_from_capture()` can derive them from real data instead of guessing.

Units: distance in metres (ultrasonic / Sensor_1). IR = Sensor_2 (raw analog, polarity
sensor-specific — see IR_PRESENT below).

  python arm_config.py        # self-check
"""

# ---- distance bins (metres) ----------------------------------------------------
# NEAR/MID/FAR channels feed Paradigm B's order-aware sequence detector.
NEAR_MAX = 0.30        # d <  NEAR_MAX            -> NEAR
MID_MAX = 0.60         # NEAR_MAX <= d < MID_MAX  -> MID   (else FAR)
IN_RANGE_MAX = 2.0     # d >= IN_RANGE_MAX        -> no object present (idle)

# ---- reflex --------------------------------------------------------------------
CONTACT = 0.04         # d < CONTACT -> imminent-contact reflex EMERGENCY_STOP
TRAJ_DEV_MAX = 0.15    # end-effector deviation (m) from the intended pose that trips the
                       # trajectory-deviation reflex (the "differential position comparator":
                       # an external perturbation knocks the arm off its commanded path).
                       # ponytail: tied to arm reach (l1+l2=2m); re-tune per real rig/gripper.

# ---- real-hardware safety gate -------------------------------------------------
# CONTACT / joint-limit / reflex values here are PLACEHOLDERS tuned to the sim, NOT your rig.
# serial_arm.SerialArm refuses to drive real servos until this is True. Flip it ONLY after
# replacing them with your arm's measured limits (docs/arduino_requirements.md #3).
SAFETY_CALIBRATED = False

# ---- windowing -----------------------------------------------------------------
W = 8                  # samples per recognition window
STRIDE = W             # W = non-overlapping; < W = sliding (lower live latency)

# ---- IR fusion (Sensor_2) ------------------------------------------------------
# ponytail: IR polarity/scale is sensor-specific and UNCALIBRATED, so fusion is OFF by
# default (None) — distance alone decides presence, exactly as before. Ceiling: idle vs
# far-but-present object is distance-only. Upgrade: capture IR with/without a hand in
# range, set IR_PRESENT to (">", v) or ("<", v) so IR confirms presence.
IR_PRESENT = None      # e.g. (">", 500) means "present when Sensor_2 > 500"

NEAR, MID, FAR = 0, 1, 2


def dbin(d):
    """Distance -> NEAR/MID/FAR channel, or None when out of range (idle)."""
    if d >= IN_RANGE_MAX:
        return None
    if d < NEAR_MAX:
        return NEAR
    if d < MID_MAX:
        return MID
    return FAR


def present(d, ir=None):
    """Is an object in range this sample? Distance-primary; IR confirms iff calibrated."""
    by_dist = d < IN_RANGE_MAX
    if IR_PRESENT is None or ir is None:
        return by_dist
    op, thr = IR_PRESENT
    by_ir = ir > thr if op == ">" else ir < thr
    return by_dist and by_ir       # both agree -> present (IR rejects distance ghosts)


def bins_from_capture(distances, lo=0.33, hi=0.66):
    """Derive NEAR/MID cut points from a real capture's in-range distances (percentiles),
    so the bins match the rig instead of magic constants. Returns (near_max, mid_max)."""
    xs = sorted(d for d in distances if d < IN_RANGE_MAX)
    if len(xs) < 3:
        return NEAR_MAX, MID_MAX
    return xs[int(lo * (len(xs) - 1))], xs[int(hi * (len(xs) - 1))]


def main():
    assert dbin(0.1) == NEAR and dbin(0.45) == MID and dbin(0.9) == FAR
    assert dbin(3.0) is None, "out-of-range must read as idle"
    assert present(0.5) and not present(2.5), "distance presence broken"
    # IR disabled by default -> distance alone decides
    assert present(0.5, ir=0) is True, "IR must be ignored while uncalibrated"
    near, mid = bins_from_capture([0.05, 0.1, 0.2, 0.4, 0.5, 0.7, 0.9])
    assert 0.0 < near <= mid < IN_RANGE_MAX, f"derived bins invalid: {near},{mid}"
    print("=" * 56)
    print("ARM CONFIG — calibration in one place")
    print("=" * 56)
    print(f"bins (m)   : NEAR<{NEAR_MAX}  MID<{MID_MAX}  in-range<{IN_RANGE_MAX}")
    print(f"reflex     : CONTACT={CONTACT} m   window W={W} stride={STRIDE}")
    print(f"IR fusion  : {'OFF (distance-only)' if IR_PRESENT is None else IR_PRESENT}")
    print(f"derived bins from sample capture: NEAR<{near:.3f} MID<{mid:.3f}")
    print("self-check OK: bins, presence, IR-gating, capture-derived bins")


if __name__ == "__main__":
    main()
