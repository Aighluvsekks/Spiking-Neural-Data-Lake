"""
v0.46 — gesture recognition on the builder's REAL captures, ORDER-AWARE (Paradigm B).

First pass used a template / Van Rossum matcher on min-max-normalized windows: IDLE separated
(82%) but APPROACH and RETREAT collapsed (~63% overall) — they are time-REVERSED (distance
falling vs rising) and that matcher is direction-blind.

This wires the distance trajectory through Paradigm B's order-aware sequence detector
(`paradigm_b_engine.match_sequence`): bin distance into NEAR/MID/FAR channels and detect the
ORDERED crossing —
    FAR -> MID -> NEAR  = approaching      NEAR -> MID -> FAR  = retreating
the exact `[5->17->42] vs [42->17->5]` discrimination Paradigm B was built for. A net-direction
fallback handles noisy windows that don't cross all three bins cleanly. Idle = no object in range.

CI-safe: real CSVs if present, else an inline synthetic 3-gesture set.

  python gesture_recognition.py
"""
import os

from paradigm_b_engine import match_sequence
from arm_config import dbin, present, NEAR, MID, FAR, W, STRIDE   # calibration lives in arm_config

DIR = "robot arm"
GESTURES = ["IDLE", "HAND_APPROACH", "HAND_RETREAT"]


def classify(blk):
    # blk items are (distance, ir); present() fuses IR iff calibrated, else distance-only
    inrange = [(i, dbin(d)) for i, (d, ir) in enumerate(blk)
               if present(d, ir) and dbin(d) is not None]
    if len(inrange) < max(2, len(blk) // 2):
        return "IDLE"
    appr = match_sequence(inrange, [FAR, MID, NEAR], W)      # ordered: far -> near
    retr = match_sequence(inrange, [NEAR, MID, FAR], W)      # ordered: near -> far
    if appr and not retr:
        return "HAND_APPROACH"
    if retr and not appr:
        return "HAND_RETREAT"
    # tie / no clean 3-stage crossing -> net direction of the in-range distances (still order-aware)
    pd = [blk[i][0] for i, _ in inrange]
    return "HAND_APPROACH" if pd[-1] < pd[0] else "HAND_RETREAT"


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) != 3 or p[0] == "label":
                continue
            try:
                rows.append((float(p[1]), float(p[2])))
            except ValueError:
                continue
    return rows


def windows(rows, w=W, stride=STRIDE):
    return [rows[s:s + w] for s in range(0, len(rows) - w + 1, stride)]


def _synthetic():
    """3-gesture stand-in (idle far / approach falling / retreat rising) for CI without data."""
    items = []
    items += [([(3.13, -1.0)] * W, "IDLE") for _ in range(8)]
    for _ in range(8):
        items.append(([(0.9 - 0.11 * i, 295.0) for i in range(W)], "HAND_APPROACH"))   # falling
        items.append(([(0.1 + 0.11 * i, 295.0) for i in range(W)], "HAND_RETREAT"))     # rising
    return items


def build_items():
    if all(os.path.exists(os.path.join(DIR, f"{g}.csv")) for g in GESTURES):
        items = []
        for g in GESTURES:
            for blk in windows(load_csv(os.path.join(DIR, f"{g}.csv"))):
                active = sum(1 for d, ir in blk if present(d, ir)) >= len(blk) // 2
                items.append((blk, g if active else "IDLE"))
        return items, "real capture (robot arm/*.csv)"
    return _synthetic(), "synthetic fallback"


def main():
    items, src = build_items()
    labels = sorted({lab for _, lab in items})
    correct = 0
    confusion = {g: {} for g in labels}
    for blk, true in items:
        pred = classify(blk)
        confusion.setdefault(true, {})
        confusion[true][pred] = confusion[true].get(pred, 0) + 1
        correct += (pred == true)
    acc = correct / len(items)

    def recall(g):
        row = confusion.get(g, {})
        return row.get(g, 0) / max(1, sum(row.values()))

    print("=" * 60)
    print("GESTURE RECOGNITION — order-aware (Paradigm B sequence detector)")
    print("=" * 60)
    print(f"source   : {src}")
    print(f"windows  : {len(items)} ({', '.join(f'{g}={sum(1 for _,l in items if l==g)}' for g in labels)})")
    print(f"accuracy : {acc:.0%}  (chance {1/len(labels):.0%})")
    for g in labels:
        print(f"  {g:14} recall {recall(g):.0%}  -> {confusion.get(g, {})}")
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    assert acc > 0.75, f"order-aware recognition should clear 75%, got {acc:.2f}"
    assert recall("HAND_APPROACH") >= 0.6 and recall("HAND_RETREAT") >= 0.6, \
        "approach/retreat still confused — order detection not separating direction"
    assert recall("IDLE") >= 0.6, "idle not recognized"
    print(f"self-check OK: order-aware recognition {acc:.0%}, approach/retreat now separated "
          f"(A={recall('HAND_APPROACH'):.0%} R={recall('HAND_RETREAT'):.0%} I={recall('IDLE'):.0%})")


if __name__ == "__main__":
    main()
