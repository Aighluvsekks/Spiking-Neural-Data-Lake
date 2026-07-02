"""
v0.39 — synthetic dataset modeling the builder's sensor.ino (ultrasonic distance + IR temp).

The builder shipped only the sketch (2 channels: distance in metres + MLX90614 object temp,
115200, 10 Hz, temp = -1 when nothing within 3 m — the agreed idle fix for his "___" token).
No recorded data yet. This generates a labelled dataset in HIS exact wire format so we can
enroll / train / test the pipeline now, then swap in his real captures unchanged.

HONEST: synthetic — models the sensor's physics + noise, not real readings. It proves the
pipeline on his 2-channel domain and is class-separable by construction; real numbers need
his hardware. Gestures are what THIS sensor can actually sense (proximity + temperature):

  HAND_APPROACH : distance falls 3.5 -> 0.1 m, a ~33 C hand enters range
  HAND_RETREAT  : distance rises 0.1 -> 3.5 m, temp drops out of range (-1)
  HOT_OBJECT    : object held close (~0.6 m), ~45 C
  COLD_OBJECT   : object held close (~0.6 m), ~8 C
  IDLE          : nothing within 3 m -> distance ~3.6 m, temp -1

Outputs:
  data/sensor_dataset.csv : label,distance,temp   (for enroll/train)
  data/sensor_stream.csv  : "distance, temp"       (his exact serial format, for --stdin --window)

  python make_sensor_dataset.py
"""
import os
import random

GESTURES = ["HAND_APPROACH", "HAND_RETREAT", "HOT_OBJECT", "COLD_OBJECT", "IDLE"]
W = 16          # samples per gesture instance (~1.6 s at his 10 Hz)
R = 25          # instances per gesture
SEED = 0
IDLE_TEMP = -1.0


def _sample(gesture, p, rng):
    """One (distance_m, temp_C) reading at progress p in [0,1] through the gesture."""
    if gesture == "HAND_APPROACH":
        d = 3.5 - 3.4 * p
        t = IDLE_TEMP if d > 3.0 else 30.0 + 4.0 * (1.0 - d / 3.0)
    elif gesture == "HAND_RETREAT":
        d = 0.1 + 3.4 * p
        t = IDLE_TEMP if d > 3.0 else 30.0 + 4.0 * (1.0 - d / 3.0)
    elif gesture == "HOT_OBJECT":
        d, t = 0.6, 45.0
    elif gesture == "COLD_OBJECT":
        d, t = 0.6, 8.0
    else:  # IDLE
        d, t = 3.6, IDLE_TEMP
    d = max(0.02, d + rng.uniform(-0.04, 0.04))          # ultrasonic jitter
    if t != IDLE_TEMP:
        t += rng.uniform(-0.3, 0.3)                       # IR jitter
    return round(d, 2), round(t, 2)


def generate():
    rng = random.Random(SEED)
    rows = []                                             # (label, d, t)
    for g in GESTURES:
        for _ in range(R):
            for i in range(W):
                rows.append((g, *_sample(g, i / (W - 1), rng)))
    return rows


def _window_feats(rows):
    """Group consecutive W rows into flat [d0,t0,...,dW-1,tW-1] feature vectors + labels."""
    feats, labels = [], []
    for s in range(0, len(rows), W):
        block = rows[s:s + W]
        if len(block) < W:
            continue
        feats.append([v for (_, d, t) in block for v in (d, t)])
        labels.append(block[0][0])
    return feats, labels


def main():
    os.makedirs("data", exist_ok=True)
    rows = generate()

    with open("data/sensor_dataset.csv", "w", encoding="utf-8") as f:
        f.write("label,distance,temp\n")
        for g, d, t in rows:
            f.write(f"{g},{d},{t}\n")
    # his exact serial format ("d, t"), gestures concatenated — a realistic raw stream
    with open("data/sensor_stream.csv", "w", encoding="utf-8") as f:
        for _, d, t in rows:
            f.write(f"{d}, {t}\n")

    # ---- separability self-check: are the gestures distinguishable as windows? -----
    feats, labels = _window_feats(rows)
    by = {}
    for v, lab in zip(feats, labels):
        by.setdefault(lab, []).append(v)
    # centroid from even instances, test odd (hold-out)
    cents = {lab: [sum(c) / len(vs[0::2]) for c in zip(*vs[0::2])] for lab, vs in by.items()}

    def nearest(v):
        return min(cents, key=lambda lab: sum((a - b) ** 2 for a, b in zip(v, cents[lab])))

    test = [(v, lab) for lab, vs in by.items() for v in vs[1::2]]
    acc = sum(nearest(v) == lab for v, lab in test) / len(test)

    print("=" * 58)
    print("SENSOR DATASET (synthetic, modelling builder's sensor.ino)")
    print("=" * 58)
    print(f"gestures : {GESTURES}")
    print(f"rows     : {len(rows)} samples ({R} x {W} per gesture), 2 channels (distance, temp)")
    print(f"windows  : {len(feats)} of {W} samples; hold-out nearest-centroid acc = {acc:.0%}")
    print("files    : data/sensor_dataset.csv (labelled), data/sensor_stream.csv (his format)")
    print("=" * 58)

    assert len(rows) == len(GESTURES) * R * W, "row count wrong"
    assert acc >= 0.9, f"gestures not separable as windows: {acc:.2f}"
    print("self-check OK: dataset written, gestures separable as windowed signals")


if __name__ == "__main__":
    main()
