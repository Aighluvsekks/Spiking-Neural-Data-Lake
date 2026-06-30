"""
v0.45 — adapter: the builder's timestamped serial CSV -> the signal loop.

MakiKuri00's `robot arm/serial_sensor_log.csv` logs `Timestamp,Sensor_1,Sensor_2`
(distance m, temp K with -1 idle) plus ESP32 boot-banner lines captured from serial start.
Our wire contract is numeric-only, so the prepended timestamp makes EVERY row unparseable
(0/812). This adapter strips the timestamp column, skips the header/banner, and feeds clean
(distance, temp) windows into signal_loop — so real hardware data flows through the pipeline.

CI-safe: if the real CSV is present it is used; otherwise (other branches / PRs that don't
carry the data file) a synthetic same-format sample is generated, so the self-check never
depends on a committed data file.

  python sensor_csv_ingest.py                                  # ingest + report
  python sensor_csv_ingest.py --stream | python signal_loop.py --stdin --window 8
"""
import os
import sys
import random

import signal_loop as S

CSV = os.path.join("robot arm", "serial_sensor_log.csv")
WINDOW = 8                       # 8 ticks x 2 channels -> 16 features (resized to N)


def _synthetic_lines(n=48, seed=0):
    """Stand-in for the builder's CSV when it is absent: SAME format — a header, ESP32
    boot-banner lines, then timestamped numeric rows alternating idle and object-approach."""
    rng = random.Random(seed)
    lines = ["Timestamp,Sensor_1,Sensor_2",
             "20:53:07,ets Jul 29 2019 12:21:46",
             "20:53:07,rst:0x1 (POWERON_RESET),boot:0x13",
             "20:53:07,mode:DIO, clock div:1"]
    for i in range(n):
        if (i // WINDOW) % 2 == 0:                       # idle regime
            d, t = 0.5 + rng.uniform(-0.05, 0.05), -1.0
        else:                                            # object-approach regime
            d, t = max(0.1, 3.0 - 0.3 * (i % WINDOW)), 295.0 + rng.uniform(-0.3, 0.3)
        lines.append(f"20:53:{10 + i % 50:02d},{d:.2f}, {t:.2f}")
    return lines


def raw_lines():
    if os.path.exists(CSV):
        with open(CSV, encoding="utf-8") as f:
            return f.read().splitlines(), f"real capture ({CSV})"
    return _synthetic_lines(), "synthetic sample (real CSV absent — CI / other branch)"


def parse_lines(lines):
    """Strip the leading timestamp column; keep only rows whose remainder is numeric."""
    rows = []
    for line in lines:
        parts = line.split(",", 1)                       # [timestamp, "d, t"]
        if len(parts) != 2:
            continue
        v = S.parse_line(parts[1])                       # header / banner -> None (skipped)
        if v:
            rows.append(v)
    return rows


def main():
    lines, source = raw_lines()
    rows = parse_lines(lines)

    if "--stream" in sys.argv[1:]:                        # numeric lines for --stdin piping
        for v in rows:
            print(", ".join(str(x) for x in v))
        return

    # ingest through the loop. Default library = robot commands (wrong domain) -> windows are
    # correctly rejected as novel and recorded; continual-learning clustering then finds the
    # sensor's own recurring regimes (idle vs object-present).
    src = S.line_windows(iter(rows), window=WINDOW)
    lib = S.build_default_library()
    sink = []
    stats, lake = S.run(src, lib, emit=False, on_unknown=lambda seq, d, w: sink.append(w))
    promoted, n_clusters, radius = S.cluster_unknowns([{"vec": w} for w in sink], lib)

    print("=" * 60)
    print("SENSOR CAPTURE -> SIGNAL LOOP")
    print("=" * 60)
    print(f"source       : {source}")
    print(f"parsed       : {len(rows)} numeric rows (timestamp stripped, banner skipped)")
    print(f"windows      : {stats['events']} (window={WINDOW}, 2ch) -> lake {len(lake)} encoded")
    print(f"vs robot lib : {stats['matched']} matched / {stats['rejected']} novel "
          f"(novel expected — wrong domain)")
    print(f"discovered   : {len(promoted)} recurring sensor regimes ({n_clusters} clusters, "
          f"sizes {[p['size'] for p in promoted]})")
    print("=" * 60)

    # ---- self-checks (work for both the real ~801-row capture and the synthetic sample) ----
    assert len(rows) >= 16, f"too few numeric rows parsed: {len(rows)}"
    assert stats["events"] > 0 and len(lake) == stats["events"], "windowing / lake mismatch"
    assert len(promoted) >= 1, "continual learning found no recurring regime"
    # adapter correctness, source-independent: banner skipped, timestamp stripped
    assert parse_lines(["20:53:07,ets Jul 29 2019 12:21:46"]) == [], "boot banner not skipped"
    assert parse_lines(["20:54:34,0.54, 294.88"]) == [[0.54, 294.88]], "timestamp not stripped"
    print("self-check OK: timestamp stripped + banner skipped, windows ingest into the lake, "
          "recurring regimes discovered (CI-safe with or without the data file)")


if __name__ == "__main__":
    main()
