"""
v0.45 — adapter: the builder's timestamped serial CSV -> the signal loop.

MakiKuri00's `robot arm/serial_sensor_log.csv` logs `Timestamp,Sensor_1,Sensor_2`
(distance m, temp K with -1 idle) plus ESP32 boot-banner lines captured from serial start.
Our wire contract is numeric-only, so the prepended timestamp makes EVERY row unparseable
(0/812). This adapter strips the timestamp column, skips the header/banner, and feeds clean
(distance, temp) windows into signal_loop — so real hardware data flows through the pipeline.

  python sensor_csv_ingest.py                                  # ingest + report
  python sensor_csv_ingest.py --stream | python signal_loop.py --stdin --window 8
"""
import os
import sys

import signal_loop as S

CSV = os.path.join("robot arm", "serial_sensor_log.csv")
WINDOW = 8                       # 8 ticks x 2 channels -> 16 features (resized to N)


def parse_csv(path):
    """Strip the leading timestamp column; keep only rows whose remainder is numeric."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(",", 1)     # [timestamp, "d, t"]
            if len(parts) != 2:
                continue
            v = S.parse_line(parts[1])                  # header/banner -> None (skipped)
            if v:
                rows.append(v)
    return rows


def main():
    rows = parse_csv(CSV)

    if "--stream" in sys.argv[1:]:                      # emit numeric lines for --stdin piping
        for v in rows:
            print(", ".join(str(x) for x in v))
        return

    # ingest through the loop. Default library = robot commands (wrong domain for this
    # sensor) -> windows are correctly REJECTED as novel and recorded; continual-learning
    # clustering then finds the sensor's own recurring regimes (idle vs object-present).
    src = S.line_windows(iter(rows), window=WINDOW)
    lib = S.build_default_library()
    sink = []
    stats, lake = S.run(src, lib, emit=False, on_unknown=lambda seq, d, w: sink.append(w))
    promoted, n_clusters, radius = S.cluster_unknowns([{"vec": w} for w in sink], lib)

    print("=" * 60)
    print("REAL SENSOR CAPTURE -> SIGNAL LOOP  (MakiKuri00's serial_sensor_log.csv)")
    print("=" * 60)
    print(f"parsed       : {len(rows)} numeric rows (timestamp stripped, banner skipped)")
    print(f"windows      : {stats['events']} (window={WINDOW}, 2ch) -> lake {len(lake)} encoded")
    print(f"vs robot lib : {stats['matched']} matched / {stats['rejected']} novel "
          f"(novel expected — wrong domain)")
    print(f"discovered   : {len(promoted)} recurring sensor regimes from the novels "
          f"({n_clusters} clusters, sizes {[p['size'] for p in promoted]})")
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    assert len(rows) > 700, f"expected ~801 real rows, got {len(rows)}"
    assert stats["events"] > 0, "no windows ingested"
    assert len(lake) == stats["events"], "lake did not store every window"
    assert len(promoted) >= 1, "continual learning found no recurring regime in real data"
    print("self-check OK: real hardware capture parses, windows ingest into the lake, "
          "and recurring sensor regimes are discovered")


if __name__ == "__main__":
    main()
