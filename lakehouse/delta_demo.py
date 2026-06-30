"""
v0.53 — Delta Lake PoC (Gemini brief's headline data-lake item).

The streaming Bronze path (gcp/dataflow_ingest.py) appends high-frequency AER events as
VANILLA Parquet -> thousands of tiny files -> the "small file problem": query engines burn
more cycles opening files + scanning metadata than aggregating. Delta Lake fixes this with a
transaction log + native OPTIMIZE compaction + ACID snapshots (consistent reads while the
closed loop keeps appending).

This proves it locally with delta-rs (the `deltalake` lib + polars `write_delta`) — NO Spark,
NO JVM. We:
  1. append many micro-batches (the streaming pattern) -> many small files
  2. OPTIMIZE/compact -> few right-sized files, data byte-identical
  3. time-travel to an earlier version (ACID snapshot / Delta's read-consistency)

Run (needs the lake venv + deltalake):
  PYTHONPATH=. .venv-lake/Scripts/python lakehouse/delta_demo.py

Scope note: this is the LAKEHOUSE-tier upgrade. It mainly benefits the UNVERIFIED cloud
streaming path (dataflow_ingest WriteToParquet -> would become a Delta sink). The verified
local stores stay the stdlib .spk/.spc/.spkl formats (see docs/STORAGE_FORMATS.md).
"""
import os
import shutil
import random

import polars as pl
from deltalake import DeltaTable

LAKE = os.path.join("lakehouse", "data")
TABLE = os.path.join(LAKE, "delta_bronze")


def micro_batch(n, seed):
    """One streaming micro-batch of AER events {t, channel} — the kind dataflow appends."""
    rng = random.Random(seed)
    return pl.DataFrame({"t": [rng.randint(0, 10_000) for _ in range(n)],
                         "channel": [rng.randint(0, 63) for _ in range(n)]},
                        schema={"t": pl.Int64, "channel": pl.Int32})


def main():
    os.makedirs(LAKE, exist_ok=True)
    if os.path.isdir(TABLE):
        shutil.rmtree(TABLE)                       # clean slate (reproducible)

    BATCHES, PER = 12, 200
    total = 0
    for i in range(BATCHES):                       # streaming: one Delta append per micro-batch
        df = micro_batch(PER, seed=i)
        df.write_delta(TABLE, mode="append")
        total += df.height
    first_batch_rows = micro_batch(PER, seed=0).height

    dt = DeltaTable(TABLE)
    files_before = len(dt.file_uris())
    rows_before = pl.read_delta(TABLE).height
    ver_before = dt.version()

    # time-travel: read the table AS OF version 0 (only the first append was committed then)
    v0_rows = pl.read_delta(TABLE, version=0).height

    # OPTIMIZE: compact the many small files into right-sized blocks (the small-file fix)
    dt.optimize.compact()
    dt = DeltaTable(TABLE)
    files_after = len(dt.file_uris())
    rows_after = pl.read_delta(TABLE).height

    print("=" * 62)
    print("DELTA LAKE PoC — small-file compaction + time travel (delta-rs)")
    print("=" * 62)
    print(f"appended         : {BATCHES} micro-batches x {PER} rows = {total} events")
    print(f"files before OPT  : {files_before}  (one+ per streaming append = small-file problem)")
    print(f"files after OPT   : {files_after}  (OPTIMIZE compacted them)")
    print(f"rows before/after : {rows_before} / {rows_after}  (data preserved)")
    print(f"time-travel v0    : {v0_rows} rows  (= first batch {first_batch_rows}; ACID snapshot)")
    print(f"version after OPT : {dt.version()}  (compaction is itself a logged transaction)")
    print("=" * 62)

    # ---- self-checks --------------------------------------------------------
    assert rows_before == total, f"Delta lost rows on append: {rows_before} != {total}"
    assert files_after < files_before, \
        f"OPTIMIZE did not compact ({files_before} -> {files_after})"
    assert rows_after == total, f"OPTIMIZE changed the data: {rows_after} != {total}"
    assert v0_rows == first_batch_rows, \
        f"time-travel to v0 wrong: {v0_rows} != {first_batch_rows}"
    print(f"self-check OK: {BATCHES} appends -> {files_before} files -> OPTIMIZE -> {files_after} "
          f"files, {total} rows intact, time-travel to v0 = {v0_rows} rows")


if __name__ == "__main__":
    main()
