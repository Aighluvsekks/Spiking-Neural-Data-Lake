# Spike storage formats

The repo uses several on-disk formats, each tuned to a different access pattern. This is the
one place that says which is which, so the `.spk / .spc / .spkl` sprawl isn't guesswork.

| Ext | Written by | Read by | Layout | Why it exists |
|-----|-----------|---------|--------|---------------|
| `.spk` | `spike_telemetry_hub.py` | `paradigm_b_{engine,matcher,genn}.py` | Sparse binary, magic header + per-channel index (offset+count) | **Partial reads** — query one channel without scanning the file (in-storage SNN query) |
| `.spkl` | `streaming_hub.py` | `streaming_hub.py` | Durable append-only event log, flushed per spike | **Streaming** — record *while* querying; events land as they fire, replayable |
| `.spc` | `signal_loop.py`, `spike_preprocessing.py` | same | gzip-compressed encoded windows | **Storage-cheap lake** — compact at-rest store of encoded signal windows / preprocessing cache |
| `.parquet` | `lakehouse/medallion.py` (polars) | medallion, polars SQL | Columnar (Bronze/Silver/Gold) | **Analytics** — predicate pushdown + column pruning; the lakehouse data path |
| Delta | `lakehouse/delta_demo.py` (delta-rs) | polars `read_delta` | Parquet + transaction log | **Streaming lakehouse** — transaction log + `OPTIMIZE` compaction kills the small-file problem; ACID snapshots + time-travel. Needs `deltalake` in `.venv-lake`. |
| `.jsonl` | `signal_loop.py` | `signal_loop.py` | JSON lines | **Unknowns** — non-matching windows queued for clustering / continual learning |

## Common invariants
- **Bronze events are immutable** — `(t:int, channel:int)`, append-only. `data_quality.bronze_hash`
  pins content; `data_quality.gate()` blocks Gold promotion on schema / duplicate / encodability /
  immutability / Gold-sanity failure (batch in `medallion.py`, per-event in `gcp/dataflow_ingest.py`).
- `.spk` and `.spkl` both store raw spike events; `.spk` is **index-optimized for reads**, `.spkl`
  is **append-optimized for writes**. They are not interchangeable — pick by access pattern.

## ponytail note
Convergence to a single format isn't worth it: each format earns its place by a distinct access
pattern (partial-read vs append vs compress vs columnar). The cost was *documentation*, not *count* —
that's what this file fixes.
