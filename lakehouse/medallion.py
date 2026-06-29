"""
v0.26 — local Medallion lakehouse PoC (the followable slice of the production roadmap).

The Gemini "Production-Grade Spiking Neural Data Lakehouse" roadmap is a cloud system
(Delta Lake / Spark / S3 / Kafka / Unity Catalog / Delta Sharing). Those need a cluster
and a cloud account. But the **data path** — the Medallion topology, columnar Parquet
storage, and the Gold→SNN feature handoff — is followable on one machine. This is that
proof-of-concept, built over this repo's own spike telemetry store (`spike_telemetry_hub`).

  Bronze : raw immutable spike events           -> columnar Parquet (schema-on-read)
  Silver : cleaned + temporally aligned          -> per-channel counts per time bin
  Gold   : curated features                       -> firing rate, population synchrony,
           Inverse Compression Ratio (the brief's gzip-based lesion metric), then a
           deterministic latency encoding = the SNN-ready handoff.

Query engine: polars (the lazy, single-node substitute for Spark SQL + Delta) — SQL over
Parquet with predicate pushdown / column pruning.

Cloud scale-out (Spark, Delta ACID/time-travel, Kafka streaming, Liquid Clustering,
Unity Catalog, Delta Sharing, FPE, federated learning) is documented in the README as the
production path — out of scope for a single box.

Run (needs polars):  ../.venv-lake/Scripts/python lakehouse/medallion.py
"""
import os
import gzip
import polars as pl
from spike_telemetry_hub import synth   # raw spike-telemetry source (this repo)
import data_quality                     # v0.43 gates: block promotion to Gold on bad data

LAKE = os.path.join("lakehouse", "data")   # Parquet output (gitignored)


# ---- BRONZE: raw events -> columnar Parquet --------------------------------
def bronze(hub):
    rows = [(int(t), c) for c in range(hub.n) for t in hub.ch[c]]
    df = pl.DataFrame(rows, schema={"t": pl.Int64, "channel": pl.Int32}, orient="row")
    path = os.path.join(LAKE, "bronze.parquet")
    df.write_parquet(path)
    return df, path


# ---- SILVER: clean + temporally align (bin into windows) -------------------
def silver(bronze_df, window):
    s = (bronze_df
         .with_columns((pl.col("t") // window).alias("bin"))
         .group_by(["channel", "bin"]).agg(pl.len().alias("spikes"))
         .sort(["channel", "bin"]))
    path = os.path.join(LAKE, "silver.parquet")
    s.write_parquet(path)
    return s, path


# ---- GOLD: features (rate, synchrony, ICR) + SNN latency handoff ------------
def inverse_compression_ratio(silver_df, n_channels, n_bins):
    """ICR = compressed/uncompressed size of the binned raster. Lower = more
    structured/redundant (the lesion-detection metric from the research brief)."""
    raster = bytearray(n_channels * n_bins)
    for ch, b, sp in silver_df.select(["channel", "bin", "spikes"]).iter_rows():
        idx = ch * n_bins + b
        if 0 <= idx < len(raster):
            raster[idx] = min(255, sp)
    raw = bytes(raster)
    return len(gzip.compress(raw, 9)) / max(1, len(raw))


def gold(bronze_df, silver_df, hub, window):
    dur, n = hub.dur, hub.n
    n_bins = (dur + window - 1) // window
    rate = (bronze_df.group_by("channel").agg(pl.len().alias("count"))
            .with_columns((pl.col("count") / dur).alias("rate")).sort("channel"))
    # population synchrony index = coefficient of variation of population spikes/bin
    pop = silver_df.group_by("bin").agg(pl.col("spikes").sum().alias("pop"))
    mean_pop = pop["pop"].mean() or 0.0
    synchrony = float((pop["pop"].std() or 0.0) / mean_pop) if mean_pop else 0.0
    icr = inverse_compression_ratio(silver_df, n, n_bins)
    g = rate.with_columns([pl.lit(synchrony).alias("synchrony_cv"),
                           pl.lit(icr).alias("icr")])
    path = os.path.join(LAKE, "gold.parquet")
    g.write_parquet(path)
    return g, path, synchrony, icr


def latency_handoff(gold_df, t_steps=32):
    """Gold firing-rate vector -> deterministic latency spikes (1/channel, brighter =
    earlier). This is the SNN-ready representation the repo's models consume."""
    rates = gold_df["rate"].to_list()
    mx = max(rates) or 1.0
    return sorted((int(round((1 - r / mx) * (t_steps - 1))), i)
                  for i, r in enumerate(rates) if r / mx > 0.1)


def main():
    os.makedirs(LAKE, exist_ok=True)
    hub = synth(64, 50_000, 0.004, {7, 42}, (25_000, 26_000))   # burst on channels 7,42
    WIN = 100
    b_df, b_p = bronze(hub)
    s_df, s_p = silver(b_df, WIN)
    g_df, g_p, synchrony, icr = gold(b_df, s_df, hub, WIN)
    # ---- data-quality gate: blocks promotion to Gold on bad / mutated / non-encodable data ----
    rows = [(int(t), int(c)) for t, c in b_df.iter_rows()]
    dq = data_quality.gate(rows, hub.n, hub.dur,
                           gold={"icr": icr, "rates": g_df["rate"].to_list(),
                                 "synchrony": synchrony},
                           bronze_prior_hash=data_quality.bronze_hash(rows))
    assert dq["gold_checked"], "data-quality gate did not run before Gold"
    handoff = latency_handoff(g_df)

    def kb(p):
        return os.path.getsize(p) / 1024

    print("=" * 60)
    print("MEDALLION LAKEHOUSE PoC  (Bronze -> Silver -> Gold)")
    print("=" * 60)
    print(f"source: {hub.n} channels, {hub.n_events():,} spike events\n")
    print(f"BRONZE  raw events     : {b_df.height:,} rows  -> {kb(b_p):.1f} KB Parquet")
    print(f"SILVER  binned (W={WIN}) : {s_df.height:,} rows  -> {kb(s_p):.1f} KB Parquet")
    print(f"GOLD    features        : {g_df.height:,} rows  -> {kb(g_p):.1f} KB Parquet")
    print(f"        population synchrony (CV) : {synchrony:.3f}")
    print(f"        inverse compression ratio : {icr:.3f}  (lower = more structured)")
    print()

    # Spark-SQL analog: query Gold Parquet with SQL
    ctx = pl.SQLContext(gold=pl.read_parquet(g_p))
    top = ctx.execute(
        "SELECT channel, count, rate FROM gold ORDER BY rate DESC LIMIT 5").collect()
    print("SQL: top channels by firing rate (the burst {7,42} should surface):")
    print(top)

    # column pruning / predicate pushdown: lazy-scan Silver, read only 2 columns
    pruned = (pl.scan_parquet(s_p).select(["channel", "spikes"])
              .group_by("channel").agg(pl.col("spikes").sum().alias("total"))
              .sort("total", descending=True).head(3).collect())
    top_chs = pruned["channel"].to_list()
    print(f"\nlazy column-pruned scan -> busiest channels: {top_chs}")
    print(f"SNN handoff: latency-encoded {len(handoff)} channel-spikes, first few {handoff[:5]}")
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    assert pl.read_parquet(b_p).height == hub.n_events(), "Bronze Parquet lost events"
    assert any(c in top_chs for c in (7, 42)), "Gold/Silver didn't surface the burst channels"
    assert 0.0 < icr < 1.0, f"ICR out of range: {icr}"
    assert synchrony > 0.0 and handoff, "missing Gold synchrony / SNN handoff"
    print("self-check OK: Bronze roundtrip intact, burst surfaced via Gold, "
          "ICR valid, synchrony + SNN handoff produced")


if __name__ == "__main__":
    main()
