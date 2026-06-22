# Changelog

All notable changes to the Spiking Neural Data Lake. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); each version is a git tag.

## [v0.28] — Streaming ingest (Pub/Sub + Dataflow) + Cloud Composer DAG
Wires the GCP scale-out's streaming + orchestration layers (artifacts only; run with your auth).
### Added
- `gcp/dataflow_ingest.py` — Apache Beam streaming pipeline: Pub/Sub subscription -> parse
  JSON spike events -> 60 s fixed windows -> Parquet in GCS Bronze (the continuous version
  of "land data into Bronze").
- `gcp/publish_spikes.py` — test publisher (this repo's synthetic telemetry -> Pub/Sub).
- `gcp/submit_dataflow.sh` — launch the pipeline on the Dataflow runner.
- `gcp/composer_dag.py` — Airflow DAG `snn_medallion` for Cloud Composer: Dataproc Serverless
  Medallion ETL -> BigQuery/BigLake Gold table refresh, daily.
- Terraform: Pub/Sub topic `spike-telemetry` + subscription `spike-telemetry-sub` (+ outputs);
  Composer left as an on-demand command (heavyweight ~$300+/mo).
- `gcp/README.md`: steps 6 (streaming) and 7 (orchestration).
### Verified
- All Python (Beam/Airflow/pubsub) + shell scripts pass parse/syntax checks locally; cloud
  execution is the user's to run.
### Remaining (documented): Dataplex governance, Cloud DLP de-id, KMS CMEK, Analytics Hub.

## [v0.27] — GCP-native deployment scaffold
Takes the local Medallion PoC toward a cloud lakehouse on GCP (GCS + BigLake/Iceberg +
BigQuery + Dataproc Serverless + Vertex AI). Artifacts only — provisioning needs the
user's auth/billing; nothing here touches a cloud.
### Added
- `infra/` — Terraform: GCS lake bucket (versioned), BigQuery dataset, BigLake connection,
  Artifact Registry, job service account + IAM. `terraform apply -var project_id=…`.
- `gcp/dataproc_medallion.py` — PySpark port of the Medallion ETL (Bronze→Silver→Gold over
  GCS), for Dataproc Serverless (no cluster).
- `gcp/Dockerfile` + `gcp/cloudbuild.yaml` + `gcp/submit_vertex.sh` — Vertex AI custom GPU
  training (cu128 + bindsnet); runs `eth_mnist_bindsnet.py --gpu` at 6400/60k on an L4 for
  the ~95% target (the run impractical on CPU).
- `gcp/submit_dataproc.sh`, `gcp/README.md` (ordered deploy guide + BigLake table SQL + cost
  notes), root `.dockerignore`.
### Verified
- Python + all shell scripts pass parse/syntax checks locally. (Cloud execution is the
  user's to run.)
### Mapping (roadmap → GCP)
- S3→GCS, Spark→Dataproc Serverless, Delta/Iceberg→BigLake Iceberg, Spark SQL→BigQuery,
  Kafka→Pub/Sub+Dataflow, Unity Catalog→Dataplex, FPE→Cloud DLP+KMS, Delta Sharing→
  Analytics Hub, GPU training→Vertex AI. Streaming/orchestration/governance documented as
  scale-out, not scripted.

## [v0.26] — Medallion lakehouse PoC (the followable slice of the production roadmap)
Assessed an external "Production-Grade Spiking Neural Data Lakehouse" roadmap (cloud:
Delta Lake / Spark / S3 / Kafka / Unity Catalog / Delta Sharing / FPE / federated). ~60%
needs cloud infra + $; the ~40% data-PATH slice is followable on one box — built here.
### Added
- `lakehouse/medallion.py` — Medallion Bronze→Silver→Gold over this repo's spike store:
  - **Bronze** raw events → columnar **Parquet**; **Silver** binned/temporally-aligned;
    **Gold** features (per-channel firing rate, population synchrony CV, **inverse
    compression ratio** via gzip = the brief's lesion metric), then a deterministic
    **latency encoding** = the SNN handoff.
  - Queried with **polars** (single-node Spark-SQL / Delta substitute): SQL over Parquet
    + lazy column-pruned scans.
- `polars` added to requirements (lakehouse PoC only; use its own venv).
### Verified (64-channel synthetic telemetry, burst on {7,42})
- Bronze 13,133 events → 29.9 KB Parquet; Silver 10.6k rows → 15.4 KB; Gold 64 rows.
- Synchrony CV 0.269, ICR 0.197; SQL + column-pruned scan both surface the burst
  channels {7,42}; latency handoff fires them earliest. Self-checks pass.
### Documented (not built — needs cloud)
- Production scale-out: Spark, Delta/Iceberg ACID + time-travel, Kafka, Liquid Clustering,
  Unity Catalog, Delta Sharing, format-preserving encryption, federated learning.

## [v0.25] — Depth: relation algebra (Paradigm C) + GPU scaling harness
Two depth tracks: richer relation types now (CPU), GPU-scale accuracy harness for later
(GPU busy with the 6400/60k run).
### Added
- `spike_kg_relations.py` — Paradigm C depth. One cyclic KG exercising the full relation
  algebra in spike-phase coding: **symmetric** (φ≈0/π, self-inverse), **inverse**
  (φ≈−φ), **composition** (φ_C≈φ_A+φ_B). Trains RotatE, then checks BOTH per-relation
  link prediction AND that the learned phases satisfy the algebra.
  - Result: **all 6 relation types 100% Hits@1**; learned-phase errors inverse 0.15 rad,
    composition 0.22 rad, symmetric 0.61 rad (looser — two basins 0/π — but ≪ random 1.57).
  - Added to the CI stdlib self-check set.
- `gpu_scaling_sweep.sh` — accuracy-vs-neurons scaling-law harness (400→1600→6400) for the
  conductance D&C model; `python -u` unbuffered so progress is live (the 6400 run was
  invisible due to Python output buffering). Runs on a CUDA box after the GPU frees up.
### Note
- GeNN-based GPU depth still needs a C++ compiler + CUDA toolkit (only the cu128 torch
  runtime is installed); the BindsNET `--gpu` path is the working GPU vehicle.

## [v0.24] — Repo presentation: catalyst-neuromorphic style + CI
Reformatted the repo to match the conventions of the catalyst-neuromorphic org
(badges → results table → architecture-at-a-glance → directory tree → competitive context).
### Added
- `.github/workflows/ci.yml` — runs every pure-stdlib script's assert-based self-check on
  each push (12 scripts), so the **CI badge is real and green**. Verified locally first.
### Changed
- README restyled: badge row (CI / Python / MIT / release / zero-deps / GPU), a
  reproducible **Results** table, an **Architecture at a Glance** table (3 paradigms +
  trainable models), a **Directory structure** tree, a **Competitive context** section
  (vs dense ANN, rate coding, the literature, Project Nord). Substance (versions, scope,
  limitations, provenance) preserved.
- Repo description + topics set to match the org's vocabulary.

## [v0.23] — GPU-verified BindsNET on RTX 5070 (the --gpu path actually runs now)
The v0.10 `--gpu` switch was never exercised on a real GPU; running it on an RTX 5070
(Blackwell sm_120, torch 2.11.0+cu128) surfaced two device bugs, now fixed.
### Fixed
- `all_activity` returns predictions on CPU while label tensors are on CUDA → align
  predictions to the device before comparing (3 sites).
- BindsNET's `proportion_weighting` has an internal CPU/GPU device bug; dropped that
  secondary readout and report all-activity accuracy (the canonical Diehl & Cook metric).
### Verified
- `torch.cuda.is_available()` True, device "NVIDIA GeForce RTX 5070", capability (12,0),
  GPU matmul OK. End-to-end smoke (100 neurons / 2k imgs) runs clean on GPU (57.5% —
  undertrained at 2k, as expected). Full **6400 neurons / 60k train / 10k test** run
  launched on the GPU for the ~95% target (68% GPU util, 4.3/12 GB).
### Note
- GeNN paths (`*_genn.py`) still additionally need a C++ compiler + CUDA toolkit; the
  BindsNET `--gpu` path needs only the cu128 torch driver stack, which the 5070 has.

## [v0.22] — Cyclic relations via RotatE (spike phase coding)
v0.21's TransE embeds translational/lattice relations but gets **0%** on cyclic ones
(`tail = (head + shift) mod N` needs a rotation). RotatE fixes it — and is more
spike-native: phase-of-firing coding.
### Added
- `spike_knowledge_graph_rotate.py` (stdlib): entities = spike PHASES θ∈[0,2π),
  relations = phase SHIFTS φ_r; a triple holds when θ_t ≈ θ_h + φ_r (mod 2π). Distance
  is the RotatE chord `2|sin(δ/2)|` with its smooth gradient; trained with margin
  ranking, LR decay, and multiple negatives per positive. Includes a TransE baseline
  on the SAME cyclic KG for direct contrast.
### Verified (modular-ring KG, 40 entities, 5 relations)
- **RotatE: Hits@1 100%, MRR 1.000** — vs **TransE: Hits@1 0%** on identical cyclic data.
- Tuning notes: chord distance (not L1), D=4 (low-dim ring, less fragmentation), and
  K=5 negatives per positive were what took it from 20% → 100%.
- Paradigm C now covers translational (v0.21) AND cyclic (v0.22) relations.

## [v0.21] — Paradigm C complete: relational spiking embeddings (SpikE)
v0.11–18 built Paradigm C's encoding side (TTFS, deterministic latency, Van Rossum). The
missing SpikE core is RELATIONAL: store a knowledge graph in spike timing and reason over it.
### Added
- `spike_knowledge_graph.py` (stdlib): a SpikE-style embedding —
  - entities = spike-latency vectors `s_e ∈ R^D`; relations = spike-time offsets `δ_r`.
  - a triple (h,r,t) holds when `s_t ≈ s_h + δ_r` (TransE in spike-time space).
  - learns `s_e`, `δ_r` from triples only (margin-ranking SGD, ball-projected entities).
  - **link prediction** (h,r,?) by ranking tails on spike-time translation, and
    **anomaly scoring** (triple score = `||s_h+δ_r−s_t||`).
### Verified (3-axis lattice KG, 64 entities, 3 relations)
- Link prediction: **Hits@1 50.0%, Hits@3 63.6%, MRR 0.581** (random ~1.6%).
- Anomaly: random triples score **2.2×** higher than true triples (clean separation).
- Self-check enforces Hits@1 ≥ 50% and anomaly separation > 2×.
### Milestone
- **All three assessment paradigms now complete: A (v0.12) · B (v0.20) · C (v0.21).**
  Notes: TransE models translational/lattice relations, not cyclic ones (that needs
  RotatE) — the demo KG is a lattice so the spike-time translation holds exactly.

## [v0.20] — Paradigm B complete: in-storage spike-query engine
v0.13–14 had one query type (coincidence). A real in-storage search engine needs the
SNN-native query a von-Neumann scan struggles with: temporal ORDER.
### Added
- `paradigm_b_engine.py` — `SpikeQueryEngine` over the v0.12 `.spk` store with two
  compiled query types, both partial-read (only queried channels) + only-matches-to-host:
  - `coincidence(channels, W, k)` — ≥k distinct channels within W (the v0.13/14 detector).
  - `sequence(ordered, W)` — channels fire in the GIVEN ORDER within W: a delay-line /
    polychronous-group detector. Order matters, which is the whole point of Paradigm B.
### Verified (256-channel store, 40 injected 5→17→42 motifs + the {7,99} burst)
- coincidence {7,99}: 634 matches (505 in burst), read 2.0% of file.
- sequence 5→17→42: **41 matches (40/40 in the motif window)**; reverse 42→17→5: **3** —
  the engine discriminates temporal order (a coincidence filter cannot). Read 1.3% of file.
- Self-checks: coincidence finds burst, sequence finds the ordered motif AND rejects
  reverse order, both partial-read.
### Roadmap
- **Paradigm A complete (v0.12) · Paradigm B complete (v0.20) · Paradigm C started (v0.11–18).**

## [v0.19] — GeNN custom plasticity (inject the working rule onto the GPU)
Acts on the RTX 5070 architecture guidance: rather than GeNN's standard pair-STDP
(which degraded, v0.18), inject the v0.17 rule that WORKS as a custom GPU weight-update.
### Added
- `snn_mnist_stdp_genn.py` — GeNN 5 STDP-MNIST trainer:
  - `burst_xtar_rule()` = `create_weight_update_model` with the v0.17 rule in C++:
    post-spike LTP `g += lr*(preTrace − xtar)`, clamped — the literal GPU form of
    `snn_mnist_stdp_fast.py`'s `W[w] += FAST_LR*(x_pre − FAST_XTAR)`.
  - SpikeSourceArray for the deterministic burst-latency input (exact timestamps to
    VRAM once → identical inputs stay identical → Van Rossum 0, query identity holds).
  - `model.build()` nvcc-compiles the network, collapsing the O(T) loop on-GPU — the
    route past the ~82% CPU ceiling toward the ~95% regime (with 6400 neurons + 60k).
  - Import-guarded: prints setup guidance + exits cleanly without GeNN.
### Status
- Cannot run on the dev box (no pygenn / CUDA / C++ compiler). The rule + encoding are
  CPU-verified in `snn_mnist_stdp_fast.py` (76.0%); this is the faithful GPU port.
  GeNN weight-update field names vary by version — the version-stable part is the C++
  rule body, which matches the verified math.

## [v0.18] — Pair-based STDP kernel (honest negative) + query-identity determinism
Tried to take the last ~6 points with a proper pair-based STDP kernel, and proved why
the query/router path must use deterministic encoding.
### Added
- Pair-based STDP in `snn_mnist_stdp_fast.py`: a post-synaptic trace `x_post` and
  pre-triggered **LTD** (`FAST_AMINUS`, `TAU_POST`) — an input arriving after the
  neuron fired (anti-causal) depresses that synapse. The full kernel = causal LTP
  (post-spike, x_pre) + anti-causal LTD (pre-spike, x_post).
- `encode_poisson` + a query-identity demo/self-check in `spike_preprocessing.py`.
### Result A — pair kernel does NOT close the gap (negative, kept opt-in)
- LTD degrades accuracy at every tested strength, in both encodings:
  burst=4 72.0% → 65–70% (A−=0.005–0.02); pure latency 60.4% → 57–58%.
- Cause: anti-causal LTD erodes prototypes in this hard-WTA unsupervised setup (and
  fights the burst's later spikes). Rate coding's edge is its repeated stochastic
  sampling, not a missing LTD term — single-pass deterministic latency can't replicate
  that. So `FAST_AMINUS` defaults to 0; the v0.17 rule (76.0%, gap −6.2) stays best.
### Result B — determinism is REQUIRED for query identity (positive)
- The same MNIST image encoded twice: deterministic latency → Van Rossum distance
  **0.000** (recognised as the SAME query); Poisson → **13.37** (looks like different
  data). Confirms the router/match path must use the deterministic encoder; Poisson's
  per-call randomness destroys query identity. Enforced by a self-check.
### Honest status
- The residual ~6-point gap stands. It is not a missing-rule problem; it is the
  information difference between one deterministic pass and many stochastic samples.
  Likely closers are more neurons / data or a fundamentally different (non-WTA)
  readout — not this kernel.

## [v0.17] — Close the latency↔rate gap (burst encoding + LTD)
Halved the v0.16 accuracy gap while keeping the latency path's efficiency edge.
### Diagnosis
- v0.16's single-spike latency loses pixel-magnitude info and gives a weak STDP trace,
  and the fast rule had no depression to sharpen prototypes.
### Changed (`snn_mnist_stdp_fast.py`, both as env knobs, new defaults)
- `FAST_BURST` (default 4): graded deterministic burst — a pixel emits up to N spikes
  (count ∝ intensity) from its latency onward. Restores magnitude + a stronger trace.
- `FAST_XTAR` (default 0.05): Diehl & Cook-style LTD — on a post-spike, synapses below
  the target trace are depressed (unused inputs → −x_tar), sharpening prototypes.
- Defaults retuned (`FAST_THRESH=2.0`, `FAST_LR=0.05`); set `FAST_BURST=1 FAST_XTAR=0`
  to recover the v0.16 pure-latency behaviour.
### Result (M=300, 6k train, 2k test)
| | rate | v0.16 latency | **v0.17 latency+** |
|---|---|---|---|
| accuracy | 82.3% | 68.8% | **76.0%** |
| gap vs rate | — | −13.5 | **−6.2** |
| train time | 70.1 s | — | 33.2 s (2.1×) |
| train SynOps | 1.95 B | 247 M | 750 M (2.6× fewer) |
- Gap **more than halved** (−13.5 → −6.2 pts) at 2.1× faster / 2.6× fewer SynOps,
  still deterministic. Sweeps showed burst=4 is the optimum (burst 5–6 and x_tar>0.05
  regress).
### Residual gap (honest)
- The last ~6 pts appear inherent to single-pass latency vs rate's repeated sampling.
  Fully closing it would need a pre/post **pair-based STDP kernel** (timing-windowed
  LTP/LTD) rather than this trace-Hebbian rule — flagged as the next experiment. The
  verified 82.3% rate path stays the default.

## [v0.16] — Precomputed latency STDP vs rate STDP (head-to-head)
Wires the v0.15 preprocessing (deterministic latency + precompute) into the real STDP
model and compares it against the verified rate-coded path, in one process.
### Added
- `snn_mnist_stdp_fast.py` — `FastStdpNetwork`: same LIF + adaptive-threshold + hard-WTA
  STDP, but driven by **precomputed deterministic latency spikes** (each pixel fires
  once; no on-the-fly `torch.rand`). Runs the rate baseline (`StdpNetwork`) and the fast
  net on the same MNIST subset and prints the comparison. Tunable: `FAST_THRESH`,
  `FAST_LR` (latency traces are sparse → low threshold, high LR).
### Comparison (M=300, 6k train, 2k test, same tuned homeostasis)
| metric | rate (Poisson, on-the-fly) | fast (latency, precomputed) |
|---|---|---|
| accuracy | **82.3%** | 68.8% (−13.5 pts) |
| train time | 47.3 s | **22.8 s** (2.1×) |
| train SynOps | 1.95 B | **247 M** (7.9× fewer) |
| encoding | stochastic | deterministic + cacheable |
### Verdict (honest)
- Precompute + latency coding is **2.1× faster, 7.9× fewer SynOps, and deterministic**
  — but costs **~13.5 accuracy points** with this simple Hebbian STDP. A single-spike
  latency code is a weaker presynaptic trace than rate coding's repeated spikes, so the
  STDP learning signal is poorer. The rate path stays the default; the fast path is the
  efficiency/reproducibility option. (The verified 82.3% rate result is untouched.)

## [v0.15] — Spike preprocessing pipeline (deterministic encode, precompute, Van Rossum)
Implements three recommended preprocessing steps for CPU-viable STDP.
### Added
- `spike_preprocessing.py` (stdlib):
  1. `encode_latency` — **deterministic** latency encoding (1 spike/feature,
     brighter→earlier, no RNG; same input → same spikes).
  2. `precompute` + `save_cache`/`load_cache` (`.spc` format) — encode the dataset
     ONCE and reuse across epochs, removing spike generation from the training loop.
  3. `van_rossum_filter` / `van_rossum_distance` — exponential-decay filtering turns
     discrete spikes into continuous waveforms so a query can be matched to stored
     data with plain numeric ops.
### Results (synthetic graded dataset)
- Deterministic encoding confirmed; `.spc` cache roundtrips intact.
- Precompute+reuse over 8 epochs = **5.7× less encode work** than re-encoding on the
  fly (18 ms → 3 ms) — the overhead the on-the-fly `torch.rand` STDP loops pay.
- Van Rossum query→nearest-stored-prototype matching: **100%** accuracy; distance to
  the correct class < distance to a wrong one (self-check enforced).
### Notes
- `temporal_coding_storage.py` (v0.11) was already deterministic; the encoder is now
  factored into a reusable, cacheable form here.
- The standalone module demonstrates the precompute technique; wiring it into the
  torch STDP loops (`snn_mnist_stdp.py`) is the recommended integration — deferred so
  the verified 82.3% rate-coded result stays reproducible (latency coding would need
  its own retune).

## [v0.14] — Paradigm B distinct-channel counting (GeNN sub-detectors)
The v0.13 GeNN detector was a single summing LIF — it counts TOTAL input spikes, so k
spikes from one channel false-trigger it. Fixed to count DISTINCT channels.
### Changed
- `paradigm_b_genn.py`: two-stage network — input → per-channel **one-shot
  sub-detectors** (LIF, refractory = W, so each channel emits ≤1 pulse per window) →
  **counter** LIF (fires when ≥k sub-pulses arrive within W). Counts distinct channels.
### Added
- `paradigm_b_matcher.subdetector_match` — pure-Python model of that exact two-stage
  network (the oracle to validate the GPU match counts against; GeNN can't run on the
  dev box). Plus `total_count_match` to demonstrate the bug it fixes.
### Verified (CPU, 256-channel store)
- Sub-detector net: 634 matches (505 in burst) — vs the v0.13 deque reference's 639;
  the small delta is the sub-detector's one-shot-per-window refractory vs raw-distinct
  re-triggering (both count distinct channels).
- **Distinctness proof:** single channel firing k×4 in a window → total-counter makes
  2 false matches; the sub-detector makes **0** (correct). Self-check enforces this.

## [v0.13] — Paradigm B: in-storage spike-stream matcher (+ GeNN GPU port)
Started Paradigm B (in-storage pattern matching, cf. NPUsearch): compile a query into
an SNN, stream stored spikes through it, transfer only matches to the host.
### Attempted: GeNN
- Tried to build it on GeNN (genn-team) per request. GeNN code-generates C++/CUDA at
  runtime, so it needs CUDA + a C++ compiler + pygenn — none present on this box
  (`pygenn` has no installable wheel here, no `cl`/`gcc`/`nvcc` on PATH). So GeNN can't
  run here; it's shipped as a ready GPU port for a CUDA box.
### Added
- `paradigm_b_matcher.py` (stdlib, **verified CPU reference**): compiles a query
  (template channels + coincidence window W + min coincident k) into a LIF coincidence
  detector, reads ONLY the template channels from the v0.12 `.spk` store (partial seek),
  and streams them through the detector, emitting a match per coincidence.
- `paradigm_b_genn.py`: the same detector as a GeNN 5 / PyGeNN network
  (SpikeSourceArray → LIF detector, output spikes = matches). Import-guarded so it
  prints setup guidance and exits cleanly when GeNN isn't installed. Needs CUDA 12.8+
  (RTX 5070 = Blackwell/sm_120) + a C++ compiler + pygenn.
- `pygenn` added to requirements (optional, GPU box only).
### Results (256-channel store, query = channels {7,99} coincide within 50 steps)
- 639 matches emitted, **505 inside the injected burst window**.
- Read **2.0% of the file** (only the 2 template channels) and emitted match stamps =
  **162× less data to host** than streaming all 103k raw events.
- Self-checks: partial-read matches == brute-force, burst detected, transfer reduced.
### Roadmap status
- Paradigm A complete (v0.12) · B started (v0.13, GeNN GPU port pending a CUDA box) ·
  C started (v0.11 TTFS).

## [v0.12] — Spike Telemetry Hub (Paradigm A complete)
Completes the assessment's Paradigm A: a hub for multi-channel spike-train telemetry
(BCI / neural-sim style) stored and queried as sparse events, never dense rasters.
### Added
- `spike_telemetry_hub.py` (stdlib):
  - `SpikeTelemetryHub` — per-channel sorted spike-time store (AER).
  - `.spk` file format with a per-channel index (offset+count) and a `disk_query`
    that SEEKS to only the requested channels — windowed queries without loading the
    whole dataset.
  - windowed range query (binary search, O(log n + hits)), bin, firing-rate, ISI,
    and a burst/anomaly detector.
  - input validation + magic-byte file check on the persistence boundary.
### Results (256 channels, 100k steps, ~104k spikes)
- Sparse `.spk` = 418 KB vs **3.2 MB** (1-bit raster, 7.7×) vs **25.6 MB** (1-byte
  raster, 61×).
- Windowed query on 2 channels read **2.0% of the file** (8.5 KB) and matched a
  brute-force scan exactly.
- Injected burst detected by the rate-threshold detector.
- Self-checks: save/load roundtrip intact, disk query == brute force, partial read,
  sparse < dense, burst found.
### Roadmap status
- Paradigm A: **complete**. Paradigm C: started (v0.11 TTFS). Paradigm B (in-storage
  NPU search): still hardware-dependent, out of scope.

## [v0.11] — Temporal (TTFS) coding, from the architectural assessment
Acted on an external architectural assessment (Gemini). The assessment was produced
WITHOUT repo access (speculative, name-based), so its applicable, on-theme ideas were
adopted and its production-infra suggestions (Parquet tier, OpenTelemetry, in-storage
NPU) were left as out-of-scope roadmap.
### Added
- `temporal_coding_storage.py` — time-to-first-spike (latency) coding, realising the
  assessment's Paradigm C ("more salient data spikes earlier", cf. the SpikE idea).
  Trains a linear readout once, then compares rate vs TTFS inference on the same
  weights: **TTFS matches accuracy (100%) at 83.5x fewer SynOps** (5,300 vs 442,775),
  deciding at avg step 6.5/32 via early exit (1 spike per input + stop at first class
  over threshold).
- README "Scope, related work & roadmap" section: honest what-this-is/isn't, the three
  assessment paradigms as a roadmap (A telemetry / B in-storage search / C spiking
  embeddings — C started here), the SNN ecosystem (snnTorch/SpikingJelly/BindsNET/
  SpikeData/SpikE/NPUsearch), and the LIF model math.
### Note
- Temporal coding is a genuinely different efficiency axis from the rest of the lake
  (which is rate-coded): the win is from spike *timing* + early exit, not sparsity.

## [v0.10] — `--gpu` switch for the BindsNET runner
### Added
- `--gpu` / `--device <dev>` flag (and `NORD_GPU=1` env) on `eth_mnist_bindsnet.py`:
  moves the network + all tensors to CUDA so the 6400-neuron → ~95% run is one switch
  on a GPU box. Defaults to CPU.
- RTX 5070 (Blackwell, sm_120) guidance: if `--gpu` is requested but CUDA torch isn't
  present, the runner prints the exact cu128 install command and falls back to CPU
  (the default CPU/older-CUDA wheels have no sm_120 kernels).
### Fixed
- Dropped the `device=` kwarg from `Monitor(...)` (not accepted by the installed
  BindsNET); `network.to(device)` moves the monitored layer's spikes instead.
### Verified
- Ran end-to-end with `--gpu` on this CPU box: prints the RTX 5070 hint, falls back to
  CPU, trains + tests clean (device threading correct).

## [v0.9] — Path to ~95%: BindsNET conductance Diehl & Cook
The v0.8 study showed our from-scratch current-based inhibition can't reach the
literature ~95%. Deep research (Diehl & Cook 2015 + BindsNET) pinned the cause:
**conductance-based synapses + a real exc/inh population + scale**. v0.9 wires that
in via BindsNET instead of re-deriving it.
### Added
- `eth_mnist_bindsnet.py` — runner around BindsNET's `DiehlAndCook2015` (conductance
  LIF, exc/inh populations, adaptive thresholds) with the paper/BindsNET constants
  (`exc=22.5 inh=120 norm=78.4 theta_plus=0.05 time=250 intensity=128`). Sizes are
  env-configurable (`NORD_M/TRAIN/TEST/EPOCHS/TIME/UPDATE`); defaults to 6400 neurons.
- A built-in `torch._six` compat shim so BindsNET (<=0.3) runs on torch >=2 with no
  manual patching.
- `bindsnet` added to requirements (optional, v0.9 only).
### Verified
- 100 neurons / 10k images / 2k test → **75.95% all-activity, 76.45% proportion**.
  The training window accuracy climbed 10% → 82% as STDP specialised — on track to
  the paper's 82.9% at 100 neurons / full 60k. Wiring is correct and learns.
### Compute reality (honest)
- The headline 95% needs **6400 neurons + all 60k images**, reported by Diehl & Cook
  (100→82.9%, 400→87.0%, 1600→91.9%, 6400→95.0%). On this CPU build that is ~hundreds
  of hours — a GPU/overnight job. The runner defaults to 6400 and prints the matching
  paper figure; tractable checkpoints: `NORD_M=100/10k` (~76%, minutes),
  `NORD_M=400/20k` (~87%, hours).
### Bottom line
- The path to ~95% is now **wired and validated end-to-end**. The remaining gap is
  compute (neurons × images × GPU), not the algorithm.

## [v0.8] — STDP inhibition study + homeostasis tuning
Follow-up on the v0.7 inhibition limitation: built and benchmarked several
explicit-inhibition designs, then found a real (if modest) accuracy gain.
### Added
- `snn_mnist_dc.py` — a from-scratch separate-population Diehl & Cook network
  (all-but-self lateral inhibition, weight-dependent STDP, re-presentation).
- New env knobs on `snn_mnist_stdp.py`: `NORD_KWTA` (k-winners co-fire),
  `NORD_THRESH`, `NORD_TPLUS`, `NORD_TDECAY` — homeostasis is now tunable.
- Label assignment now uses a 6k-image subset (faster on large training sets).
### Benchmark (M=100/1.5k smoke unless noted; chance = 10%)
| inhibition design | result | verdict |
|---|---|---|
| hard single-winner WTA + adaptive theta (baseline) | 70.6% | best |
| graded global inhibitory pool (`NORD_INHIB` 0.5–3) | 27–31% | collapses coverage |
| separate D&C population (`snn_mnist_dc.py`, swept) | 9–22% | collapses (needs conductance synapses) |
| k-WTA multi-winner (`NORD_KWTA` 3 / 7) | 66% / 58% | less selective |
### Result — what actually helped
- Hard single-winner WTA + adaptive thresholds is the effective strong-inhibition
  limit; explicit graded/population inhibition needs conductance dynamics to match.
- Retuned homeostasis **`NORD_TDECAY=0.99999 NORD_TPLUS=0.8`** lifts the STDP model
  **81.5% → 82.3%** at M=300/6k (the new best).
- Naive scale-up regressed (M=400/20k = 78.3% with frozen theta); theta-equilibrium
  recovered it to 80.9%, but M=300/6k stays the sweet spot at this tuning.
- Reaching the literature's ~95% needs conductance-based exc/inh LIF populations and
  all 60k images — out of scope for a stdlib-spirit prototype.

## [v0.7] — Fix the limitations
Addresses the three caveats from the README's limitations section.
### Changed
- **Associative memory storage is now factored (O(P·k), not O(N²)).**
  `spiking_storage_prototype.py` keeps the P sparse patterns and reconstructs the
  Hopfield field on the fly (exact, not approximate — same arg-top-k ranking).
  874× smaller memory (600 B vs 512 KB at N=256/P=15); recall identical (still 80
  patterns @ 99.6%, 60% noise tolerance); compute also drops (~109×).
- `snn_mnist_stdp.py`: added an optional explicit **lateral-inhibition population**
  (`NORD_INHIB`) — each exc spike charges a global inhibitory pool that suppresses
  all other exc neurons, decaying over time (Diehl & Cook inhibition, lumped).
### Added
- `snn_classifier.py sweep` — capacity/difficulty sweep: pixel-noise curve and a
  class-count curve, exposing where the classifier breaks.
### Results
- Storage fix: associative memory now wins on storage too — 874× vs the dense W.
- Capacity sweep: 100% at ≤20% pixel noise → 86% at 30% → 40% at 40% → ~chance at
  50%. Class-count stays ≥92% up to 8 (these shapes stay separable at 15% noise).
- Inhibition (honest negative result): at every tested strength (0.5–3.0) the
  lumped inhibition *underperforms* hard-WTA (70.6% → 27–31% on the smoke config)
  by destabilising class coverage. Default remains hard-WTA. Reaching ~95% needs
  the full machinery (separate exc/inh LIF populations, adaptive membrane
  thresholds, all 60k images) — out of scope for this prototype.

## [v0.6] — MoE + STDP hybrid
### Added
- `snn_moe_stdp_mnist.py` — fuses the two real-primitive lines: N unsupervised-STDP
  expert populations on real MNIST, routed by Project-Nord-style firing-rate gating
  (top-K of N experts, no learned router network), with a load-balance penalty.
- `make_results_plot.py` + `assets/results.svg` — reproducible results plot, embedded
  in the README.
### Results
- 74.4% test accuracy (6 experts × 60 neurons, top-2, 4000 images, chance 10%).
- Routing runs only 2 of 6 experts per image → **3.0× less expert compute** than a
  dense MoE; **70.3×** vs a dense ANN of the same neuron count.
- Router has **0 learned parameters** (routing is the spike drive) vs 4,704 for a
  learned N×784 router.
### Notes
- The load-balance penalty drove expert usage perfectly even, so routing balances
  rather than content-specialises here — accuracy matches a single STDP net; the win
  is compute + router storage, not accuracy. Lowering `LOAD_BALANCE` trades balance
  for content routing (and collapse risk).

## [v0.5] — Scale with real data
### Added
- Configurable scaling for the MNIST STDP model via environment variables —
  `NORD_M` (neurons), `NORD_TRAIN`, `NORD_TEST` — no code edit needed to scale.
### Changed
- `snn_mnist_stdp.py` now reads size knobs from the environment (defaults
  unchanged, so v0.3 behaviour is preserved).
### Results
- Scaled config `NORD_M=300 NORD_TRAIN=6000 NORD_TEST=2000`:
  test accuracy **81.5%** (up from 74.6% at the v0.3 default), compute reduction
  held at **23.6×**. More neurons (300) + more real data (6000 imgs) improve
  specialisation — neurons now spread across all 10 classes
  `[40,19,32,35,26,26,33,26,33,30]`.

## [v0.4] — Spike-driven MoE routing
### Added
- `snn_moe_classifier.py` — ports Project Nord's `SpikeDrivenMoE`: firing-rate
  routing (no learned router network), top-k sparse experts, homeostatic load
  balance.
### Results
- 100% on 4 shapes; **4× compute reduction** (top-2 of 8 experts);
  **64× smaller router** (8 bias params vs 512 for a learned N×experts router);
  all 8 experts used (balanced).

## [v0.3] — Real data: unsupervised STDP on MNIST
### Added
- `snn_mnist_stdp.py` — real MNIST + snnTorch, Diehl & Cook-style unsupervised
  STDP. No labels and no backprop during training; neurons labelled afterward by
  majority vote.
### Fixed
- Initial collapse (one neuron winning every WTA → chance accuracy) fixed by
  adding **per-neuron adaptive thresholds** (theta homeostasis) — the mechanism
  that forces neurons to specialise.
### Results
- 74.6% test accuracy (100 neurons, 3000 images, chance 10%); 23.5× compute
  reduction from input-spike sparsity.

## [v0.2] — Supervised spiking classifier
### Added
- `snn_classifier.py` — rate-coded spiking classifier trained with a stable
  local delta rule. Weights are the long-term store; sparse spikes are the
  compute saving.
### Results
- 100% test accuracy on 4 synthetic 8×8 shapes; 3.6× fewer ops than the dense
  same-architecture baseline.

## [v0.1] — Associative memory prototype
### Added
- `spiking_storage_prototype.py` — sparse k-winners-take-all associative memory.
  Data written to weights by a covariance Hebbian rule; content-addressable
  recall via attractor dynamics from a noisy cue.
- `test_prototype.py` — capacity and noise stress tests.
- `snn_storage_core_snntorch.py` — reference snnTorch blueprint extracted from
  the source research brief.
### Results
- Holds 80 patterns (31% of N=256) at ~99.6% recall; tolerates 60% cue
  corruption before degrading; recalled state 25.6× smaller as an event list
  than a dense float32 vector.
