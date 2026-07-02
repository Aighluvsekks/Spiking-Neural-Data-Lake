# Spiking Neural Data Lake

[![CI](https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake/actions/workflows/ci.yml/badge.svg)](https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/tag/Aighluvsekks/Spiking-Neural-Data-Lake?label=release)](https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake/releases)
[![Core: zero deps](https://img.shields.io/badge/core-zero%20dependencies-success.svg)](#quickstart)
[![GPU: CUDA 12.8](https://img.shields.io/badge/GPU-CUDA%2012.8%20%2F%20RTX%205070-76B900.svg)](#gpu)

Store, search, and reason over data **in spike timing** — using sparse, binary,
event-driven spiking neural networks instead of dense floating-point tensors.

The question: **can we store and recall data with less computational power and less
storage space using spikes?** Every prototype here both demonstrates a mechanism and
*measures* the two target metrics (compute = synaptic operations, storage = bytes) against
a dense baseline. All results are reproducible — clone, run, read the printed metrics.
The pure-Python core has **zero dependencies**; only the real-data (MNIST) models use
PyTorch + snnTorch / BindsNET.

---

## Clone & run

Python 3.11+. Install the package (editable) — the core pulls **no runtime deps**:

```bash
git clone https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake.git
cd Spiking-Neural-Data-Lake
pip install -e .                 # editable install; core = zero runtime deps

snn-demo                                       # whole loop in one file: sensor -> recognize -> command -> arm + reflex
snn-arm                                        # live loop on the builder's REAL captured gestures (95%)
snn-loop                                       # full closed loop: gesture -> command -> reward -> learn
python -m snn_data_lake.gesture_recognition    # order-aware recognition + confusion table
```

**Every addon:** `pip install -e ".[all]"` adds every optional dependency (arm+lake+gpu+nmnist)
in one env; or pick a path — `.[arm]` `.[lake]` `.[gpu]` `.[nmnist]`.

Run the **full zero-dep self-check suite** (exactly what CI runs — the 32 stdlib modules; the
MNIST/GPU models are excluded as they need PyTorch). The list lives in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) and [docs/RUNNING.md](docs/RUNNING.md):
```bash
python research/spiking_storage_prototype.py   # storage
python -m snn_data_lake.paradigm_b_engine           # in-storage order-aware query
python -m snn_data_lake.closed_loop                 # the full brain-stack loop (encode→match→reward→learn)
# ...full list (32) in docs/RUNNING.md §1
```

Optional extras (own venv each; see [docs/RUNNING.md](docs/RUNNING.md)): `polars` for the
Medallion lakehouse, `tonic` for real N-MNIST, `torch`+`snntorch`+`bindsnet` for the MNIST/GPU
models, `pybullet` for arm physics, `pyserial` for the live ESP32 (`python -m snn_data_lake.live_arm --serial COM8`).

---

## What is this?

A "lake" of runnable spiking-network prototypes covering the three things a data system
does — **store** (associative memory, telemetry hub), **search** (in-storage SNN query
engine), and **reason** (relational spike-time embeddings) — plus the trainable models
(unsupervised STDP, spike-driven MoE) that learn the representations. Each file prints a
metrics block and ends with an `assert`-based self-check; CI runs every stdlib self-check
on each push.

---

## Results

| Component | Task | Metric | Result |
|-----------|------|--------|--------|
| Associative memory | recall from 40%-corrupted cue | capacity / robustness | **80 patterns @ 99.6%**, 60% noise tolerated |
| Associative memory | memory footprint | factored vs dense | **874× smaller** (O(P·k), not N²) |
| Supervised classifier | 4-shape, spiking | accuracy / compute | **100%**, 3.6× fewer ops than dense |
| Unsupervised STDP MNIST (CPU) | no labels, no backprop | test accuracy | **82.3%** (300 neurons / 6k) |
| Conductance D&C (GPU, RTX 5070) | BindsNET, measured | test accuracy | **90.0% @ 1600n** (scale-tuned); 86.4% @400n; 6400/default 47.8% |
| Spike-driven MoE | firing-rate routing | router parameters | **0 learned** (vs 512), 3× compute cut |
| Temporal (TTFS) coding | latency inference + early exit | SynOps vs rate | **83.5× fewer** |
| Telemetry hub (Paradigm A) | sparse `.spk` store | size / query I/O | **61× smaller** than raster, query reads 2% |
| In-storage query (Paradigm B) | coincidence + sequence | host transfer | **162× less**, order-discriminating |
| Relational KG (Paradigm C) | link prediction | Hits@1 | **50%** TransE · **100%** RotatE (cyclic) |
| N-MNIST ingestion (real DVS) | 1.25M real events → Bronze → raster → classify | accuracy (no learning) | **71.0%** (200/100, nearest-prototype, chance 10%) · ICR 0.10 · 35% active |

---

## Architecture at a Glance

The three data-system paradigms (after an external architectural assessment), all complete:

| Paradigm | Capability | Implementation | Status |
|----------|------------|----------------|--------|
| **A** — telemetry hub | store + query multi-channel spike trains as sparse events | `snn_data_lake/spike_telemetry_hub.py` | ✅ complete (v0.12) |
| **B** — in-storage search | compile a query into an SNN, stream stored spikes, emit only matches (coincidence **and** temporal sequence) | `snn_data_lake/paradigm_b_engine.py` (+ `research/paradigm_b_genn.py` GPU) | ✅ complete (v0.20) |
| **C** — relational embeddings | a knowledge graph in spike timing; link prediction, anomaly scoring, full relation algebra (symmetric / inverse / composition) | `research/spike_knowledge_graph.py` (TransE), `research/spike_knowledge_graph_rotate.py` (RotatE), `research/spike_kg_relations.py` (algebra) | ✅ complete (v0.21–25) |

Trainable models (learn the representations):

| Model | Mechanism | File | Headline |
|-------|-----------|------|----------|
| Unsupervised STDP | rate-coded, adaptive threshold, hard-WTA | `research/snn_mnist_stdp.py` | 82.3% CPU |
| Conductance Diehl & Cook | exc/inh populations, BindsNET, GPU | `research/eth_mnist_bindsnet.py` | **90.0% @1600n** (scale-tuned); 95% via 6400 + multi-epoch |
| Latency STDP | deterministic, precomputed, burst+x_tar | `research/snn_mnist_stdp_fast.py` | 76%, 2.1× faster |
| Spike-driven MoE + STDP | firing-rate routing over expert pops | `research/snn_moe_stdp_mnist.py` | 0-param router |
| GeNN custom plasticity | v0.17 rule as a CUDA weight-update model | `research/snn_mnist_stdp_genn.py` | GPU port |

![Results by version](assets/results.svg?v=0.30)

---

## Scope: one unified project

**Core — the spiking data lake + SNN:** storage prototypes, the three paradigms (telemetry hub,
in-storage query, relational embeddings), the streaming hub (live append + record-while-query),
the trainable STDP / D&C models, N-MNIST event-camera ingestion, the Medallion lakehouse, and
the GCP scale-out.

**Robot-arm application** (built on those primitives): the real-time closed loop — signal
encoding, in-lake matching, continual learning, instinct (reflex), neuromodulators (RPE dopamine
+ cortisol), the Interpreter (matched label → robot command), and the Arduino wire contract +
the builder's sensor work. All on `main`. (The `robot-arm` branch remains on the remote for
collaborator hardware work; its contents are merged here.)

---

## Directory structure

```
spiking-neural-data-lake/
  research/spiking_storage_prototype.py     associative memory (factored O(P·k) storage)
  research/test_prototype.py                capacity / noise stress sweeps
  snn_data_lake/snn_classifier.py                supervised spiking classifier (+ `sweep` mode)
  research/snn_moe_classifier.py            spike-driven MoE routing
  research/temporal_coding_storage.py       time-to-first-spike (TTFS) latency coding
  snn_data_lake/spike_telemetry_hub.py           Paradigm A — sparse .spk store + windowed queries
  snn_data_lake/paradigm_b_matcher.py            Paradigm B — coincidence matcher (CPU, verified)
  snn_data_lake/paradigm_b_engine.py             Paradigm B — coincidence + temporal-sequence engine
  research/paradigm_b_genn.py               Paradigm B — GeNN GPU port
  snn_data_lake/spike_preprocessing.py           deterministic encode + precompute cache + Van Rossum
  research/spike_knowledge_graph.py         Paradigm C — SpikE relational embeddings (TransE)
  research/spike_knowledge_graph_rotate.py  Paradigm C — RotatE (cyclic relations, phase coding)
  research/snn_mnist_stdp.py                unsupervised STDP on MNIST (rate, snnTorch)
  research/snn_mnist_stdp_fast.py           latency STDP (deterministic, precomputed) + rate compare
  research/snn_mnist_dc.py                  from-scratch Diehl & Cook (documented negative result)
  research/snn_moe_stdp_mnist.py            MoE + STDP hybrid
  research/eth_mnist_bindsnet.py            BindsNET conductance Diehl & Cook (GPU via --gpu)
  research/snn_mnist_stdp_genn.py           GeNN custom-plasticity GPU port
  research/snn_storage_core_snntorch.py     extracted snnTorch blueprint (reference)
  research/nmnist_ingest.py                 N-MNIST event-camera ingestion (Tonic opt; synth fallback)
  research/make_results_plot.py             regenerates assets/results.svg
  docs/RUNNING.md                  local + cloud runbook
  snn_data_lake/lakehouse/medallion.py           Medallion Bronze/Silver/Gold PoC (Parquet + polars)
  infra/                           Terraform — GCP-native lakehouse infrastructure
  gcp/                             Dataproc/Vertex scaffold + deploy guide (gcp/README.md)
  research/                        source research briefs (the designs)
  assets/results.svg               results chart
```

---

## Quickstart

> **Full local + cloud runbook: [docs/RUNNING.md](docs/RUNNING.md)** — zero-dep core, the
> robot-arm closed loop, the Medallion lakehouse, real N-MNIST, GPU training, and the GCP path.

```bash
# Pure-stdlib — no install needed (these run in CI):
python research/spiking_storage_prototype.py      # associative memory + savings
python -m snn_data_lake.snn_classifier                 # supervised spiking classifier
python research/temporal_coding_storage.py        # TTFS latency coding (83× fewer ops)
python -m snn_data_lake.spike_telemetry_hub            # Paradigm A — sparse spike-train store
python -m snn_data_lake.paradigm_b_engine              # Paradigm B — coincidence + sequence queries
python research/spike_knowledge_graph.py          # Paradigm C — SpikE relational embeddings
python research/spike_knowledge_graph_rotate.py   # Paradigm C — RotatE cyclic relations
python research/spike_kg_relations.py             # Paradigm C — relation algebra (sym/inverse/composition)

# Real-data models — need deps (CPU build is fine):
pip install -r requirements.txt
python research/snn_mnist_stdp.py                                          # 74.6%
NORD_M=300 NORD_TRAIN=6000 NORD_TDECAY=0.99999 NORD_TPLUS=0.8 \
  python research/snn_mnist_stdp.py                                        # 82.3% (best CPU)
```

### GPU

Verified on an RTX 5070 (Blackwell sm_120, `torch 2.11.0+cu128`). Install the CUDA build,
then one switch:

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install bindsnet
python research/eth_mnist_bindsnet.py --gpu        # default 400 neurons / 20k → 86.4% (verified)
```

**Measured GPU results (honest):**
- **400 neurons / 20k → 86.4%** — matches Diehl & Cook 2015 (~87%). Pipeline confirmed.
- **1600 neurons / 60k, scale-tuned → 90.0%** (new best) with `NORD_INH=60 NORD_THETA_PLUS=0.20`.
  The scaling recipe: as neurons grow, *lower* the per-synapse inhibition (more neurons already
  deliver more total inhibition) and *raise* theta_plus (spread firing across more cells).
- **6400 neurons / 60k, 1 epoch, default hyperparameters → 47.8%** — naive scale-up *regresses*
  (under-inhibited + under-trained). With the recipe above + multiple epochs it should reach the
  paper's ~95%, but one 6400 epoch is ~day-scale on a single GPU → best as parallel Vertex jobs.

So: **90.0% reproduced locally (1600n); 95% is the next leg (6400n + recipe + multi-epoch, on Vertex).**
Knobs: `NORD_INH`, `NORD_THETA_PLUS`, `NORD_EXC`, `NORD_NORM`, `NORD_EPOCHS`.

```bash
# the verified 90% run:
NORD_M=1600 NORD_TRAIN=60000 NORD_INH=60 NORD_THETA_PLUS=0.20 python research/eth_mnist_bindsnet.py --gpu
```
Chart the scaling law (400 → 1600 → 6400) with `bash gpu_scaling_sweep.sh` (unbuffered).

---

## Competitive context

- **vs dense ANNs** — spiking inference does a synaptic op only when a neuron fires:
  3.6–83× fewer operations across these models, and spikes are 1-bit events vs 32-bit
  activations.
- **vs rate coding** — deterministic temporal (TTFS) coding cuts inference ~83×; latency
  STDP trains 2.1× faster at 7.9× fewer SynOps (at a measured −6.2 pt accuracy tradeoff).
- **vs the literature** — unsupervised STDP MNIST tops out at ~95% with 6400 neurons +
  full 60k (Diehl & Cook 2015). This repo reaches **82.3% on CPU** (300n) and **90.0% on
  GPU** (1600n, scale-tuned — vs the paper's ~92% at that size). The 6400/95% point needs
  the same recipe + multiple epochs at day-scale compute (Vertex), see GPU section.
- **vs [Project Nord](https://github.com/gtausa197-svg/-Project-Nord-Spiking-Neural-Network-Language-Model)**
  (a 1B-param pure-SNN LLM) — the same primitives (LIF, STDP, sparse WTA / firing-rate
  MoE, attractor memory), scaled down to small, verifiable demos.

---

## Scope & roadmap

**What this is:** small, runnable, mostly-stdlib SNN prototypes that demonstrate and
measure the spike-based data-storage thesis. All three assessment paradigms (A/B/C) have
complete implementations.

**Lakehouse path (v0.26):** `snn_data_lake/lakehouse/medallion.py` follows the *single-node-feasible*
slice of a production "spiking neural data lakehouse" roadmap — the **Medallion**
topology (Bronze raw events → Silver binned/aligned → Gold features: firing rate,
population synchrony, inverse-compression ratio) over **columnar Parquet**, queried with
polars (the local Spark/Delta substitute), ending in a deterministic latency encoding =
the SNN handoff. **Production scale-out** (Spark clusters, Delta Lake/Iceberg ACID +
time-travel, Kafka streaming, Liquid Clustering, Unity Catalog, Delta Sharing,
format-preserving encryption, federated learning) needs cloud infrastructure and is
documented as the next-scale path. A **GCP-native scaffold** for the first slice of that
path (GCS + BigLake/Iceberg + BigQuery + Dataproc Serverless + Vertex AI GPU training)
lives in [`infra/`](infra/) (Terraform) and [`gcp/`](gcp/README.md) (PySpark Medallion,
training container, submit scripts, deploy guide).

See [CHANGELOG.md](CHANGELOG.md) for the full per-version history (every version is a git
tag + GitHub release).

---

## Limitations & open problems

- **95% is the last leg — 90% is reproduced.** The conductance D&C path verifies at
  **86.4% (400n)** and **90.0% (1600n, scale-tuned: `NORD_INH=60 NORD_THETA_PLUS=0.20`)** on
  the GPU. A naive 6400/60k/1-epoch run with *default* hyperparameters measured only 47.8%
  (under-inhibited + under-trained); the scaling recipe (lower inhibition, higher theta_plus
  as neurons grow) fixes that. Reaching ~95% needs 6400 neurons + the recipe + multiple
  epochs — ~day-scale on one GPU, so best as parallel Vertex jobs. An open compute task, not
  a method gap.
- **Latency↔rate gap (−6.2 pts).** Deterministic latency STDP (76%) trails rate (82.3%) —
  an information gap (one deterministic pass vs many stochastic samples). A pair-based
  STDP kernel did **not** help (v0.18, kept opt-in, default off).
- **Determinism is mandatory for the query path.** Poisson re-encoding the same input
  gives different spike trains (Van Rossum ~13 vs 0), so stochastic encoding can't be used
  for identity/matching.
- **Paradigm B's real win needs neuromorphic silicon.** The CPU matcher is verified; the
  GeNN GPU port needs CUDA + a C++ compiler; line-rate in-storage search needs an NPU.
- **TransE can't do cyclic relations** — RotatE (phase coding) does (v0.22).

Numbers are fixed-seed; rerun to reproduce.

---

## Provenance

Designs come from two research briefs (in [`research/`](research/)) surveying SNN
data-storage methods, plus an external architectural assessment that mapped the concept to
the three paradigms above. The spike-driven MoE is a port of Project Nord's
`SpikeDrivenMoE`. License: [MIT](LICENSE).
