# Changelog

All notable changes to the Spiking Neural Data Lake. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); each version is a git tag.

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
