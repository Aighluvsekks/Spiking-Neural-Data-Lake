# Changelog

All notable changes to the Spiking Neural Data Lake. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); each version is a git tag.

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
