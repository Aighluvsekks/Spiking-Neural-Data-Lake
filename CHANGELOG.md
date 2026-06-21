# Changelog

All notable changes to the Spiking Neural Data Lake. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); each version is a git tag.

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
