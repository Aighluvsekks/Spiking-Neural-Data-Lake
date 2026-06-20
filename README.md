# Spiking Neural Data Lake

A growing collection ("lake") of **spiking neural network** prototypes for **data
storage and retrieval**, built to answer one question:

> Can we store and recall data with **less computational power and less storage
> space** by using spikes — sparse, binary, event-driven — instead of dense
> floating-point activations?

Every prototype both *demonstrates* a spiking storage mechanism and *measures*
the two target metrics (compute = synaptic operations, storage = bytes/params)
against a dense baseline. Each version is committed and tagged so the progression
is auditable.

Most prototypes are **pure Python standard library — zero dependencies.** Only
the real-data (MNIST) files use PyTorch + snnTorch.

---

## Versions at a glance

| Ver | File | What it adds | Headline result |
|-----|------|--------------|-----------------|
| v0.1 | `spiking_storage_prototype.py` | Sparse k-WTA **associative memory** (Hebbian write, attractor recall) | 80 patterns @ 31% of N, recall through 60% cue corruption; 25.6× smaller recalled state |
| v0.1 | `test_prototype.py` | Capacity + noise stress test | confirms graceful degradation |
| v0.2 | `snn_classifier.py` | Supervised **spiking classifier** (local delta rule) | 100% on 4 shapes, 3.6× fewer ops than dense |
| v0.3 | `snn_mnist_stdp.py` | **Real MNIST** + snnTorch, **unsupervised STDP** (no labels, no backprop) | 74.6% (100 neurons / 3k imgs), 23.5× compute reduction |
| v0.4 | `snn_moe_classifier.py` | **Spike-driven MoE** routing (ported from Project Nord) | 100%, 4× compute cut, **64× smaller router** |
| v0.5 | `snn_mnist_stdp.py` | **Scale with real data** — configurable size via env vars | see CHANGELOG |

Reference file `snn_storage_core_snntorch.py` is the original snnTorch blueprint
extracted from the source research brief (encoder only — does no storage).

---

## The storage idea (one paragraph)

In a von Neumann machine, data sits in memory addresses and is shuttled to the
CPU to compute on. In these spiking nets, **data lives in the synaptic weights**
(written by a local Hebbian / STDP rule) and recall is just network dynamics —
storage and compute are co-located, so there is no shuttle. Two savings follow:
(1) a synaptic operation happens **only when a neuron spikes**, so compute scales
with sparsity, not with the dense matrix size; (2) spikes are **1-bit events**,
far cheaper to move and store than 32-bit activations, and sparse codes can be
stored as event lists (AER) instead of dense tensors.

---

## Quickstart

```bash
# Pure-stdlib prototypes — no install needed:
python spiking_storage_prototype.py     # associative memory + savings report
python test_prototype.py                # capacity / noise sweeps
python snn_classifier.py                # supervised spiking classifier
python snn_moe_classifier.py            # spike-driven MoE routing

# Real-data prototypes — need deps (CPU build is fine):
pip install -r requirements.txt
python snn_mnist_stdp.py                # unsupervised STDP on MNIST

# Scale it (v0.5) — no code edit, just env vars:
NORD_M=300 NORD_TRAIN=6000 NORD_TEST=2000 python snn_mnist_stdp.py
```

Every script prints a metrics block and ends with a runnable `assert`-based
self-check.

---

## Provenance

The designs come from two research briefs (in [`research/`](research/)) surveying
SNN data-storage methods. The spike-driven MoE routing in v0.4 is a faithful port
of the `SpikeDrivenMoE` class from
[Project Nord](https://github.com/gtausa197-svg/-Project-Nord-Spiking-Neural-Network-Language-Model)
— a 1B-parameter pure-SNN language model — scaled *down* to a readable,
verifiable stdlib form. These prototypes are the bottom rungs of the same ladder
Project Nord climbs: same primitives (LIF, STDP, sparse WTA / firing-rate MoE,
attractor memory), small enough to actually check.

---

## Honest limitations

- The associative memory wins on **activation compute + traffic + content-
  addressable recall**, not on shrinking stored weights (the weight matrix is
  N×N). It is not a byte-compressor.
- Synthetic-data classifiers (v0.2, v0.4) use easy, separable shapes — they show
  the compute/storage win, not capacity limits.
- The MNIST STDP model is a *simplified* Diehl & Cook (hard k-WTA + adaptive
  thresholds, no explicit inhibitory population), so it lands well above chance
  but below the literature's ~95%.

Numbers reported are from fixed seeds; rerun to reproduce. See
[CHANGELOG.md](CHANGELOG.md) for the per-version history.
