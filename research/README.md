# research/

Neuromorphic benchmarks and experiments that seeded the product (the arm-loop + data-lake at
the repo root). Kept for reproducibility and history — **not** part of the shipped product.

Two tiers:
- **Stdlib** (run in CI via `run_selfchecks.py`): `spiking_storage_prototype`, `test_prototype`,
  `snn_moe_classifier`, `temporal_coding_storage`, `spike_knowledge_graph` (+ `_rotate`),
  `spike_kg_relations`, `nmnist_ingest`, `srm_neuron`, `stats_bounds`, `make_results_plot`.
- **Torch / GPU** (own venvs, not in CI): `snn_motor_control`, `snn_world_model_rl`,
  `snn_dreamer_6dof`, `snn_storage_core_snntorch`, `eth_mnist_bindsnet`, the `snn_mnist_*`
  variants, `snn_moe_stdp_mnist`, `paradigm_b_genn`.

Run from the repo root so the root product modules resolve:
```bash
PYTHONPATH=. python research/<name>.py      # or:  python -m research.<name>
```
`snn_classifier.py` stays at the root — it's a product dependency (`learned_matcher` uses it).
