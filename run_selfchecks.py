"""
v0.50 — single canonical runner for every pure-stdlib self-check.

The CI module list used to live hand-typed in BOTH .github/workflows/ci.yml and
docs/RUNNING.md — add or rename a module and the two silently drift. This is the ONE
source of truth: CI runs `python run_selfchecks.py`, docs point here. Each module's
`if __name__ == "__main__"` block is its own assert-based test; this runs them as
subprocesses (isolation) and fails loudly if any non-zero exits.

All ZERO-DEP (pure stdlib). Two tiers: CORE = the product package (src/snn_data_lake/, run as
`-m snn_data_lake.<mod>`); RESEARCH = benchmarks / storage / KG demos (research/). Both CI-green;
the MNIST/GPU/cloud scripts need torch/beam and are excluded by design.

  python run_selfchecks.py          # run all, exit non-zero on any failure
  python run_selfchecks.py --list   # print the module list, one per line
"""
import os
import sys
import subprocess

# children print Unicode (φ, −, …); force UTF-8 so a Windows cp1252 console can't crash them.
# PYTHONPATH="src" so `snn_data_lake` (the package) + research/ modules importing it resolve.
_ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONPATH": "src"}

# CORE — the product: arm-loop + data-lake, clone-and-run (root).
CORE = [
    "snn_classifier", "spike_telemetry_hub", "streaming_hub", "data_quality",
    "paradigm_b_matcher", "paradigm_b_engine", "spike_preprocessing",
    "signal_loop", "learned_matcher", "reflex", "valence_stdp", "cortisol",
    "interpreter", "closed_loop", "make_sensor_dataset", "sensor_demo",
    "arm_sim", "arm_config", "population_encoding", "sensor_csv_ingest",
    "gesture_recognition", "live_arm", "serial_arm", "arm_bridge", "spiking_pid", "demo",
]
# RESEARCH — benchmarks / storage / KG demos (research/), run with root on PYTHONPATH.
RESEARCH = [
    "research/spiking_storage_prototype", "research/test_prototype", "research/snn_moe_classifier",
    "research/temporal_coding_storage", "research/spike_knowledge_graph",
    "research/spike_knowledge_graph_rotate", "research/spike_kg_relations", "research/nmnist_ingest",
    "research/srm_neuron", "research/stats_bounds", "research/make_results_plot",
]
MODULES = CORE + RESEARCH


def run_all():
    failed = []
    for m in MODULES:
        cmd = ([sys.executable, f"{m}.py"] if m.startswith("research/")   # research/: run as a script
               else [sys.executable, "-m", f"snn_data_lake.{m}"])         # CORE: run as a package module
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=_ENV)
        ok = r.returncode == 0
        print(f"{'PASS' if ok else 'FAIL'}  {m}")
        if not ok:
            failed.append(m)
            sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:] + "\n")
    print("=" * 56)
    print(f"{len(MODULES) - len(failed)}/{len(MODULES)} self-checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
    return not failed


if __name__ == "__main__":
    if "--list" in sys.argv:
        print("\n".join(MODULES))
        sys.exit(0)
    sys.exit(0 if run_all() else 1)
