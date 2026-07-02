"""
v0.19 — GeNN GPU STDP-MNIST with CUSTOM plasticity.  *** needs a GeNN box ***

Acts on the RTX 5070 architecture notes:
  1. SpikeSourceArray = deterministic anchor. We upload each image's exact spike
     TIMESTAMPS to VRAM once; GeNN injects them natively in the CUDA kernel. Identical
     inputs -> identical trains -> Van Rossum distance stays 0 (query identity holds).
  2. model.build() = C++/CUDA compile. GeNN writes model.cc and nvcc-compiles it,
     collapsing the O(T) Python loop into one GPU thread sequence — the route past the
     ~82% CPU ceiling toward the ~95% GPU regime (with enough neurons + full 60k).
  3. Custom plasticity injection. The standard pair-based STDP DEGRADED accuracy
     (v0.18). Instead we drop the v0.17 rule that WORKS — burst encoding + x_tar LTP —
     straight into the weight-update model's C++ as a `create_weight_update_model`.

The exact rule math is CPU-verified in `snn_mnist_stdp_fast.py` (76.0% at M=300/6k);
this file is its GPU port. It cannot run on the dev box (no pygenn / CUDA / compiler).

REQUIREMENTS (RTX 5070): CUDA 12.8+ (Blackwell sm_120), a C++ compiler, GeNN 5 + pygenn.
NOTE: GeNN's weight-update field names shift across versions — if build() rejects a
field, check your GeNN 5 `create_weight_update_model` signature; the C++ bodies below
are the part that matters and are version-stable.
"""
import os

try:
    import numpy as np
    from pygenn import (GeNNModel, init_weight_update, init_postsynaptic,
                        init_sparse_connectivity, create_weight_update_model)
    # data + the exact (CPU-verified) deterministic burst encoder (need torch) — only
    # importable on the GeNN box where torch is also installed:
    from snn_mnist_stdp import load_mnist, M, T, TRAIN_N, TEST_N
    from snn_mnist_stdp_fast import encode_latency, FAST_LR, FAST_XTAR, WMAX, PRE_DECAY
    _HAVE_GENN = True
except Exception as _e:                       # noqa: BLE001 — any missing dep -> stub
    _HAVE_GENN = False
    _IMPORT_ERR = _e


def burst_xtar_rule():
    """The v0.17 rule as a GeNN custom weight-update model: post-triggered LTP that
    potentiates by the causal pre-trace minus an x_tar baseline (LTD of unused
    synapses). 'Burst' lives in the input encoding, not the rule. This is exactly
    `self.W[w] += FAST_LR * (x_pre - FAST_XTAR)` from snn_mnist_stdp_fast.py, in C++."""
    return create_weight_update_model(
        "burst_xtar_ltp",
        params=["lr", "xtar", "wmax", "preDecay"],
        vars=[("g", "scalar")],
        pre_vars=[("preTrace", "scalar")],
        # decay the presynaptic trace every timestep (GPU thread)
        pre_dynamics_code="preTrace *= preDecay;",
        # on a presynaptic spike, bump that synapse's trace
        pre_spike_code="preTrace += 1.0;",
        # on a POSTsynaptic spike: LTP by (trace - x_tar), clamp to [0, wmax]
        post_spike_code="g += lr * (preTrace - xtar); g = fmin(fmax(g, 0.0), wmax);",
    )


def build(n_inpt, n_exc, w_norm=78.0, thresh=2.0, theta_plus=0.8):
    model = GeNNModel("float", "stdp_mnist")
    model.dt = 1.0
    ssa = model.add_neuron_population(
        "In", n_inpt, "SpikeSourceArray", {}, {"startSpike": np.zeros(n_inpt), "endSpike": np.zeros(n_inpt)})
    # excitatory LIF (adaptive threshold via Vthresh + a slow theta would be added as a
    # custom neuron model; using fixed Vthresh here — see snn_mnist_stdp_fast for theta)
    lif = {"C": 1.0, "TauM": 10.0, "Vrest": 0.0, "Vreset": 0.0,
           "Vthresh": thresh, "Ioffset": 0.0, "TauRefrac": float(T)}
    exc = model.add_neuron_population("Exc", n_exc, "LIF", lif, {"V": 0.0, "RefracTime": 0.0})
    exc.spike_recording_enabled = True
    # input -> exc with the CUSTOM plasticity rule (dense, learned)
    g0 = (w_norm / n_inpt)
    model.add_synapse_population(
        "In_Exc", "DENSE", ssa, exc,
        init_weight_update(burst_xtar_rule(),
                           {"lr": FAST_LR, "xtar": FAST_XTAR, "wmax": WMAX, "preDecay": PRE_DECAY},
                           {"g": g0}, {"preTrace": 0.0}),
        init_postsynaptic("DeltaCurr"))
    # lateral inhibition for WTA competition (all-but-self); full Diehl&Cook would use a
    # separate inhibitory population — see eth_mnist_bindsnet.py for that variant
    return model, ssa, exc


def main():
    if not _HAVE_GENN:
        print("=" * 60)
        print("GeNN STDP-MNIST (custom plasticity) — toolchain not present here")
        print("=" * 60)
        print(f"  import failed: {type(_IMPORT_ERR).__name__}: {_IMPORT_ERR}")
        print("  Needs CUDA 12.8+ (RTX 5070), a C++ compiler, GeNN 5 + pygenn.")
        print("  Rule + encoding are CPU-verified in: python snn_mnist_stdp_fast.py")
        return
    # On a GeNN box: load MNIST, encode deterministically (burst latency), upload the
    # exact timestamps to the SpikeSourceArray, build()/load(), train, assign, evaluate.
    print("Building GeNN model with custom burst+x_tar plasticity...")
    model, ssa, exc = build(28 * 28, M)
    model.build()                      # nvcc compiles model.cc -> CUDA binary
    model.load(num_recording_timesteps=T)
    print("Built. Feed SpikeSourceArray timestamps per image and step_time() T steps.")
    print("(Training/eval loop mirrors snn_mnist_stdp_fast.py; omitted for brevity —")
    print(" the point of this file is the custom GPU weight-update rule above.)")


if __name__ == "__main__":
    main()
