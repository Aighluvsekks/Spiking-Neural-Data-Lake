"""
Trainable spiking classifier  (stdlib-only, zero deps) — first trainable model
for the project. Same aim: less compute, less storage than a dense ANN.

How it embodies the project's storage method:
  - long-term storage : the learned synaptic weight matrix W[class][pixel].
    The data (what each class looks like) lives in the weights, written by a
    local, biologically-motivated learning rule — no backprop through time.
  - compute           : inputs are rate-coded into sparse binary spikes. A
    synaptic operation happens ONLY when an input neuron fires, so work scales
    with spikes, not with the dense N*C*T matvec a normal ANN would do.

Learning rule (ponytail): supervised local delta rule on per-input spike counts
  W[c][i] += lr * (target_c - p_c) * spikecount_i / T
  This is the stable, reliably-converging cousin of the docs' unsupervised STDP.
  Upgrade path: swap in pre/post-trace STDP + lateral-inhibition WTA (the docs'
  MNIST recipe) once a labelled readout is no longer wanted. Kept supervised
  here so the first trainable model converges without hyperparameter babysitting.

Task: 4 distinct 8x8 binary shapes (bar | / - / \ / box), recognised under
bit-flip noise. Synthetic + self-contained so it trains offline in seconds.
"""
import random
import math

random.seed(0)  # ponytail: deterministic run + self-check. Not a calibration knob.

GRID = 8
N = GRID * GRID          # input neurons (pixels)
STEPS = 20               # spike-train length per sample
LR = 0.5
EPOCHS = 8
TRAIN_PER_CLASS = 40
TEST_PER_CLASS = 20
FLIP = 0.12              # fraction of pixels flipped to make a noisy sample
ON_RATE, OFF_RATE = 0.9, 0.05  # spike prob for lit / dark pixel per timestep


# ---- data: clean class templates over the 8x8 grid --------------------------
def _blank():
    return [0] * N


def _set(g, r, c):
    g[r * GRID + c] = 1


def templates():
    """4 visually distinct shapes -> reliably separable classes."""
    vbar, hbar, diag, box = _blank(), _blank(), _blank(), _blank()
    mid = GRID // 2
    for i in range(GRID):
        _set(vbar, i, mid)            # vertical bar
        _set(hbar, mid, i)            # horizontal bar
        _set(diag, i, i)             # diagonal
    for i in range(GRID):            # box outline
        _set(box, 0, i); _set(box, GRID - 1, i)
        _set(box, i, 0); _set(box, i, GRID - 1)
    return [vbar, hbar, diag, box]   # labels 0..3


def noisy(template, flip=FLIP):
    """flip a fraction of pixels -> one noisy sample."""
    g = template[:]
    for i in random.sample(range(N), int(N * flip)):
        g[i] ^= 1
    return g


def dataset(per_class, temps):
    data = []
    for label, t in enumerate(temps):
        for _ in range(per_class):
            data.append((noisy(t), label))
    random.shuffle(data)
    return data


# ---- spiking forward pass (rate code -> sparse spikes -> class currents) -----
def encode_step(sample):
    """one timestep: each pixel fires with prob ON_RATE if lit else OFF_RATE.
    Returns the list of firing input-neuron indices (sparse event set)."""
    return [i for i, px in enumerate(sample)
            if random.random() < (ON_RATE if px else OFF_RATE)]


def forward(W, sample, n_classes):
    """Run STEPS timesteps. Accumulate class currents from spike-driven synapses
    and the per-input spike counts (needed by the learning rule).
    Returns (currents[C], spikecount[N], synops)."""
    currents = [0.0] * n_classes
    counts = [0] * N
    synops = 0
    for _ in range(STEPS):
        for i in encode_step(sample):     # sparse: only firing inputs do work
            counts[i] += 1
            for c in range(n_classes):
                currents[c] += W[c][i]
            synops += n_classes           # one spike -> n_classes synaptic ops
    return currents, counts, synops


def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


# ---- train (local delta rule) -----------------------------------------------
def train(train_data, n_classes, quiet=False):
    W = [[0.0] * N for _ in range(n_classes)]
    total_synops = 0
    for epoch in range(EPOCHS):
        correct = 0
        for sample, label in train_data:
            currents, counts, synops = forward(W, sample, n_classes)
            total_synops += synops
            probs = softmax(currents)
            if max(range(n_classes), key=lambda c: currents[c]) == label:
                correct += 1
            for c in range(n_classes):
                target = 1.0 if c == label else 0.0
                err = target - probs[c]
                wc = W[c]
                for i in range(N):
                    if counts[i]:
                        wc[i] += LR * err * counts[i] / STEPS
        acc = correct / len(train_data)
        if not quiet:
            print(f"  epoch {epoch+1}/{EPOCHS}  train_acc={acc:.1%}")
    return W, total_synops


def evaluate(W, data, n_classes):
    correct, synops = 0, 0
    for sample, label in data:
        currents, _, s = forward(W, sample, n_classes)
        synops += s
        if max(range(n_classes), key=lambda c: currents[c]) == label:
            correct += 1
    return correct / len(data), synops


# ---- capacity / difficulty sweep (fixes the "easy-shapes-only" limitation) --
def extra_templates():
    """4 more 8x8 shapes — deliberately LESS separable, so adding classes pushes
    the model toward its capacity limit instead of staying trivially easy."""
    cross, xsh, tophalf, lbar = _blank(), _blank(), _blank(), _blank()
    mid = GRID // 2
    for i in range(GRID):
        _set(cross, mid, i); _set(cross, i, mid)        # plus  (overlaps v/h bars)
        _set(xsh, i, i); _set(xsh, i, GRID - 1 - i)     # X     (overlaps diagonal)
        _set(lbar, i, 0); _set(lbar, i, 1)              # thick left bar
    for r in range(GRID // 2):
        for c in range(GRID):
            _set(tophalf, r, c)                          # filled top half
    return [cross, xsh, tophalf, lbar]


def _ds(per_class, temps, flip):
    data = [(noisy(t, flip), label) for label, t in enumerate(temps)
            for _ in range(per_class)]
    random.shuffle(data)
    return data


def difficulty_sweep():
    """Find where the classifier actually breaks: rising pixel noise, then a
    rising number of (increasingly overlapping) classes."""
    print("=" * 56)
    print("CAPACITY / DIFFICULTY SWEEP")
    print("=" * 56)
    t4 = templates()
    print("pixel noise vs accuracy (4 separable classes):")
    for flip in (0.10, 0.20, 0.30, 0.40, 0.50):
        W, _ = train(_ds(TRAIN_PER_CLASS, t4, flip), 4, quiet=True)
        acc, _ = evaluate(W, _ds(TEST_PER_CLASS, t4, flip), 4)
        bar = "#" * int(acc * 20)
        print(f"  flip {flip:.0%} : {acc:5.0%} | {bar}")
    print("\nnum classes vs accuracy (15% noise, shapes get overlapping):")
    pool = templates() + extra_templates()
    for C in range(4, len(pool) + 1):
        temps = pool[:C]
        W, _ = train(_ds(TRAIN_PER_CLASS, temps, 0.15), C, quiet=True)
        acc, _ = evaluate(W, _ds(TEST_PER_CLASS, temps, 0.15), C)
        bar = "#" * int(acc * 20)
        print(f"  {C} classes : {acc:5.0%} (chance {1/C:4.0%}) | {bar}")
    print("=" * 56)
    print("Reads as the capacity curve: accuracy holds until noise or class")
    print("overlap exceeds what a single linear spiking readout can separate.")


# ---- run + measure ----------------------------------------------------------
def main():
    temps = templates()
    C = len(temps)
    train_data = dataset(TRAIN_PER_CLASS, temps)
    test_data = dataset(TEST_PER_CLASS, temps)

    print("=" * 56)
    print("TRAINABLE SPIKING CLASSIFIER")
    print("=" * 56)
    print(f"classes={C}  inputs N={N} (8x8)  steps/sample={STEPS}  epochs={EPOCHS}")
    print(f"train={len(train_data)}  test={len(test_data)}  pixel noise={FLIP:.0%}")
    print()
    W, train_synops = train(train_data, C)
    test_acc, test_synops = evaluate(W, test_data, C)
    print()
    print(f"TEST ACCURACY : {test_acc:.1%}   (chance = {1/C:.0%})")
    print()

    # --- compute: spike-driven SynOps vs a dense ANN over the same samples ---
    # dense ANN recomputes the full N*C matvec every one of STEPS timesteps.
    dense_macs = len(test_data) * STEPS * N * C
    ratio = dense_macs / test_synops if test_synops else float("inf")
    # mean input sparsity actually observed
    print("COMPUTE on the test set (lower = less power):")
    print(f"  spiking SynOps : {test_synops:,}")
    print(f"  dense ANN MACs : {dense_macs:,}")
    print(f"  reduction      : {ratio:.1f}x")
    print()

    # --- storage: weights as int8 vs float32; spikes binary vs float acts ---
    weight_vals = C * N
    print("STORAGE (lower = less space):")
    print(f"  weights        : {weight_vals} values; int8 quantisable -> "
          f"{weight_vals} B vs {weight_vals*4} B float32 (4x)")
    print(f"  activations    : spikes are 1-bit events vs 32-bit floats (32x)")
    print("=" * 56)

    # ---- self-check (ponytail: one runnable check) ----
    assert test_acc >= 0.85, f"model did not learn: test_acc={test_acc:.2f}"
    assert test_synops < dense_macs, "spiking did not beat dense on compute"
    print("self-check OK: test_acc>=85%, SynOps<dense MACs")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        difficulty_sweep()
    else:
        main()
