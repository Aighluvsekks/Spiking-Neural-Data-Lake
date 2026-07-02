"""
v0.15 — spike preprocessing pipeline  (stdlib-only, zero deps).

Implements the three recommended next steps:

  1. DETERMINISTIC encoding — `encode_latency` maps each feature to ONE spike whose
     time is a pure function of intensity (brighter -> earlier). No RNG, so the same
     input always yields the same spikes (reproducible, and safe to cache).

  2. PRE-COMPUTE the spike tensors — `precompute` encodes a whole dataset once;
     `save_cache`/`load_cache` persist it. Re-using the cache across epochs removes
     spike generation from the training hot loop (the on-the-fly `torch.rand` per
     step in the STDP models is exactly the overhead this kills). A benchmark below
     shows the speedup.

  3. VAN ROSSUM filter — `van_rossum_filter` convolves a spike train with an
     exponential-decay kernel, turning discrete spikes into a continuous waveform;
     `van_rossum_distance` is the L2 between two filtered trains. This is how an
     incoming QUERY is matched against STORED data with plain numeric ops.

Run:  python spike_preprocessing.py
"""
import os
import math
import time
import struct
import random
from array import array

T = 32          # time steps
N = 64          # features
C = 5           # classes (for the query-matching demo)
FLOOR = 0.10    # features dimmer than this never spike (sparse)
TAU = 4.0       # Van Rossum decay constant (steps)


# ---- 1. deterministic latency encoding --------------------------------------
def encode_latency(vec, t_steps=T, floor=FLOOR):
    """Deterministic: each feature >= floor emits ONE spike; brighter -> earlier.
    Returns sorted [(time, feature), ...]. No randomness anywhere."""
    out = [(int(round((1.0 - v) * (t_steps - 1))), i)
           for i, v in enumerate(vec) if v >= floor]
    out.sort()
    return out


def encode_poisson(vec, t_steps=T, floor=FLOOR, rate=0.5):
    """Stochastic RATE encoding: each feature >= floor spikes with prob v*rate per
    step. NON-deterministic by design — two calls on the SAME input give DIFFERENT
    spike trains. That is exactly why a Poisson-encoded query can't be recognised as
    'the same data': an R-STDP router or a Van Rossum metric sees two distinct trains.
    Hence the query/router path must use the deterministic encoder above."""
    out = []
    for i, v in enumerate(vec):
        if v < floor:
            continue
        p = v * rate
        for t in range(t_steps):
            if random.random() < p:
                out.append((t, i))
    out.sort()
    return out


# ---- 2. precompute + cache --------------------------------------------------
def precompute(dataset, t_steps=T, floor=FLOOR):
    """Encode the whole dataset up front -> list of spike-event lists."""
    return [encode_latency(x, t_steps, floor) for x in dataset]


def save_cache(path, encoded, t_steps):
    """Persist precomputed spikes compactly (uint16 time/feature pairs)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"SPC1", len(encoded), t_steps))
        for ev in encoded:
            f.write(struct.pack("<I", len(ev)))
            a = array("H")
            for t, i in ev:
                a.append(t); a.append(i)
            f.write(a.tobytes())
    return os.path.getsize(path)


def load_cache(path):
    with open(path, "rb") as f:
        magic, n, t_steps = struct.unpack("<4sII", f.read(12))
        if magic != b"SPC1":
            raise ValueError("not a .spc cache")
        out = []
        for _ in range(n):
            (cnt,) = struct.unpack("<I", f.read(4))
            a = array("H")
            a.frombytes(f.read(cnt * 2 * 2))
            out.append([(a[k], a[k + 1]) for k in range(0, len(a), 2)])
    return out, t_steps


# ---- 3. Van Rossum filtering + distance --------------------------------------
def van_rossum_filter(spike_times, t_steps=T, tau=TAU):
    """Convolve a single channel's spikes with exp(-t/tau): a continuous waveform.
    Recurrence: w[t] = w[t-1]*decay + (spikes at t)."""
    decay = math.exp(-1.0 / tau)
    counts = [0] * t_steps
    for t in spike_times:
        if 0 <= t < t_steps:
            counts[t] += 1
    w = [0.0] * t_steps
    acc = 0.0
    for t in range(t_steps):
        acc = acc * decay + counts[t]
        w[t] = acc
    return w


def van_rossum_distance(ev_a, ev_b, n_features=N, t_steps=T, tau=TAU):
    """Distance between two encoded patterns = sqrt(sum over features of the L2^2
    between their Van Rossum waveforms). Continuous, comparable with plain math."""
    by_a = [[] for _ in range(n_features)]
    by_b = [[] for _ in range(n_features)]
    for t, i in ev_a:
        by_a[i].append(t)
    for t, i in ev_b:
        by_b[i].append(t)
    total = 0.0
    for i in range(n_features):
        fa = van_rossum_filter(by_a[i], t_steps, tau)
        fb = van_rossum_filter(by_b[i], t_steps, tau)
        total += sum((fa[t] - fb[t]) ** 2 for t in range(t_steps))
    return math.sqrt(total / tau)


# ---- Silver-tier temporal denoise -------------------------------------------
def denoise(events, window=2, min_neighbors=1):
    """Silver-tier noise filter (Gemini brief): drop ISOLATED spikes — events with fewer
    than `min_neighbors` other spikes within +/- `window` steps on the SAME channel.

    Spurious spikes (membrane leak, random synaptic firing) arrive alone; real signal fires
    in temporally correlated bursts. Removing isolated specks before learning stops noise
    from contaminating STDP / matching. Returns a new sorted [(t, feature), ...].
    Pure stdlib; opt-in (callers pass already-encoded events) so verified paths don't shift."""
    by_ch = {}
    for t, i in events:
        by_ch.setdefault(i, []).append(t)
    kept = []
    for i, times in by_ch.items():
        ts = sorted(times)
        for idx, t in enumerate(ts):
            neighbors = sum(1 for jdx, u in enumerate(ts)
                            if jdx != idx and abs(u - t) <= window)
            if neighbors >= min_neighbors:
                kept.append((t, i))
    kept.sort()
    return kept


# ---- demo + measure ---------------------------------------------------------
def main():
    random.seed(0)
    protos = [[random.random() if random.random() < 0.45 else 0.0 for _ in range(N)]
              for _ in range(C)]

    def noisy(p):
        return [min(1.0, max(0.0, v + random.uniform(-0.12, 0.12))) for v in p]

    dataset = [noisy(protos[i % C]) for i in range(400)]
    labels = [i % C for i in range(400)]

    print("=" * 60)
    print("SPIKE PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. determinism
    det = (encode_latency(dataset[0]) == encode_latency(dataset[0]))
    print(f"1. deterministic encoding : {det}  (same input -> same spikes, no RNG)")

    # 2. precompute + cache + speedup over re-encoding each 'epoch'
    EPOCHS = 8
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        _ = [encode_latency(x) for x in dataset]          # on-the-fly every epoch
    on_the_fly = time.perf_counter() - t0

    t0 = time.perf_counter()
    enc = precompute(dataset)                              # once
    for _ in range(EPOCHS):
        for ev in enc:                                     # reuse: just index
            _ = ev
    precomputed = time.perf_counter() - t0

    path = os.path.join(".", "data", "spikes.spc")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cbytes = save_cache(path, enc, T)
    enc2, _ = load_cache(path)
    print(f"2. precompute + cache     : {cbytes:,} B cache; "
          f"{EPOCHS}-epoch encode {on_the_fly*1e3:.0f}ms on-the-fly "
          f"vs {precomputed*1e3:.0f}ms precomputed "
          f"-> {on_the_fly/max(1e-6,precomputed):.1f}x less encode work")

    # 3. Van Rossum query matching: classify each query by nearest stored prototype
    proto_enc = [encode_latency(p) for p in protos]
    correct = 0
    for x, y in zip(dataset, labels):
        q = encode_latency(x)
        pred = min(range(C), key=lambda c: van_rossum_distance(q, proto_enc[c]))
        correct += (pred == y)
    acc = correct / len(dataset)
    print(f"3. Van Rossum matching    : query->nearest stored prototype acc={acc:.1%}")

    # 4. determinism is REQUIRED for query identity (Poisson breaks it)
    img = dataset[0]
    d_det = van_rossum_distance(encode_latency(img), encode_latency(img))
    d_poi = van_rossum_distance(encode_poisson(img), encode_poisson(img))
    print(f"4. query identity         : SAME image encoded twice ->")
    print(f"     deterministic latency : Van Rossum dist = {d_det:.3f}  (recognised as SAME query)")
    print(f"     Poisson (stochastic)  : Van Rossum dist = {d_poi:.2f}  (looks like DIFFERENT data!)")

    # 5. Silver-tier denoise: a real burst survives, isolated noise specks are removed
    burst = [(5, 0), (6, 0), (7, 0)]                  # correlated signal on channel 0
    noise = [(20, 1), (3, 2)]                          # lone specks on channels 1, 2
    cleaned = denoise(burst + noise, window=2, min_neighbors=1)
    print(f"5. Silver denoise         : {len(burst+noise)} spikes -> {len(cleaned)} "
          f"(burst kept, {len(burst+noise)-len(cleaned)} isolated specks dropped)")
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    assert det, "encoding is not deterministic"
    assert enc2 == enc, "cache roundtrip changed the spikes"
    assert precomputed < on_the_fly, "precompute did not reduce encode work"
    # Van Rossum: a pattern matches its own class better than a different one
    q0 = encode_latency(dataset[0])
    d_same = van_rossum_distance(q0, proto_enc[labels[0]])
    d_other = van_rossum_distance(q0, proto_enc[(labels[0] + 1) % C])
    assert d_same < d_other, "Van Rossum distance does not separate classes"
    assert acc >= 0.85, f"query matching too weak: {acc:.2f}"
    # determinism is required for query identity:
    assert d_det == 0.0, "deterministic encoding must reproduce identical spikes"
    assert d_poi > 0.0, "Poisson should produce different trains for the same input"
    assert set(cleaned) == set(burst), f"denoise should keep the burst, drop specks; got {cleaned}"
    assert denoise([(5, 0)]) == [], "a single lone spike must be removed as noise"
    print("self-check OK: deterministic, cache intact, precompute faster, "
          "Van Rossum separates & matches, Poisson breaks query identity, denoise drops isolated specks")


if __name__ == "__main__":
    main()
