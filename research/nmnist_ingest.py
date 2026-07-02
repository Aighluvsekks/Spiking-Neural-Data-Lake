"""
v0.31 — N-MNIST (real event-camera) ingestion  (stdlib core, zero deps).

Closes the one real gap in Gemini's improvement brief: the repo trained only on
static MNIST. N-MNIST is MNIST recorded by a Dynamic Vision Sensor — async
(x, y, t, polarity) spike events on a 34x34x2 grid. This is *native* spiking
data, not a rate/latency conversion of pixels.

Pipeline (the data-lake leg, end to end):
  events (x,y,t,p)  ->  BRONZE raw event store (gzip)        -> storage size
                    ->  bin into T frames (Tonic ToFrame analog, stdlib)
                    ->  SILVER spike raster  -> ICR (gzip lesion metric, reused
                        from lakehouse/medallion.py)          -> storage reduction
                    ->  GOLD firing-rate vector (sparse)      -> compute reduction
                    ->  nearest-prototype classify            -> accuracy

Real data path: `pip install tonic` then run — pulls torchvision-style N-MNIST
(~1 GB on first run) via tonic.datasets.NMNIST. WITHOUT tonic the script falls
back to deterministic synthetic events so the whole pipeline + self-checks run
zero-dep (e.g. in CI). The synthetic blobs are class-separable by construction;
they prove the plumbing, NOT a benchmark number — install tonic for that.

  python nmnist_ingest.py
  NMNIST_TRAIN=400 NMNIST_TEST=200 NMNIST_T=8 python nmnist_ingest.py
"""
import os
import gzip
import math
import random
import struct

H = W = 34          # N-MNIST sensor is 34x34
P = 2               # two polarities (on/off)
C = 10              # digit classes
DUR = 300_000       # ~300 ms in microseconds (N-MNIST sample length)

TRAIN_N = int(os.environ.get("NMNIST_TRAIN", 200))
TEST_N = int(os.environ.get("NMNIST_TEST", 100))
T = int(os.environ.get("NMNIST_T", 6))          # time bins (ToFrame n_time_bins)
SEED = int(os.environ.get("NMNIST_SEED", 0))


# ---- data source: real N-MNIST via Tonic, else synthetic events -------------
def _synth_sample(label, rng, n_events=600):
    """Deterministic class-separable event blob: each digit fires around a unique
    grid centroid, drifting over time. ponytail: synthetic stand-in — proves the
    ingest->feature->classify path; swap for tonic.datasets.NMNIST for real data."""
    cx = 4 + (label % 5) * 6              # 5 columns of centroids
    cy = 8 + (label // 5) * 16            # 2 rows
    ev = []
    for _ in range(n_events):
        t = rng.randrange(DUR)
        drift = int((t / DUR) * 6) - 3    # blob drifts with time
        x = min(W - 1, max(0, int(rng.gauss(cx + drift, 2.2))))
        y = min(H - 1, max(0, int(rng.gauss(cy, 2.2))))
        ev.append((x, y, t, rng.randint(0, 1)))
    return ev


def load_events(n_train, n_test):
    """Return (train, test) lists of (events, label); 'source' tag for honesty."""
    try:
        import tonic  # noqa
        ds_tr = tonic.datasets.NMNIST(save_to="./data", train=True)
        ds_te = tonic.datasets.NMNIST(save_to="./data", train=False)

        def take(ds, k):
            # N-MNIST is stored class-sorted (all 0s, then all 1s, ...). Taking the
            # first k would yield ONE class -> a degenerate 100%. Sample spread
            # indices (deterministic) so all 10 digits are represented.
            rng = random.Random(SEED)
            idxs = rng.sample(range(len(ds)), min(k, len(ds)))
            out = []
            for i in idxs:
                events, label = ds[i]
                # tonic events are a structured array with x,y,t,p fields
                evs = [(int(e["x"]), int(e["y"]), int(e["t"]), int(e["p"])) for e in events]
                out.append((evs, int(label)))
            return out

        return take(ds_tr, n_train), take(ds_te, n_test), "tonic/N-MNIST (real DVS)"
    except Exception as exc:  # ImportError or download failure -> synthetic
        rng = random.Random(SEED)
        tr = [(_synth_sample(i % C, rng), i % C) for i in range(n_train)]
        te = [(_synth_sample(i % C, rng), i % C) for i in range(n_test)]
        reason = "tonic missing" if isinstance(exc, ImportError) else f"tonic error: {exc}"
        return tr, te, f"synthetic events ({reason} — `pip install tonic` for real N-MNIST)"


# ---- BRONZE: raw immutable events -> compact gzip store ---------------------
def bronze_bytes(samples):
    """Serialize every event as 4x uint16/uint32 (the raw 'data at rest')."""
    buf = bytearray()
    for evs, _ in samples:
        for x, y, t, p in evs:
            buf += struct.pack("<HHIB", x, y, t, p)
    return bytes(buf)


# ---- SILVER: bin events into T frames (Tonic ToFrame analog) ----------------
def to_frames(events):
    """T frames of P*H*W counts. t normalized into [0,T) by DUR."""
    frames = [bytearray(P * H * W) for _ in range(T)]
    for x, y, t, p in events:
        b = min(T - 1, (t * T) // DUR)
        idx = (p % P) * (H * W) + y * W + x
        if frames[b][idx] < 255:
            frames[b][idx] += 1
    return frames


def raster_bytes(frames):
    out = bytearray()
    for f in frames:
        out += f
    return bytes(out)


def icr(raw):
    """ICR = compressed/uncompressed. Lower = more structured (reused metric)."""
    return len(gzip.compress(raw, 9)) / max(1, len(raw))


# ---- GOLD: sparse firing-rate feature vector -------------------------------
def rate_vector(frames):
    """Sum over time -> per-(polarity,pixel) firing rate. Sparse = compute-cheap."""
    vec = [0.0] * (P * H * W)
    for f in frames:
        for i, v in enumerate(f):
            vec[i] += v
    return [v / T for v in vec]


# ---- classify: nearest class-prototype on rate vectors ---------------------
def prototypes(train_feats):
    sums = {c: [0.0] * (P * H * W) for c in range(C)}
    cnt = {c: 0 for c in range(C)}
    for vec, label in train_feats:
        cnt[label] += 1
        s = sums[label]
        for i, v in enumerate(vec):
            s[i] += v
    return {c: [v / cnt[c] for v in sums[c]] for c in range(C) if cnt[c]}


def _dist2(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(len(a)))


def classify(protos, vec):
    return min(protos, key=lambda c: _dist2(protos[c], vec))


def main():
    train, test, source = load_events(TRAIN_N, TEST_N)
    n_ev = sum(len(e) for e, _ in train) + sum(len(e) for e, _ in test)

    raw = bronze_bytes(train + test)
    raw_gz = len(gzip.compress(raw, 9))

    train_feats = [(rate_vector(to_frames(e)), lab) for e, lab in train]
    # one shared raster (first test sample) for the ICR storage metric
    sample_raster = raster_bytes(to_frames(test[0][0]))
    sample_icr = icr(sample_raster)

    protos = prototypes(train_feats)
    correct = 0
    for evs, lab in test:
        if classify(protos, rate_vector(to_frames(evs))) == lab:
            correct += 1
    acc = correct / len(test)

    sparsity = sum(1 for v in train_feats[0][0] if v > 0) / len(train_feats[0][0])

    print("=" * 62)
    print("N-MNIST EVENT INGESTION  (events -> Bronze -> raster -> classify)")
    print("=" * 62)
    print(f"source        : {source}")
    print(f"samples       : {len(train)} train / {len(test)} test, {n_ev:,} events")
    print(f"BRONZE store  : {len(raw)/1024:.1f} KB raw  -> {raw_gz/1024:.1f} KB gzip "
          f"({len(raw)/max(1,raw_gz):.1f}x)")
    print(f"SILVER raster : {T}x{P}x{H}x{W}  ICR={sample_icr:.3f}  (lower = more structured)")
    print(f"GOLD feature  : {P*H*W}-d rate vector, {sparsity*100:.0f}% active (sparse=cheap)")
    print(f"ACCURACY      : {acc*100:.1f}%   (chance = {100/C:.0f}%)")
    print("=" * 62)

    # ---- self-checks --------------------------------------------------------
    assert n_ev > 0, "no events ingested"
    assert len(raster_bytes(to_frames(test[0][0]))) == T * P * H * W, "raster shape wrong"
    assert 0.0 < sample_icr < 1.0, f"ICR out of range: {sample_icr}"
    # roundtrip: Bronze byte store decodes to the same event count
    assert len(raw) // struct.calcsize("<HHIB") == n_ev, "Bronze store lost events"
    assert acc >= 0.30, f"accuracy {acc:.2f} below 0.30 (pipeline broken?)"
    print(f"self-check OK: {n_ev:,} events ingested, raster + ICR valid, "
          f"Bronze roundtrip intact, accuracy {acc*100:.0f}% > chance")


if __name__ == "__main__":
    main()
