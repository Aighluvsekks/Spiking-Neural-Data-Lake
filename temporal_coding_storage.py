"""
v0.11 — temporal (time-to-first-spike) coding  (stdlib-only, zero deps).

Motivated by the Gemini architectural assessment (Paradigm C / the SpikE idea:
"more significant or surprising data points cause earlier spikes"). Every other
classifier in this lake uses RATE coding — many spikes per neuron over a window.
Temporal coding instead puts the information in WHEN a neuron fires: a bright /
salient input fires once, early; a dim one fires late or not at all.

Why it matters for the project aim (less compute):
  - 1 spike per input neuron (vs many in rate coding) -> far fewer events.
  - "first answer wins": class neurons integrate the early (most salient) spikes
    first, so the readout can DECIDE as soon as one class crosses threshold and
    stop — an early exit long before the time window ends. Compute scales with
    time-to-decision, not with the full window.

This file trains a plain linear readout (analog delta rule) ONCE, then compares
two inference modes on the SAME weights:
  - rate inference  : Poisson spikes over T steps, full integration (baseline).
  - TTFS inference  : one latency-coded spike per input, early-exit at first
                      class to cross threshold.
and reports accuracy + the synaptic-operation (SynOps) reduction.
"""
import random

random.seed(0)  # deterministic run + self-check

N = 64           # input neurons
C = 5            # classes
T = 32           # time window (steps)
TRAIN_PER = 60
TEST_PER = 40
NOISE = 0.12     # additive intensity noise
LR = 0.3
EPOCHS = 12
FRAC = 0.55      # TTFS fires when leading current >= FRAC * typical final current


# ---- graded data: each class is a random analog intensity vector in [0,1] ----
def prototypes():
    ps = []
    for _ in range(C):
        p = [random.random() if random.random() < 0.45 else 0.0 for _ in range(N)]
        ps.append(p)
    return ps


def sample(proto):
    return [min(1.0, max(0.0, v + random.uniform(-NOISE, NOISE))) for v in proto]


def dataset(per, protos):
    data = [(sample(p), c) for c, p in enumerate(protos) for _ in range(per)]
    random.shuffle(data)
    return data


# ---- train a linear readout (analog delta rule) -----------------------------
def softmax(xs):
    import math
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def train(data):
    W = [[0.0] * N for _ in range(C)]
    for _ in range(EPOCHS):
        for x, y in data:
            cur = [sum(W[c][i] * x[i] for i in range(N)) for c in range(C)]
            p = softmax(cur)
            for c in range(C):
                err = (1.0 if c == y else 0.0) - p[c]
                wc = W[c]
                for i in range(N):
                    wc[i] += LR * err * x[i]
    return W


# ---- inference mode 1: rate coding (Poisson over T steps) --------------------
def infer_rate(W, x):
    cur = [0.0] * C
    ops = 0
    for _ in range(T):
        for i in range(N):
            if random.random() < x[i]:          # Poisson spike, prob = intensity
                ops += C
                for c in range(C):
                    cur[c] += W[c][i]
    return max(range(C), key=lambda c: cur[c]), ops


# ---- inference mode 2: TTFS (one latency-coded spike per input, early exit) --
def spike_time(v):
    """brighter -> earlier; below floor never spikes (sparse)."""
    if v < 0.10:
        return None
    return int((1.0 - v) * (T - 1))


def infer_ttfs(W, x, thresh):
    # bucket inputs by their single spike time
    buckets = [[] for _ in range(T)]
    for i in range(N):
        t = spike_time(x[i])
        if t is not None:
            buckets[t].append(i)
    cur = [0.0] * C
    ops = 0
    for t in range(T):
        for i in buckets[t]:                     # fire this input's one spike
            ops += C
            for c in range(C):
                cur[c] += W[c][i]
        lead = max(range(C), key=lambda c: cur[c])
        if cur[lead] >= thresh:                   # first class over threshold -> decide
            return lead, ops, t
    return max(range(C), key=lambda c: cur[c]), ops, T - 1


# ---- run + measure ----------------------------------------------------------
def main():
    protos = prototypes()
    train_data = dataset(TRAIN_PER, protos)
    test_data = dataset(TEST_PER, protos)
    W = train(train_data)

    # calibrate TTFS threshold from typical final winning current (no early exit)
    fins = []
    for x, _ in train_data:
        cur = [sum(W[c][i] * x[i] for i in range(N)) for c in range(C)]
        fins.append(max(cur))
    thresh = FRAC * (sum(fins) / len(fins))

    # evaluate both inference modes on the same weights
    r_ok = r_ops = 0
    t_ok = t_ops = t_dec = 0
    for x, y in test_data:
        pr, ro = infer_rate(W, x)
        r_ok += (pr == y); r_ops += ro
        pt, to, td = infer_ttfs(W, x, thresh)
        t_ok += (pt == y); t_ops += to; t_dec += td

    n = len(test_data)
    print("=" * 58)
    print("TEMPORAL (time-to-first-spike) CODING")
    print("=" * 58)
    print(f"inputs N={N}  classes C={C}  window T={T}  test={n}  noise={NOISE:.0%}")
    print()
    print(f"RATE inference  : acc={r_ok/n:.1%}   SynOps={r_ops:,}")
    print(f"TTFS inference  : acc={t_ok/n:.1%}   SynOps={t_ops:,}")
    print(f"  avg decision step : {t_dec/n:.1f} / {T}  (early exit)")
    print(f"  compute reduction : {r_ops/max(1,t_ops):.1f}x  (1 spike/input + early exit)")
    print()
    print("STORAGE: data encoded in spike TIMING — 1 event per salient input")
    print("         (vs a rate-coded spike train), and the most salient inputs")
    print("         arrive first, so the answer is read from the earliest events.")
    print("=" * 58)

    assert t_ok / n >= 0.85, f"TTFS accuracy too low: {t_ok/n:.2f}"
    assert t_ops < r_ops, "TTFS did not cut compute vs rate coding"
    print("self-check OK: TTFS acc>=85% AND fewer SynOps than rate coding")


if __name__ == "__main__":
    main()
