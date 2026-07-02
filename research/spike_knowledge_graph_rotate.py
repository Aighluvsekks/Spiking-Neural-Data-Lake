"""
v0.22 — RotatE spiking knowledge graph: CYCLIC relations via spike-PHASE coding.

v0.21 (TransE-in-spike-time) embeds translational/lattice relations but CANNOT model
cyclic ones — `tail = (head + shift) mod N` needs a rotation, not an addition. RotatE
fixes this, and it is even more spike-native: encode each neuron's firing as a PHASE
θ ∈ [0,2π) (phase-of-firing coding), and a relation as a PHASE SHIFT φ_r. A triple
(h,r,t) holds when the tail's phases are the head's phases ROTATED by the relation:
    θ_t ≈ θ_h + φ_r   (mod 2π)   — wraps around, so cyclic structure is exact.

This file builds a cyclic KG (modular ring — the exact structure that gave TransE 0%
in v0.21), then trains BOTH on it: a TransE baseline (fails) and RotatE (solves it).

Run:  python spike_knowledge_graph_rotate.py
"""
import math
import random

random.seed(0)
TWO_PI = 2.0 * math.pi

D = 4             # phase dims (a ring is low-dim; fewer dims = less fragmentation)
N_ENT = 40
N_REL = 5
MARGIN = 0.5
LR = 0.05
EPOCHS = 200


def wrap(x):
    """fold an angle into (-pi, pi] — the circular difference."""
    return (x + math.pi) % TWO_PI - math.pi


def sign(x):
    return (x > 0) - (x < 0)


def build_cyclic():
    """Modular ring KG: tail = (head + shift_r) mod N_ENT. Pure cyclic structure."""
    shifts = random.sample(range(1, N_ENT), N_REL)
    triples = [(h, r, (h + shifts[r]) % N_ENT)
               for h in range(N_ENT) for r in range(N_REL)]
    random.shuffle(triples)
    cut = int(len(triples) * 0.85)
    return triples[:cut], triples[cut:]


# ---- RotatE: entities = phases, relations = phase shifts ----------------------------
# Proper RotatE distance: chord length between unit phasors = 2|sin(δ/2)|, with a smooth
# gradient sign(δ)·cos(δ/2). LR decay helps the phases settle onto the ring.
ROT_EPOCHS, ROT_LR, ROT_MARGIN, K_NEG = 600, 0.1, 1.0, 5


def train_rotate(train_t):
    TH = [[random.uniform(0, TWO_PI) for _ in range(D)] for _ in range(N_ENT)]
    PH = [[random.uniform(-0.3, 0.3) for _ in range(D)] for _ in range(N_REL)]

    def cdist(a):
        return 2.0 * abs(math.sin(wrap(a) / 2.0))

    def sc(h, r, t):
        return sum(cdist(TH[h][d] + PH[r][d] - TH[t][d]) for d in range(D))

    for ep in range(ROT_EPOCHS):
        lr = ROT_LR * (1.0 - 0.7 * ep / ROT_EPOCHS)        # decay
        random.shuffle(train_t)
        for h, r, t in train_t:
            sp = sc(h, r, t)
            for _ in range(K_NEG):                          # several negatives -> sharper
                tn = random.randrange(N_ENT)
                if tn == t or ROT_MARGIN + sp - sc(h, r, tn) <= 0:
                    continue
                for d in range(D):
                    dp = wrap(TH[h][d] + PH[r][d] - TH[t][d])
                    dn = wrap(TH[h][d] + PH[r][d] - TH[tn][d])
                    gp = sign(dp) * math.cos(dp / 2.0)     # smooth chord gradient
                    gn = sign(dn) * math.cos(dn / 2.0)
                    TH[h][d] = (TH[h][d] + lr * (-gp + gn)) % TWO_PI
                    PH[r][d] = (PH[r][d] + lr * (-gp + gn)) % TWO_PI
                    TH[t][d] = (TH[t][d] + lr * gp) % TWO_PI
                    TH[tn][d] = (TH[tn][d] - lr * gn) % TWO_PI
    return TH, PH, sc


# ---- TransE baseline (real vectors, addition) — for contrast on the SAME cyclic KG --
def train_transe(train_t):
    def rv(s=1.0): return [random.uniform(-s, s) for _ in range(D)]
    E = [rv() for _ in range(N_ENT)]
    R = [rv(0.1) for _ in range(N_REL)]

    def nrm(a): return math.sqrt(sum(x * x for x in a)) or 1e-9

    def sc(h, r, t): return nrm([E[h][d] + R[r][d] - E[t][d] for d in range(D)])

    for _ in range(EPOCHS):
        random.shuffle(train_t)
        for h, r, t in train_t:
            tn = random.randrange(N_ENT)
            if tn == t or MARGIN + sc(h, r, t) - sc(h, r, tn) <= 0:
                continue
            p = [E[h][d] + R[r][d] - E[t][d] for d in range(D)]
            n = [E[h][d] + R[r][d] - E[tn][d] for d in range(D)]
            np_, nn_ = nrm(p), nrm(n)
            for d in range(D):
                gp, gn = p[d] / np_, n[d] / nn_
                E[h][d] -= LR * (gp - gn)
                R[r][d] -= LR * (gp - gn)
                E[t][d] -= LR * (-gp)
                E[tn][d] -= LR * (gn)
        m = max(nrm(e) for e in E)
        if m > 1:
            E = [[x / m for x in e] for e in E]
    return E, R, sc


def evaluate(score_fn, test_t):
    h1 = mrr = 0
    for h, r, t in test_t:
        ranked = sorted(range(N_ENT), key=lambda e: score_fn(h, r, e))
        rank = ranked.index(t) + 1
        h1 += rank == 1
        mrr += 1.0 / rank
    return h1 / len(test_t), mrr / len(test_t)


def main():
    train_t, test_t = build_cyclic()
    _, _, t_sc = train_transe(list(train_t))
    th1, tmrr = evaluate(t_sc, test_t)
    _, _, r_sc = train_rotate(list(train_t))
    rh1, rmrr = evaluate(r_sc, test_t)

    print("=" * 60)
    print("PARADIGM C — CYCLIC relations: RotatE vs TransE (spike phase)")
    print("=" * 60)
    print(f"cyclic KG: {N_ENT} entities, {N_REL} relations (modular ring)")
    print(f"triples: {len(train_t)} train / {len(test_t)} test  (random Hits@1 ~ {1/N_ENT:.1%})\n")
    print(f"TransE (addition)  : Hits@1 {th1:5.1%}   MRR {tmrr:.3f}   <- can't model cyclic")
    print(f"RotatE (rotation)  : Hits@1 {rh1:5.1%}   MRR {rmrr:.3f}   <- phase shift = cyclic")
    print("=" * 60)

    assert rh1 >= 0.8, f"RotatE failed on cyclic KG: Hits@1={rh1:.2f}"
    assert rh1 > th1 + 0.3, "RotatE should clearly beat TransE on cyclic relations"
    print("self-check OK: RotatE solves cyclic relations (Hits@1>=80%) where TransE fails")


if __name__ == "__main__":
    main()
