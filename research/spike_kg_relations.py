"""
v0.25 — Paradigm C depth: richer relation types (the relation algebra).

v0.22 showed RotatE handles CYCLIC relations. Its real value is the full RELATION
ALGEBRA — in spike-phase coding, where a relation is a phase shift φ_r:

  - symmetric   r(a,b) ⟺ r(b,a)      ->  φ_r ≈ 0 or π   (its own inverse)
  - inverse     r2 = r1⁻¹            ->  φ_r2 ≈ −φ_r1
  - composition r3 = r1 ∘ r2         ->  φ_r3 ≈ φ_r1 + φ_r2

We build one KG that exercises all of these (a cyclic group Z_N so the phase model can
embed it exactly), train RotatE, and check TWO things:
  1. per-relation link prediction (it predicts each relation type), and
  2. the learned phases actually SATISFY the algebra (inverse = negation, symmetric =
     half-turn, composition = sum) — i.e. it learned the relational structure, not just
     memorised triples.

Run:  python spike_kg_relations.py
"""
import math
import random

random.seed(0)
TWO_PI = 2.0 * math.pi

N = 24            # entities on a ring (Z_N)
D = 4
ROT_EPOCHS, ROT_LR, ROT_MARGIN, K_NEG = 700, 0.12, 1.0, 6

# relation shifts on Z_N, chosen to exhibit each algebraic pattern
SHIFT = {
    "F": 5,           # a generic relation
    "F_inv": -5,      # its inverse  -> φ ≈ −φ_F
    "S": N // 2,      # symmetric (shift by N/2 is its own inverse) -> φ ≈ π
    "A": 3,
    "B": 7,
    "C": 10,          # composition A∘B  (3 + 7) -> φ_C ≈ φ_A + φ_B
}
RELS = list(SHIFT)
RIDX = {r: i for i, r in enumerate(RELS)}


def wrap(x):
    return (x + math.pi) % TWO_PI - math.pi


def sign(x):
    return (x > 0) - (x < 0)


def build():
    triples = [(h, RIDX[r], (h + SHIFT[r]) % N) for h in range(N) for r in RELS]
    random.shuffle(triples)
    cut = int(len(triples) * 0.85)
    return triples[:cut], triples[cut:]


def train(train_t):
    TH = [[random.uniform(0, TWO_PI) for _ in range(D)] for _ in range(N)]
    PH = [[random.uniform(-0.3, 0.3) for _ in range(D)] for _ in range(len(RELS))]

    def cd(a):
        return 2.0 * abs(math.sin(wrap(a) / 2.0))

    def sc(h, r, t):
        return sum(cd(TH[h][d] + PH[r][d] - TH[t][d]) for d in range(D))

    for ep in range(ROT_EPOCHS):
        lr = ROT_LR * (1.0 - 0.7 * ep / ROT_EPOCHS)
        random.shuffle(train_t)
        for h, r, t in train_t:
            sp = sc(h, r, t)
            for _ in range(K_NEG):
                tn = random.randrange(N)
                if tn == t or ROT_MARGIN + sp - sc(h, r, tn) <= 0:
                    continue
                for d in range(D):
                    dp = wrap(TH[h][d] + PH[r][d] - TH[t][d])
                    dn = wrap(TH[h][d] + PH[r][d] - TH[tn][d])
                    gp = sign(dp) * math.cos(dp / 2.0)
                    gn = sign(dn) * math.cos(dn / 2.0)
                    TH[h][d] = (TH[h][d] + lr * (-gp + gn)) % TWO_PI
                    PH[r][d] = (PH[r][d] + lr * (-gp + gn)) % TWO_PI
                    TH[t][d] = (TH[t][d] + lr * gp) % TWO_PI
                    TH[tn][d] = (TH[tn][d] - lr * gn) % TWO_PI
    return TH, PH, sc


def per_relation_hits1(sc, test_t):
    by_rel = {r: [0, 0] for r in range(len(RELS))}
    for h, r, t in test_t:
        rank1 = min(range(N), key=lambda e: sc(h, r, e)) == t
        by_rel[r][0] += rank1
        by_rel[r][1] += 1
    return {RELS[r]: (c / max(1, n)) for r, (c, n) in by_rel.items()}


def circ(a, b):
    """mean circular distance between two phase vectors, in [0, π]."""
    return sum(abs(wrap(a[d] - b[d])) for d in range(D)) / D


def main():
    train_t, test_t = build()
    TH, PH, sc = train(train_t)
    hits = per_relation_hits1(sc, test_t)

    # learned-phase algebra checks
    neg_F = [-PH[RIDX["F"]][d] for d in range(D)]
    inv_err = circ(PH[RIDX["F_inv"]], neg_F)                      # φ_Finv ≈ −φ_F
    # symmetric = self-inverse: 2·φ_S ≡ 0 (so φ_S ∈ {0, π}, per-dim)
    sym_err = sum(abs(wrap(2.0 * PH[RIDX["S"]][d])) for d in range(D)) / D
    comp = [PH[RIDX["A"]][d] + PH[RIDX["B"]][d] for d in range(D)]
    comp_err = circ(PH[RIDX["C"]], comp)                          # φ_C ≈ φ_A + φ_B

    print("=" * 60)
    print("PARADIGM C DEPTH — richer relation types (relation algebra)")
    print("=" * 60)
    print(f"cyclic KG Z_{N}, {len(RELS)} relations, {len(train_t)} train / {len(test_t)} test\n")
    print("Per-relation link prediction (Hits@1):")
    for r in RELS:
        print(f"  {r:6} (shift {SHIFT[r]:+d}) : {hits[r]:5.0%}")
    print()
    print("Learned-phase ALGEBRA (mean circular error, 0 = exact, lower better):")
    print(f"  inverse      φ(F_inv) ≈ −φ(F) : {inv_err:.3f} rad")
    print(f"  symmetric    2·φ(S) ≈ 0       : {sym_err:.3f} rad")
    print(f"  composition  φ(C) ≈ φ(A)+φ(B) : {comp_err:.3f} rad")
    print(f"  (random baseline ≈ {math.pi/2:.2f} rad; symmetric is looser — two basins 0/π)")
    print("=" * 60)

    assert all(h >= 0.8 for h in hits.values()), f"a relation type failed: {hits}"
    # inverse/composition recover tightly; symmetric (self-inverse, 2φ≡0) is looser but
    # still well below the random ~1.57 rad baseline.
    assert inv_err < 0.4 and comp_err < 0.4 and sym_err < 0.7, \
        f"phase algebra not recovered: inv={inv_err:.2f} sym={sym_err:.2f} comp={comp_err:.2f}"
    print("self-check OK: all relation types predicted (Hits@1>=80%) AND the algebra "
          "(inverse/symmetric/composition) recovered in the learned phases")


if __name__ == "__main__":
    main()
