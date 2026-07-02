"""
v0.21 — Paradigm C complete: relational spiking knowledge-graph embeddings (SpikE).

v0.11–18 built the encoding side of Paradigm C (TTFS coding, deterministic latency,
Van Rossum query matching). The missing piece — the SpikE core — is the RELATIONAL
one: store a knowledge graph inside spike timing and reason over it.

SpikE idea, made concrete (stdlib):
  - each ENTITY is a vector of spike times  s_e ∈ R^D  (one spike latency per neuron).
  - each RELATION is a spike-time OFFSET     δ_r ∈ R^D  (a temporal translation).
  - a triple (head, rel, tail) holds when the tail's spikes are the head's spikes
    shifted by the relation:  s_tail ≈ s_head + δ_r.  (TransE, but in spike-time space.)

From triples alone we LEARN s_e and δ_r (margin-ranking SGD), then do:
  - link prediction / relational inference: (head, rel, ?) -> rank tails by how well the
    temporal translation lands (hits@1/@3, MRR).
  - anomaly evaluation: a triple's score = the spike-time mismatch ||s_h+δ_r−s_t||;
    true triples score low, implausible ones high (the assessment's "anomaly" use).

Run:  python spike_knowledge_graph.py
"""
import math
import random

random.seed(0)

GRID = [4, 4, 4]  # 3-axis lattice -> entities = product, relations = axes
D = 20            # neurons per entity population (embedding dim = spike-latency vector)
N_ENT = GRID[0] * GRID[1] * GRID[2]   # 64 entities
N_REL = len(GRID)                      # 3 relations (one temporal offset per axis)
MARGIN = 1.0
LR = 0.1
EPOCHS = 250


# ---- tiny vector ops (stdlib) -----------------------------------------------
def rand_vec(scale=1.0):
    return [random.uniform(-scale, scale) for _ in range(D)]


def sub(a, b): return [a[i] - b[i] for i in range(D)]
def add(a, b): return [a[i] + b[i] for i in range(D)]
def norm(a):   return math.sqrt(sum(x * x for x in a)) or 1e-9
def unit(a):
    n = norm(a)
    return [x / n for x in a]


# ---- build a knowledge graph WITH translational (spike-shift) structure ------
def build_graph():
    """A lattice knowledge graph: entities are 3-D grid cells, each relation r is the
    step +1 along axis r. Tail = head's neighbour one step along that axis. Translation
    holds exactly (neighbour = head + axis-offset), so TransE-in-spike-time embeds it
    cleanly and link prediction generalises the offset to held-out edges."""
    def to_idx(c):
        i = 0
        for k in range(len(GRID)):
            i = i * GRID[k] + c[k]
        return i
    coords = [(a, b, c) for a in range(GRID[0]) for b in range(GRID[1]) for c in range(GRID[2])]
    triples = []
    for cc in coords:
        for r in range(len(GRID)):
            if cc[r] + 1 < GRID[r]:
                nb = list(cc); nb[r] += 1
                triples.append((to_idx(cc), r, to_idx(tuple(nb))))
    random.shuffle(triples)
    cut = int(len(triples) * 0.85)
    return triples[:cut], triples[cut:]


# ---- learn embeddings from triples only (TransE margin-ranking SGD) ----------
def clip_ball(a):
    """TransE entity constraint: keep inside the unit ball (project only if outside).
    Unlike forcing onto the unit sphere, this lets the lattice spread out."""
    n = norm(a)
    return a if n <= 1.0 else [x / n for x in a]


def train(train_triples):
    E = [rand_vec(0.5) for _ in range(N_ENT)]
    R = [rand_vec(0.1) for _ in range(N_REL)]
    for _ in range(EPOCHS):
        random.shuffle(train_triples)
        for h, r, t in train_triples:
            tn = random.randrange(N_ENT)                 # corrupt the tail
            if tn == t:
                continue
            p = sub(add(E[h], R[r]), E[t])               # h + r − t   (true)
            n = sub(add(E[h], R[r]), E[tn])              # h + r − t'  (corrupt)
            if MARGIN + norm(p) - norm(n) <= 0:
                continue                                 # already separated
            dp, dn = unit(p), unit(n)
            g = sub(dp, dn)                              # shared grad for h and r
            for i in range(D):
                E[h][i] -= LR * g[i]
                R[r][i] -= LR * g[i]
                E[t][i] -= LR * (-dp[i])
                E[tn][i] -= LR * (dn[i])
        E = [clip_ball(e) for e in E]                    # TransE: keep entities in unit ball
    return E, R


# ---- relational inference + anomaly scoring ---------------------------------
def score(E, R, h, r, t):
    """spike-time mismatch ||s_h + δ_r − s_t|| — low = plausible triple."""
    return norm(sub(add(E[h], R[r]), E[t]))


def evaluate(E, R, test):
    hits1 = hits3 = 0
    mrr = 0.0
    for h, r, t in test:
        ranked = sorted(range(N_ENT), key=lambda e: score(E, R, h, r, e))
        rank = ranked.index(t) + 1
        hits1 += rank == 1
        hits3 += rank <= 3
        mrr += 1.0 / rank
    n = len(test)
    return hits1 / n, hits3 / n, mrr / n


def main():
    train_t, test_t = build_graph()
    E, R = train(train_t)
    h1, h3, mrr = evaluate(E, R, test_t)

    # anomaly: true test triples vs random (false) triples
    true_d = sum(score(E, R, h, r, t) for h, r, t in test_t) / len(test_t)
    rnd = [(random.randrange(N_ENT), random.randrange(N_REL), random.randrange(N_ENT))
           for _ in range(len(test_t))]
    false_d = sum(score(E, R, h, r, t) for h, r, t in rnd) / len(rnd)

    print("=" * 60)
    print("PARADIGM C — relational spiking knowledge graph (SpikE)")
    print("=" * 60)
    print(f"entities={N_ENT}  relations={N_REL}  spike-dims D={D}")
    print(f"triples: {len(train_t)} train / {len(test_t)} test\n")
    print("LINK PREDICTION (head, rel, ?) -> rank tails by spike-time translation:")
    print(f"  Hits@1 : {h1:.1%}   (random ~ {1/N_ENT:.1%})")
    print(f"  Hits@3 : {h3:.1%}")
    print(f"  MRR    : {mrr:.3f}")
    print()
    print("ANOMALY SCORING (||s_h + δ_r − s_t||, lower = plausible):")
    print(f"  mean score, true triples  : {true_d:.3f}")
    print(f"  mean score, random triples: {false_d:.3f}  ({false_d/true_d:.1f}x higher)")
    print("=" * 60)

    assert h1 >= 0.5, f"link prediction too weak: Hits@1={h1:.2f}"
    assert false_d > 2 * true_d, "anomaly scoring does not separate true vs false"
    print("self-check OK: relational inference works (Hits@1>=50%, random~1.7%), "
          "anomalies score far higher than true triples")


if __name__ == "__main__":
    main()
