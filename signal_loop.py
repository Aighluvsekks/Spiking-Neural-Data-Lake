"""
v0.32 — real-time signal loop: Arduino signal -> spike encoding -> data lake ->
        find matching signature -> emit to Interpreter.   (stdlib core, zero deps)

This is the "my part" of the robot-arm processing loop:

  Arduino signal  ->  [encode]  ->  [data lake]  ->  [match]  ->  Interpreter
   (windowed          encode_latency   .spc lake     Van Rossum    JSON line
    samples)          (deterministic)  (audit/query) nearest ref   on stdout

It owns three boxes and nothing else: it does NOT drive servos, run the
Interpreter, or flash the Arduino. Input = signal windows; output = a matched
signature label the Interpreter maps to a robot command.

Matching: each incoming window is encoded and compared (Van Rossum distance) to a
library of enrolled REFERENCE signatures (1 per command). Nearest wins; a Lowe-style
ratio test rejects ambiguous/novel signals (best must clearly beat 2nd-best) so the
loop stays silent rather than firing a wrong command.

Matcher: the HYBRID (learned spiking classifier + Van Rossum novelty gate) is the
default — strongest under noise (see learned_matcher.py). `--fast` selects the
zero-startup template baseline.

  python signal_loop.py                       # simulated + hybrid + self-check
  python signal_loop.py --fast                # template baseline (no training)
  python signal_loop.py --serial COM3         # real Arduino (needs `pip install pyserial`)
  python signal_loop.py --serial COM3 --window 8   # buffer 8 raw lines -> 1 window
  cat windows.csv | python signal_loop.py --stdin  # exercise the wire contract, no hardware
  python signal_loop.py --serial COM3 --reflex     # instinctive STOP/WITHDRAW on danger channels
  python signal_loop.py --enroll GRIPPER_CLOSE     # record a reference into signatures.json

Continual learning: non-matching (novel) signals are recorded to data/unknowns.jsonl.
A signal that recurs >= MIN_SUPPORT times can be clustered and promoted into a new
signature — the loop grows its own vocabulary instead of silently dropping unknowns.
  python signal_loop.py --learn                    # cluster unknowns -> new DISCOVERED_n signatures
  python signal_loop.py --learn-as WAVE            # name the largest discovered cluster 'WAVE'

Arduino wire contract (full spec + example sketch in docs/arduino_contract.md):
  115200 8N1, newline-terminated lines of comma/space-separated floats. Direct mode
  (default) = one N-feature window per line; windowed mode (--window W) = W lines of
  C raw channel samples -> one window. Values any range (auto-normalized); '#' and
  unparseable lines skipped.
"""
import os
import sys
import json
import gzip
import math
import random
import zlib

from spike_preprocessing import encode_latency, van_rossum_distance, N, T
from reflex import Reflex, reflex_guard      # #2 instinctive fast-path (no cycle: reflex is leaf)

SIG_PATH = os.path.join(".", "data", "signatures.json")
LAKE_PATH = os.path.join(".", "data", "lake.spc")
UNK_PATH = os.path.join(".", "data", "unknowns.jsonl")   # novel signals, awaiting learning
RATIO = float(os.environ.get("LOOP_RATIO", 0.85))   # match only if best < RATIO*second
NOISE = float(os.environ.get("LOOP_NOISE", 0.08))   # simulated sensor jitter
MIN_SUPPORT = int(os.environ.get("LOOP_MIN_SUPPORT", 3))  # repeats before a novel -> new signature

# Default command library (the Interpreter maps these labels -> JOINT_A_ROTATE, etc.)
COMMANDS = ["JOINT_A_ROTATE", "JOINT_B_ROTATE", "GRIPPER_CLOSE", "GRIPPER_OPEN", "HOME"]


# ---- signal source ----------------------------------------------------------
def normalize(vec):
    """Min-max a raw sample window into [0,1]^N (pad/truncate to N features)."""
    v = list(vec)[:N] + [0.0] * max(0, N - len(vec))
    lo, hi = min(v), max(v)
    rng = hi - lo
    return [(x - lo) / rng for x in v] if rng > 1e-9 else [0.0] * N


def _proto(label, rng):
    """Deterministic reference vector for a command label. crc32 (not hash()) so it is
    stable across processes — a saved signatures.json must reproduce next run."""
    r = random.Random(zlib.crc32(label.encode()))
    return [r.random() if r.random() < 0.45 else 0.0 for _ in range(N)]


def simulated_stream(library, n_events, seed=0, novel_every=7):
    """Yield (true_label_or_None, window): noisy copies of enrolled refs, plus the
    occasional novel signal (true_label=None) to exercise rejection."""
    rng = random.Random(seed)
    labels = list(library)
    for k in range(n_events):
        if novel_every and k % novel_every == novel_every - 1:
            yield None, [rng.random() for _ in range(N)]            # novel/unknown
        else:
            lab = labels[k % len(labels)]
            base = library[lab]
            yield lab, [min(1.0, max(0.0, v + rng.uniform(-NOISE, NOISE))) for v in base]


# ---- Arduino wire contract (see docs/arduino_contract.md) -------------------
# A line = comma- OR space-separated floats, newline-terminated, 115200 8N1.
#   direct  (window=1): each line is one N-feature window.
#   windowed(window=W): each line is C raw channel samples; W lines -> one window
#                        (flattened row-major, resized to N).
# Values may be any range (auto min-max normalized per window). '#' lines + lines
# that don't parse are skipped, never crash the loop.
def parse_line(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    try:
        return [float(x) for x in line.replace(",", " ").split()]
    except ValueError:
        return None


def resize_to_n(vec, n=N):
    """Truncate or zero-pad a raw vector to exactly N features."""
    return vec[:n] if len(vec) >= n else vec + [0.0] * (n - len(vec))


def line_windows(raw_rows, window=1, n=N):
    """Turn parsed float rows into normalized N-feature windows."""
    if window <= 1:
        for row in raw_rows:
            yield None, normalize(resize_to_n(row, n))
    else:
        buf = []
        for row in raw_rows:
            buf.append(row)
            if len(buf) == window:
                flat = [v for r in buf for v in r]
                yield None, normalize(resize_to_n(flat, n))
                buf = []


def serial_stream(port, baud, window=1, reflex=None, on_reflex=None):
    """Read float windows from a serial port (real Arduino). Guarded import.
    If a Reflex is given, it monitors the RAW sample stream ahead of windowing."""
    import serial  # pyserial — only needed on hardware

    def rows():
        ser = serial.Serial(port, baud, timeout=1)
        while True:
            p = parse_line(ser.readline().decode("ascii", "ignore"))
            if p is not None:
                yield p

    src = reflex_guard(rows(), reflex, on_reflex) if reflex else rows()
    yield from line_windows(src, window)


def stdin_stream(window=1, reflex=None, on_reflex=None):
    """Same wire contract, read from stdin — exercise the loop with no hardware."""
    raw = (p for line in sys.stdin if (p := parse_line(line)) is not None)
    src = reflex_guard(raw, reflex, on_reflex) if reflex else raw
    yield from line_windows(src, window)


# ---- signature library (data lake of enrolled references) -------------------
def load_library():
    if os.path.exists(SIG_PATH):
        return {k: v for k, v in json.load(open(SIG_PATH)).items()}
    return build_default_library()


def build_default_library():
    return {c: _proto(c, None) for c in COMMANDS}


def save_library(lib):
    os.makedirs(os.path.dirname(SIG_PATH), exist_ok=True)
    json.dump(lib, open(SIG_PATH, "w"), indent=2)


# ---- match: encoded live signal -> nearest enrolled reference ---------------
def build_matcher(library):
    """Baseline: nearest enrolled template (Van Rossum) + Lowe ratio reject."""
    refs = [(lab, encode_latency(vec)) for lab, vec in library.items()]

    def match(window):
        q = encode_latency(window)
        scored = sorted(((van_rossum_distance(q, ev), lab) for lab, ev in refs))
        best_d, best_lab = scored[0]
        second_d = scored[1][0] if len(scored) > 1 else math.inf
        matched = best_d < RATIO * second_d            # ratio (Lowe) test
        return (best_lab if matched else None), best_d, second_d, q

    return match


def build_learned_matcher(library):
    """Stronger: supervised spiking classifier picks the command, Van Rossum
    distance gate vetoes novel signals (hybrid — see learned_matcher.py). Trains
    at build time. Returns the same (label, dist, ref, encoded) shape as build_matcher."""
    import learned_matcher as L          # lazy: avoids the signal_loop<->learned_matcher cycle
    labels = list(library)
    nC = len(labels)
    refs_enc = [encode_latency(v) for v in library.values()]
    W, _ = L.train_learned(library)
    gate = L.calibrate_gate(library, refs_enc)

    def match(window):
        idx = L.hybrid_match(W, nC, refs_enc, gate, window)
        label = labels[idx] if idx is not None else None
        return label, L.nearest_dist(refs_enc, window), gate, encode_latency(window)

    return match


# ---- continual learning: record non-matching signals, cluster, promote ------
def record_unknown_line(seq, dist, window, path=UNK_PATH):
    """Append one novel/rejected window to the unknowns store for later learning."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"seq": seq, "dist": round(dist, 4),
                            "vec": [round(v, 4) for v in window]}) + "\n")


def _mean(vecs):
    n = len(vecs)
    return [sum(col) / n for col in zip(*vecs)]


def _auto_radius(library):
    """Typical same-gesture Van Rossum distance (ref vs a jittered copy) x2.5 —
    the cluster radius scales with the data, not a magic constant."""
    rng = random.Random(0)
    ds = []
    for v in library.values():
        a = encode_latency(v)
        b = encode_latency([min(1.0, max(0.0, x + rng.uniform(-NOISE, NOISE))) for x in v])
        ds.append(van_rossum_distance(a, b))
    # x1.5: room for within-gesture spread, but below the inter-gesture gap (~x1.85)
    # so unrelated noise stays in its own singleton clusters, not absorbed.
    return (sum(ds) / len(ds)) * 1.5 if ds else 4.0


def cluster_unknowns(rows, library, min_support=MIN_SUPPORT, radius=None):
    """Greedy single-link cluster of recorded unknown windows (Van Rossum distance).
    A signal that recurs >= min_support times is a candidate new gesture.
    Returns promoted clusters [{vec: mean, size: k}] sorted by size desc."""
    radius = _auto_radius(library) if radius is None else radius
    clusters = []                                    # {members:[vec], enc: encoded mean}
    for r in rows:
        v = r["vec"]
        e = encode_latency(v)
        best, bd = None, float("inf")
        for c in clusters:
            d = van_rossum_distance(e, c["enc"])
            if d < bd:
                bd, best = d, c
        if best is not None and bd <= radius:
            best["members"].append(v)
            best["enc"] = encode_latency(_mean(best["members"]))
        else:
            clusters.append({"members": [v], "enc": e})
    promoted = [{"vec": _mean(c["members"]), "size": len(c["members"])}
                for c in clusters if len(c["members"]) >= min_support]
    promoted.sort(key=lambda p: -p["size"])
    return promoted, len(clusters), radius


def learn_unknowns(library, persist=True, label=None):
    """Cluster the recorded unknowns; promote recurring ones into the signature
    library (the 'learned' step). Optional `label` names the largest new cluster."""
    if not os.path.exists(UNK_PATH):
        return {}, 0, 0
    rows = [json.loads(l) for l in open(UNK_PATH, encoding="utf-8") if l.strip()]
    promoted, n_clusters, _ = cluster_unknowns(rows, library)
    base = sum(1 for k in library if k.startswith("DISCOVERED_"))
    added = {}
    for i, p in enumerate(promoted):
        name = label if (label and i == 0) else f"DISCOVERED_{base + i + 1}"
        library[name] = p["vec"]
        added[name] = p["size"]
    if persist and added:
        save_library(library)
        open(UNK_PATH, "w").close()                  # consumed -> avoid re-discovering
    return added, len(rows), n_clusters


# ---- data lake: persist the encoded incoming stream (audit / re-query) ------
def flush_lake(events):
    """Append the session's encoded windows to a gzip .spc lake (storage-cheap)."""
    os.makedirs(os.path.dirname(LAKE_PATH), exist_ok=True)
    buf = bytearray()
    for ev in events:
        buf += len(ev).to_bytes(4, "little")
        for t, i in ev:
            buf += t.to_bytes(2, "little") + i.to_bytes(2, "little")
    raw = bytes(buf)
    with open(LAKE_PATH, "wb") as f:
        f.write(gzip.compress(raw, 9))
    return len(raw), os.path.getsize(LAKE_PATH)


# ---- the loop ---------------------------------------------------------------
def run(source, library, matcher=None, emit=True, on_unknown=None):
    match = matcher or build_matcher(library)
    lake = []
    stats = {"events": 0, "matched": 0, "correct": 0, "rejected": 0, "labeled": 0}
    for true_lab, window in source:
        label, d, d2, q = match(window)
        lake.append(q)
        stats["events"] += 1
        if label is not None:
            stats["matched"] += 1
            if true_lab is not None and label == true_lab:
                stats["correct"] += 1
        else:
            stats["rejected"] += 1
            if on_unknown:                          # record the non-matching signal to learn later
                on_unknown(stats["events"], d, window)
        if true_lab is not None:
            stats["labeled"] += 1
        if emit:   # JSON line -> Interpreter reads stdin
            print(json.dumps({"t": stats["events"], "match": label,
                              "dist": round(d, 3), "confident": label is not None}))
            sys.stdout.flush()
    return stats, lake


def main():
    args = sys.argv[1:]
    library = load_library()

    if "--enroll" in args:
        label = args[args.index("--enroll") + 1]
        # record one window from the source as this command's reference
        src = simulated_stream({label: _proto(label, None)}, 1, seed=1)
        _, window = next(src)
        library[label] = normalize(window) if max(window) > 1.0 else window
        save_library(library)
        print(f"enrolled '{label}' -> {SIG_PATH} ({len(library)} signatures)")
        return

    if "--learn" in args or "--learn-as" in args:
        # cluster recorded non-matching signals; promote recurring ones to signatures
        label = args[args.index("--learn-as") + 1] if "--learn-as" in args else None
        added, n_rows, n_clusters = learn_unknowns(library, persist=True, label=label)
        if added:
            print(f"learned {len(added)} signature(s) from {n_rows} unknowns "
                  f"({n_clusters} clusters): "
                  + ", ".join(f"{k} (x{v})" for k, v in added.items()))
            print(f"library now {len(library)}: {list(library)}")
        else:
            print(f"nothing learned: {n_rows} unknowns in {n_clusters} clusters "
                  f"(need >= {MIN_SUPPORT} similar repeats). collect more, then --learn")
        return

    # matcher: hybrid (learned + novelty gate) is the DEFAULT; --fast = template
    fast = "--fast" in args
    if not fast:
        sys.stderr.write("training learned hybrid matcher (snn_classifier + gate)...\n")
    matcher = None if fast else build_learned_matcher(library)
    matcher_name = "template (Van Rossum)" if fast else "hybrid (learned + novelty gate)"
    window = int(args[args.index("--window") + 1]) if "--window" in args else 1

    # #2 reflex fast-path: instinctive STOP/WITHDRAW on danger channels, ahead of match
    reflex = Reflex() if "--reflex" in args else None

    def emit_reflex(action, seq):                # preemptive instinct -> Interpreter
        print(json.dumps({"reflex": action, "t_sample": seq, "preempt": True}))
        sys.stdout.flush()

    if reflex:
        sys.stderr.write(f"reflex fast-path ON: {reflex.rules}\n")

    if "--serial" in args:
        port = args[args.index("--serial") + 1]
        baud = int(args[args.index("--baud") + 1]) if "--baud" in args else 115200
        sys.stderr.write(f"reading {port}@{baud} window={window}, {len(library)} "
                         f"signatures, matcher={matcher_name}\n")
        run(serial_stream(port, baud, window, reflex, emit_reflex), library, matcher,
            on_unknown=record_unknown_line)                        # until interrupted
        return

    if "--stdin" in args:
        sys.stderr.write(f"reading stdin window={window}, matcher={matcher_name}\n")
        run(stdin_stream(window, reflex, emit_reflex), library, matcher,
            on_unknown=record_unknown_line)                        # EOF
        return

    # default: simulated stream + report + self-check
    if os.path.exists(UNK_PATH):
        open(UNK_PATH, "w").close()                  # fresh unknowns bin for the demo
    src = simulated_stream(library, 40, seed=0)
    stats, lake = run(src, library, matcher, emit=True, on_unknown=record_unknown_line)
    raw, gz = flush_lake(lake)
    acc = stats["correct"] / max(1, stats["labeled"])

    sys.stderr.write("=" * 60 + "\n")
    sys.stderr.write("SIGNAL LOOP  (encode -> lake -> match -> Interpreter)\n")
    sys.stderr.write("=" * 60 + "\n")
    sys.stderr.write(f"matcher      : {matcher_name}\n")
    sys.stderr.write(f"library      : {len(library)} signatures {list(library)}\n")
    sys.stderr.write(f"events       : {stats['events']} ({stats['labeled']} known, "
                     f"{stats['events']-stats['labeled']} novel)\n")
    sys.stderr.write(f"matched      : {stats['matched']}  rejected: {stats['rejected']}\n")
    sys.stderr.write(f"recorded     : {stats['rejected']} novel -> {UNK_PATH} "
                     f"(run `--learn` to cluster + enroll them)\n")
    sys.stderr.write(f"accuracy     : {acc:.0%} on known signals (chance "
                     f"{1/len(library):.0%})\n")
    sys.stderr.write(f"data lake    : {len(lake)} encoded windows, {raw} B -> {gz} B gzip\n")
    sys.stderr.write("=" * 60 + "\n")

    # ---- self-checks --------------------------------------------------------
    assert stats["events"] == 40, "stream did not produce all events"
    assert acc >= 0.85, f"known-signal matching too weak: {acc:.2f}"
    # novel signals must mostly be rejected, not fired as a command
    novel = stats["events"] - stats["labeled"]
    assert stats["rejected"] >= novel * 0.5, "loop fires commands on novel signals"
    assert gz < raw, "lake gzip did not shrink the encoded stream"

    # ---- Arduino wire-contract self-check (no hardware needed) ---------------
    assert parse_line("# comment") is None and parse_line("   ") is None
    assert parse_line("0.1, 0.2 0.3") == [0.1, 0.2, 0.3], "CSV/space parse"
    assert parse_line("nope,nan?") is None, "malformed line must be skipped"
    assert len(resize_to_n([1.0] * 9)) == N and len(resize_to_n([1.0] * 999)) == N
    wins = list(line_windows([[float(i)] for i in range(3 * N)], window=N))
    assert len(wins) == 3 and all(len(w[1]) == N for w in wins), "windower shape wrong"
    sys.stderr.write("self-check OK: known matched, novel rejected, lake stored, "
                     "wire contract parses\n")

    # ---- continual-learning self-check: novel recurring gesture -> learn -> match
    rng = random.Random(1)
    newg = _proto("NEW_GESTURE", None)               # a gesture NOT in the library
    pre = build_matcher(library)(newg)[0]            # unknown today -> should be rejected
    rows = [{"vec": [min(1.0, max(0.0, v + rng.uniform(-NOISE, NOISE))) for v in newg]}
            for _ in range(MIN_SUPPORT + 2)]         # it recurs a few times (gets recorded)
    rows += [{"vec": [rng.random() for _ in range(N)]} for _ in range(4)]  # scattered noise
    promoted, n_clusters, _ = cluster_unknowns(rows, library)
    lib2 = dict(library)
    for i, p in enumerate(promoted):
        lib2[f"DISCOVERED_{i + 1}"] = p["vec"]       # learn them into a copy
    post = build_matcher(lib2)(newg)[0]              # same gesture now matches
    assert pre is None, "new gesture should start novel (rejected)"
    assert promoted, "recurring novel gesture was not discovered"
    assert post and post.startswith("DISCOVERED"), "learned gesture not matched after learning"
    sys.stderr.write(f"continual-learning OK: novel rejected -> recorded -> clustered "
                     f"({len(promoted)} new, scattered noise ignored) -> now matches '{post}'\n")


if __name__ == "__main__":
    main()
