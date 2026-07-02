"""
v0.20 — Paradigm B complete: an in-storage spike-query ENGINE  (stdlib, zero deps).

v0.13–14 gave one query type (coincidence: >= k channels within W). A real in-storage
search engine (NPUsearch-style) needs more than "fire together" — it needs the
SNN-native query that von-Neumann scans struggle with: TEMPORAL ORDER.

`SpikeQueryEngine` wraps the v0.12 `.spk` telemetry store and compiles two query types
into detectors, each reading ONLY the queried channels from storage (partial seek) and
emitting ONLY matches to the host:

  - coincidence(channels, W, k) : >= k DISTINCT channels spike within window W
                                  (the v0.13/14 detector — order-agnostic).
  - sequence(ordered, W)        : channels fire in the GIVEN ORDER, all within W of the
                                  first — a delay-line / polychronous-group detector.
                                  This is what distinguishes Paradigm B from a coincidence
                                  filter: [5->17->42] matches, [42->17->5] does not.

Run:  python paradigm_b_engine.py
"""
import os
import random
from snn_data_lake.spike_telemetry_hub import SpikeTelemetryHub, disk_query, synth
from snn_data_lake.paradigm_b_matcher import subdetector_match   # verified distinct-channel coincidence

_MAXT = 2 ** 31 - 1


def match_sequence(events_sorted, ordered, W):
    """Detect the ordered channels firing in sequence within W of the first spike.
    Candidate state machine: a spike of ordered[stage] advances any candidate waiting
    for it; reaching the last stage emits a match. Order matters — reverse won't match."""
    L = len(ordered)
    matches, cands = [], []                      # cands: [(start_time, stage), ...]
    for t, ch in events_sorted:
        cands = [c for c in cands if t - c[0] <= W]      # prune by window
        nxt = []
        for t0, stage in cands:
            if ch == ordered[stage]:
                if stage + 1 == L:
                    matches.append(t)            # completed in order, within W
                else:
                    nxt.append((t0, stage + 1))
            else:
                nxt.append((t0, stage))
        cands = nxt
        if ch == ordered[0]:
            if L == 1:
                matches.append(t)
            else:
                cands.append((t, 1))
    return matches


class SpikeQueryEngine:
    """In-storage SNN query engine over a `.spk` telemetry file."""

    def __init__(self, spk_path):
        self.path = spk_path
        self.file_bytes = os.path.getsize(spk_path)

    def _read(self, channels):
        ev, nbytes = disk_query(self.path, 0, _MAXT, channels=sorted(set(channels)))
        ev.sort()
        return ev, nbytes

    def coincidence(self, channels, W, k):
        ev, nbytes = self._read(channels)
        matches = subdetector_match(ev, {"channels": sorted(set(channels)), "window": W, "k": k})
        return matches, len(ev), nbytes

    def sequence(self, ordered, W):
        ev, nbytes = self._read(ordered)
        return match_sequence(ev, ordered, W), len(ev), nbytes


# ---- demo + measure ---------------------------------------------------------
def main():
    N, DUR, RATE = 256, 100_000, 0.004
    BURST_CH, BURST = {7, 99}, (50_000, 52_000)
    SEQ, SEQ_WIN, SEQ_N = [5, 17, 42], (70_000, 75_000), 40
    hub = synth(N, DUR, RATE, BURST_CH, BURST)         # coincidence burst on {7,99}
    rng = random.Random(1)
    for _ in range(SEQ_N):                              # inject ordered motif 5->17->42
        t0 = rng.randint(SEQ_WIN[0], SEQ_WIN[1] - 30)
        for j, c in enumerate(SEQ):
            hub.ingest(c, [t0 + j * 3 + rng.randint(0, 2)])
    path = os.path.join(".", "data", "telemetry.spk")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hub.save(path)
    eng = SpikeQueryEngine(path)

    print("=" * 64)
    print("PARADIGM B — in-storage spike-query engine (complete)")
    print("=" * 64)
    print(f"store: {N} channels, {hub.n_events():,} spikes, {eng.file_bytes:,} B\n")

    # query 1: coincidence
    cm, cread, cbytes = eng.coincidence({7, 99}, W=50, k=2)
    cb = [m for m in cm if BURST[0] <= m < BURST[1]]
    print(f"coincidence {{7,99}} W=50 k=2 : {len(cm)} matches ({len(cb)} in burst), "
          f"read {100*cbytes/eng.file_bytes:.1f}% of file")

    # query 2: sequence (forward) vs reverse — the order discriminator
    sm, sread, sbytes = eng.sequence([5, 17, 42], W=20)
    sb = [m for m in sm if SEQ_WIN[0] <= m < SEQ_WIN[1]]
    rm, _, _ = eng.sequence([42, 17, 5], W=20)
    rb = [m for m in rm if SEQ_WIN[0] <= m < SEQ_WIN[1]]
    print(f"sequence  5->17->42  W=20  : {len(sm)} matches ({len(sb)} in motif window)")
    print(f"sequence  42->17->5  W=20  : {len(rm)} matches ({len(rb)} in motif window)  <- reverse order")
    print(f"  read {100*sbytes/eng.file_bytes:.1f}% of file (only the 3 queried channels)")
    print()
    print(f"host transfer: matches only ({len(cm)+len(sm)} stamps) vs {hub.n_events():,} raw events")
    print("=" * 64)

    # ---- self-checks --------------------------------------------------------
    assert len(cb) >= 1, "coincidence missed the burst"
    assert len(sb) >= SEQ_N * 0.7, f"sequence missed the ordered motif ({len(sb)}/{SEQ_N})"
    assert len(rb) < len(sb) / 2, "reverse-order query should NOT match the forward motif"
    assert sbytes < eng.file_bytes and cbytes < eng.file_bytes, "queries read the whole file"
    print("self-check OK: coincidence finds burst, sequence finds ordered motif AND "
          "rejects reverse order, partial reads")


if __name__ == "__main__":
    main()
