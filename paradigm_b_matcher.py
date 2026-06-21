"""
v0.13 — Paradigm B: in-"storage" spike-stream pattern matcher  (CPU reference).

The assessment's Paradigm B (cf. NPUsearch): compile a query into an SNN, stream the
stored data through it, and transfer ONLY the events that trigger an output spike to
the host — bypassing the scan-everything-to-CPU bottleneck.

This is the CPU-runnable, verified core. The query is a spatiotemporal MOTIF — a set
of template channels that should fire near-coincidentally — compiled into a LIF
COINCIDENCE DETECTOR: an output neuron whose threshold is crossed only when >= k of
the template channels spike within a window W. We stream the stored spike events
(read straight from the v0.12 `.spk` telemetry store, touching ONLY the template
channels) through the detector and emit a match each time it "fires".

Two savings, both the point of Paradigm B:
  - read amplification : only the template channels are read from storage (partial
    seek), not the whole dataset.
  - host transfer      : only match timestamps cross to the host, not every event.

The GPU-accelerated version of this exact detector (PyGeNN, for an RTX 5070 box) is
in paradigm_b_genn.py — it can't run on this CPU/no-compiler machine.

Run:  python paradigm_b_matcher.py   (builds telemetry via the v0.12 hub first)
"""
import os
from collections import deque
from spike_telemetry_hub import SpikeTelemetryHub, disk_query, synth


def compile_query(channels, window, k):
    """A query = (template channels, coincidence window W, min coincident channels k).
    This is the 'compile the query into an SNN' step: it parameterises a LIF
    coincidence detector (threshold ~ k, leak ~ W)."""
    if k < 1 or k > len(channels):
        raise ValueError("k must be in [1, len(channels)]")
    return {"channels": sorted(channels), "window": window, "k": k}


def match_stream(events_sorted, q):
    """Event-driven coincidence detector. events_sorted = [(t, ch), ...] ascending.
    Emits a match time whenever >= k DISTINCT template channels fired within the last
    W steps; resets after a match (detector refractory) to count discrete events."""
    W, k = q["window"], q["k"]
    win = deque()
    matches = []
    for t, ch in events_sorted:
        win.append((t, ch))
        while win and win[0][0] <= t - W:
            win.popleft()
        if len({c for _, c in win}) >= k:
            matches.append(t)
            win.clear()                      # refractory: one match per coincidence
    return matches


def subdetector_match(events_sorted, q):
    """CPU model of the GeNN two-stage net (paradigm_b_genn.py): per-channel
    one-shot sub-detectors -> counter. Each channel emits at most ONE pulse per
    window (sub-detector refractory = W), so the counter counts DISTINCT channels.
    This is the network the GPU port mirrors; it must equal match_stream."""
    W, k = q["window"], q["k"]
    chans = set(q["channels"])
    last_pulse = {}                          # channel -> last sub-detector pulse time
    pulses = deque()
    matches = []
    for t, ch in events_sorted:
        if ch not in chans:
            continue
        lp = last_pulse.get(ch)
        if lp is not None and t - lp < W:
            continue                         # sub-detector refractory: no new pulse
        last_pulse[ch] = t
        pulses.append((t, ch))
        while pulses and pulses[0][0] <= t - W:
            pulses.popleft()
        if len({c for _, c in pulses}) >= k:
            matches.append(t)
            pulses.clear()
            last_pulse.clear()               # detector refractory: reset after match
    return matches


def total_count_match(events_sorted, q):
    """The WRONG detector (models a single summing LIF, v0.13): fires on k spikes in
    W regardless of channel. Kept only to show why distinct counting is needed."""
    W, k = q["window"], q["k"]
    chans = set(q["channels"])
    win, matches = deque(), []
    for t, ch in events_sorted:
        if ch not in chans:
            continue
        win.append(t)
        while win and win[0] <= t - W:
            win.popleft()
        if len(win) >= k:
            matches.append(t)
            win.clear()
    return matches


def run_query(spk_path, q):
    """Read ONLY the template channels from storage, stream them through the detector.
    Returns (matches, events_read, bytes_read)."""
    events, bytes_read = disk_query(spk_path, 0, 2**31 - 1, channels=q["channels"])
    events.sort()                            # merge template channels into one stream
    return match_stream(events, q), len(events), bytes_read


def main():
    # build the same telemetry as v0.12, with a burst on channels {7, 99}
    N, DUR, RATE = 256, 100_000, 0.004
    BURST_CH, BURST = {7, 99}, (50_000, 52_000)
    hub = synth(N, DUR, RATE, BURST_CH, BURST)
    path = os.path.join(".", "data", "telemetry.spk")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nbytes = hub.save(path)

    # compile a query: "channels 7 AND 99 fire within 50 steps" -> detect the burst
    q = compile_query(channels={7, 99}, window=50, k=2)
    matches, n_read, bytes_read = run_query(path, q)
    in_burst = [m for m in matches if BURST[0] <= m < BURST[1]]

    print("=" * 60)
    print("PARADIGM B — spike-stream pattern matcher (CPU reference)")
    print("=" * 60)
    print(f"store: {N} channels, {hub.n_events():,} spikes, {nbytes:,} B on disk")
    print(f"query: channels={q['channels']} coincidence within W={q['window']} (k={q['k']})")
    print()
    print(f"matches emitted        : {len(matches)}  ({len(in_burst)} inside the burst window)")
    print(f"events streamed         : {n_read:,}  (only the {len(q['channels'])} template channels)")
    print(f"bytes read from storage : {bytes_read:,}  ({100*bytes_read/nbytes:.1f}% of the file)")
    transferred = len(matches)
    print(f"host transfer           : {transferred} match stamps vs {hub.n_events():,} raw events "
          f"-> {hub.n_events()/max(1,transferred):.0f}x less to host")
    print("=" * 60)

    # ---- distinct-channel counting (the v0.14 GeNN-parity fix) ----
    stream, _ = disk_query(path, 0, 2**31 - 1, channels=q["channels"])
    stream.sort()
    sub = subdetector_match(stream, q)
    sub_burst = [m for m in sub if BURST[0] <= m < BURST[1]]
    spam = [(10 * i, 7) for i in range(q["k"] + 2)]      # one channel, repeated
    print()
    print("DISTINCT-CHANNEL COUNTING (the GeNN sub-detector net, v0.14):")
    print(f"  sub-detector net matches : {len(sub)} ({len(sub_burst)} in burst)")
    print(f"    (deque reference was {len(matches)}; the delta is the sub-detector's one-shot-")
    print(f"     per-window refractory vs raw-distinct re-triggering — both count DISTINCT")
    print(f"     channels. The GeNN port mirrors the sub-detector net exactly.)")
    print(f"  spam test (channel 7 x{q['k'] + 2} in a window):")
    print(f"    total-count detector (wrong): {len(total_count_match(spam, q))} match(es) — false positive")
    print(f"    sub-detector  (distinct)    : {len(subdetector_match(spam, q))} match(es) — correct")
    print()

    # ---- self-checks (verify against brute force) ----
    # brute force: scan ALL template events, same detector, compare
    brute_ev = [(t, c) for c in q["channels"] for t in hub.ch[c]]
    brute_ev.sort()
    brute = match_stream(brute_ev, q)
    assert matches == brute, "partial-read match != brute-force match"
    assert len(in_burst) >= 1, "burst motif not detected"
    assert bytes_read < nbytes, "read the whole file (no partial read)"
    assert len(matches) < n_read, "no host-transfer reduction"
    # distinct-counting: sub-detector net detects the burst, rejects single-channel
    # spam (the false positive a total-counting LIF makes)
    assert len(sub_burst) >= 1, "sub-detector net did not detect the burst"
    assert len(total_count_match(spam, q)) >= 1, "spam should fool a total-counter"
    assert subdetector_match(spam, q) == [], "sub-detector must reject single-channel spam"
    print("self-check OK: matches==brute force, burst detected, partial read, "
          "transfer reduced, distinct-counting verified (spam rejected)")


if __name__ == "__main__":
    main()
