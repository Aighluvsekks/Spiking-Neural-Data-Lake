"""
v0.12 — Spike Telemetry Hub  (Paradigm A, stdlib-only, zero deps).

Completes the assessment's Paradigm A: a hub that ingests, stores, manages and
queries multi-channel spike-train telemetry — the kind streamed by high-density
BCIs (e.g. Neuralink, 1024 channels) or large neural simulations — as SPARSE
events, never as dense waveforms/rasters.

Core idea (the data-lake bit): a spike train is mostly silence. Store only the
events (per-channel sorted spike times = Address-Event Representation), persist
them to a self-describing file with a per-channel index, and answer
"which spikes fired on channels X in window [t0,t1)?" by SEEKING to just those
channels and binary-searching the window — reading neither other channels nor the
silent gaps into memory.

Capabilities:
  - ingest multi-channel spike times
  - .spk file format with a channel index (offset+count) -> partial reads
  - windowed range query (in-memory and disk-partial), O(log n + hits)
  - bin / firing-rate / inter-spike-interval (ISI) / burst (anomaly) detection
  - storage + query-efficiency measurement vs a dense raster

Run:  python spike_telemetry_hub.py
"""
import os
import struct
import random
from array import array
from bisect import bisect_left, bisect_right

random.seed(0)

_MAGIC = b"SPK1"
_HEADER = struct.Struct("<4sII")     # magic, n_channels, duration
_INDEX = struct.Struct("<QI")        # per-channel: byte offset, spike count


class SpikeTelemetryHub:
    """In-memory sparse store: one sorted uint32 spike-time array per channel."""

    def __init__(self, n_channels, duration):
        if n_channels <= 0 or duration <= 0:
            raise ValueError("n_channels and duration must be positive")
        self.n = n_channels
        self.dur = duration
        self.ch = [array("I") for _ in range(n_channels)]

    # ---- ingest -------------------------------------------------------------
    def ingest(self, channel, times):
        if not 0 <= channel < self.n:
            raise IndexError(f"channel {channel} out of range [0,{self.n})")
        a = self.ch[channel]
        a.extend(int(t) for t in times)
        if any(a[i] > a[i + 1] for i in range(len(a) - 1)):   # keep sorted
            self.ch[channel] = array("I", sorted(a))

    def n_events(self):
        return sum(len(a) for a in self.ch)

    # ---- persistence (.spk with channel index for partial reads) ------------
    def save(self, path):
        index, offset = [], _HEADER.size + self.n * _INDEX.size
        for a in self.ch:
            index.append((offset, len(a)))
            offset += len(a) * 4
        with open(path, "wb") as f:
            f.write(_HEADER.pack(_MAGIC, self.n, self.dur))
            for off, cnt in index:
                f.write(_INDEX.pack(off, cnt))
            for a in self.ch:
                f.write(a.tobytes())
        return os.path.getsize(path)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            magic, n, dur = _HEADER.unpack(f.read(_HEADER.size))
            if magic != _MAGIC:
                raise ValueError(f"not a .spk file (bad magic {magic!r})")
            idx = [_INDEX.unpack(f.read(_INDEX.size)) for _ in range(n)]
            hub = SpikeTelemetryHub(n, dur)
            for c, (off, cnt) in enumerate(idx):
                f.seek(off)
                a = array("I")
                a.frombytes(f.read(cnt * 4))
                hub.ch[c] = a
        return hub

    # ---- query / analyse ----------------------------------------------------
    def query_window(self, t0, t1, channels=None):
        """events with t0 <= t < t1; binary-search per channel (no full scan)."""
        out = []
        for c in (channels if channels is not None else range(self.n)):
            a = self.ch[c]
            for k in range(bisect_left(a, t0), bisect_left(a, t1)):
                out.append((a[k], c))
        return out

    def rates(self):
        return [len(a) / self.dur for a in self.ch]

    def bin(self, channel, window):
        a = self.ch[channel]
        nb = (self.dur + window - 1) // window
        counts = [0] * nb
        for t in a:
            counts[t // window] += 1
        return counts

    def isi(self, channel):
        a = self.ch[channel]
        return [a[i + 1] - a[i] for i in range(len(a) - 1)]

    def bursts(self, channel, window, k):
        """bins (anomalies) where spike count >= k — simple rate-threshold detector."""
        return [(b * window, c) for b, c in enumerate(self.bin(channel, window)) if c >= k]


def disk_query(path, t0, t1, channels):
    """Windowed query that reads ONLY the requested channels from disk — the
    'don't load the whole dataset' path. Returns (events, bytes_read)."""
    out, bytes_read = [], 0
    with open(path, "rb") as f:
        magic, n, _ = _HEADER.unpack(f.read(_HEADER.size))
        if magic != _MAGIC:
            raise ValueError("not a .spk file")
        out_events = []
        for c in channels:
            if not 0 <= c < n:
                raise IndexError(f"channel {c} out of range")
            f.seek(_HEADER.size + c * _INDEX.size)
            off, cnt = _INDEX.unpack(f.read(_INDEX.size))
            f.seek(off)
            a = array("I")
            a.frombytes(f.read(cnt * 4))
            bytes_read += cnt * 4
            for k in range(bisect_left(a, t0), bisect_left(a, t1)):
                out_events.append((a[k], c))
    return out_events, bytes_read


# ---- demo + measure ---------------------------------------------------------
def synth(n_channels, duration, rate, burst_channels, burst_win):
    """simulate sparse multi-channel telemetry; inject a dense burst on a few
    channels inside burst_win (the 'anomaly' to be detected/queried)."""
    hub = SpikeTelemetryHub(n_channels, duration)
    for c in range(n_channels):
        k = max(1, int(rate * duration))
        times = sorted(random.sample(range(duration), k))
        if c in burst_channels:                      # add a dense burst
            b0, b1 = burst_win
            # set(): a neuron fires at most once per step (canonical AER) — the burst sample
            # can land on a baseline time, and a duplicate (t,channel) is not a real 2nd spike.
            times = sorted(set(times + random.sample(range(b0, b1), (b1 - b0) // 3)))
        hub.ingest(c, times)
    return hub


def main():
    N, DUR, RATE = 256, 100_000, 0.004          # 256 channels, ~400 spikes each
    BURST_CH, BURST = {7, 99}, (50_000, 52_000)
    hub = synth(N, DUR, RATE, BURST_CH, BURST)
    path = os.path.join(".", "data", "telemetry.spk")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nbytes = hub.save(path)
    ev = hub.n_events()

    print("=" * 60)
    print("SPIKE TELEMETRY HUB  (Paradigm A)")
    print("=" * 60)
    print(f"channels={N}  duration={DUR:,} steps  total spikes={ev:,}")
    print()

    # storage: sparse AER file vs dense rasters
    dense_byte = N * DUR                          # 1 byte / (channel,step)
    dense_bit = N * DUR // 8                       # 1 bit  / (channel,step)
    print("STORAGE (lower = less space):")
    print(f"  sparse .spk file : {nbytes:,} B")
    print(f"  dense 1-bit raster: {dense_bit:,} B  -> {dense_bit/nbytes:.1f}x larger")
    print(f"  dense 1-byte raster: {dense_byte:,} B  -> {dense_byte/nbytes:.0f}x larger")
    print()

    # windowed partial-read query: only the 2 burst channels in the burst window
    events, read = disk_query(path, BURST[0], BURST[1], channels=sorted(BURST_CH))
    print("WINDOWED QUERY (channels {7,99}, window [50000,52000)):")
    print(f"  events returned : {len(events):,}")
    print(f"  bytes read      : {read:,}  ({100*read/nbytes:.1f}% of the file)")
    print()

    # analyse: firing rate + burst (anomaly) detection
    rts = hub.rates()
    print("ANALYSIS:")
    print(f"  mean firing rate : {1000*sum(rts)/len(rts):.2f} mHz/step (avg over channels)")
    found = hub.bursts(7, window=2000, k=300)
    print(f"  burst detector on ch7 (>=300 spikes/2000-step bin): {len(found)} hit(s) at {found[:1]}")
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    hub2 = SpikeTelemetryHub.load(path)            # roundtrip integrity
    assert hub2.n_events() == ev, "save/load lost events"
    brute = [(t, c) for c in sorted(BURST_CH) for t in hub.ch[c] if BURST[0] <= t < BURST[1]]
    assert sorted(events) == sorted(brute), "disk windowed query != brute force"
    assert read < nbytes, "partial read touched the whole file"
    assert nbytes < dense_bit, "sparse store not smaller than 1-bit raster"
    assert len(found) >= 1, "injected burst not detected"
    print("self-check OK: roundtrip intact, windowed query correct & partial, "
          "sparse < dense, burst detected")


if __name__ == "__main__":
    main()
