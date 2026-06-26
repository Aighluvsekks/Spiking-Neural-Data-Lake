"""
v0.41 — streaming telemetry hub: live append + query on one machine.

spike_telemetry_hub.py is BATCH — it builds the whole store in memory, then `save()` writes
the `.spk` with its channel index at save time. This is the STREAMING counterpart:

  - events are appended to a durable append-only log (`.spkl`) AS THEY FIRE (flushed)
  - an in-memory per-channel index grows on each append
  - a windowed query runs at ANY time over the growing store (record-WHILE-query)
  - restart -> `replay()` rebuilds the index by scanning the log once (crash recovery)

8 bytes/event on disk (channel u32, time u32). The distributed version of this is the cloud
path (Pub/Sub -> Dataflow -> GCS Bronze); this is the single-machine one.

  python streaming_hub.py
"""
import os
import struct
from array import array
from bisect import bisect_left

_REC = struct.Struct("<II")     # (channel, time) = 8 bytes per event


class StreamingHub:
    def __init__(self, path, n_channels, resume=False):
        if n_channels <= 0:
            raise ValueError("n_channels must be positive")
        self.path = path
        self.n = n_channels
        self.ch = [array("I") for _ in range(n_channels)]   # in-memory sorted spike times
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if resume and os.path.exists(path):
            self._replay()
        self._f = open(path, "ab")                          # durable append-only handle

    # ---- record (live) ------------------------------------------------------
    def append(self, channel, t):
        """Record one spike now: write to the durable log, update the live index."""
        if not 0 <= channel < self.n:
            raise IndexError(f"channel {channel} out of range [0,{self.n})")
        t = int(t)
        self._f.write(_REC.pack(channel, t))
        self._f.flush()                                     # durable per event
        a = self.ch[channel]
        if a and t < a[-1]:                                 # out-of-order arrival
            a.insert(bisect_left(a, t), t)                  # ponytail: O(n) insert; spikes are
        else:                                               #   normally monotonic so this is rare
            a.append(t)

    # ---- query (any time, while still recording) ----------------------------
    def query_window(self, t0, t1, channels=None):
        """events with t0 <= t < t1; touches only the requested channels (no full scan)."""
        out = []
        for c in (channels if channels is not None else range(self.n)):
            a = self.ch[c]
            for k in range(bisect_left(a, t0), bisect_left(a, t1)):
                out.append((a[k], c))
        return out

    def n_events(self):
        return sum(len(a) for a in self.ch)

    def close(self):
        self._f.close()

    # ---- recovery -----------------------------------------------------------
    def _replay(self):
        """Rebuild the in-memory index from the durable log (called on resume)."""
        with open(self.path, "rb") as f:
            data = f.read()
        for i in range(0, len(data) - _REC.size + 1, _REC.size):
            c, t = _REC.unpack(data[i:i + _REC.size])
            if 0 <= c < self.n:
                self.ch[c].append(t)
        for c in range(self.n):
            a = self.ch[c]
            if any(a[i] > a[i + 1] for i in range(len(a) - 1)):
                self.ch[c] = array("I", sorted(a))


def main():
    path = os.path.join(".", "data", "stream.spkl")
    if os.path.exists(path):
        os.remove(path)
    N, DUR = 16, 10_000
    hub = StreamingHub(path, N)

    # ---- record WHILE querying: two bursts, query between them --------------
    import random
    rng = random.Random(0)
    # burst 1 on ch3 in [1000,2000)
    for t in sorted(rng.sample(range(1000, 2000), 200)):
        hub.append(3, t)
    q1 = hub.query_window(1000, 2000, channels=[3])          # query mid-stream
    # burst 2 on ch3 in [3000,4000) — recorded AFTER the first query
    for t in sorted(rng.sample(range(3000, 4000), 150)):
        hub.append(3, t)
    q2_old = hub.query_window(1000, 2000, channels=[3])      # old window unchanged
    q2_new = hub.query_window(3000, 4000, channels=[3])      # new events now visible
    # background traffic on other channels
    for c in range(N):
        for t in sorted(rng.sample(range(DUR), 30)):
            hub.append(c, t)
    partial = hub.query_window(0, DUR, channels=[3])         # only ch3, not all 16
    q_final = hub.query_window(1000, 2000, channels=[3])     # final state of that window
    disk_bytes = os.path.getsize(path)
    total = hub.n_events()
    hub.close()

    # ---- restart: replay the durable log -----------------------------------
    hub2 = StreamingHub(path, N, resume=True)
    replayed = hub2.n_events()
    q_after = hub2.query_window(1000, 2000, channels=[3])
    hub2.close()

    print("=" * 60)
    print("STREAMING HUB  (live append + record-while-query)")
    print("=" * 60)
    print(f"recorded     : {total:,} events -> {disk_bytes:,} B durable log (8 B/event)")
    print(f"mid-stream q : window [1000,2000) ch3 -> {len(q1)} events (during recording)")
    print(f"after burst2 : same window still {len(q2_old)}; new window [3000,4000) -> {len(q2_new)}")
    print(f"partial query: ch3 over full range -> {len(partial)} events (touched 1 of {N} channels)")
    print(f"replay       : restart from log -> {replayed:,} events, window q -> {len(q_after)}")
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    assert len(q1) == 200, f"mid-stream query wrong: {len(q1)}"
    assert len(q2_old) == 200, "earlier window changed after later appends"
    assert len(q2_new) == 150, "events recorded after a query are not visible"
    assert len(partial) == 200 + 150 + 30, "partial query miscounted ch3"
    assert disk_bytes == total * _REC.size, "durable log size != events x 8"
    assert replayed == total and q_after == q_final, "replay did not reconstruct the store"
    print("self-check OK: record-while-query works, queries are partial + live, "
          "durable log replays exactly")


if __name__ == "__main__":
    main()
