"""
v0.43 — data-quality gates: block promotion to Gold unless the data passes.

The Medallion lakehouse (lakehouse/medallion.py) refines Bronze -> Silver -> Gold. This adds
the production discipline Gemini flagged: strict assertions enforced BEFORE the highly
aggregated Gold layer. stdlib + self-checked (runs in CI with zero deps); medallion.py calls
`gate()` before it writes Gold, and the streaming ingest (dataflow_ingest.py) can call the
same `gate()` in its DoFn.

Gates:
  schema       : every Bronze event is (t:int in [0,duration), channel:int in [0,n_channels))
  encodability : events serialize -> parse back IDENTICALLY (lossless, deterministic)
  immutability : Bronze content hash is stable across the pipeline (Bronze is append-only)
  gold sanity  : ICR in (0,1], rates finite & >= 0, synchrony finite & >= 0 (no NaN / inf)

  python data_quality.py
"""
import math
import struct
import hashlib

_REC = "<qi"                       # one event = int64 time + int32 channel (12 bytes)
_SZ = struct.calcsize(_REC)


class DataQualityError(AssertionError):
    """Raised when a gate fails — promotion to Gold must stop."""


def bronze_hash(rows):
    h = hashlib.sha256()
    for t, c in rows:
        h.update(struct.pack(_REC, int(t), int(c)))
    return h.hexdigest()


def check_schema(rows, n_channels, duration):
    for i, (t, c) in enumerate(rows):
        if not (isinstance(t, int) and isinstance(c, int)):
            raise DataQualityError(f"row {i}: non-int event (t={t!r}, c={c!r})")
        if not (0 <= t < duration):
            raise DataQualityError(f"row {i}: t={t} outside [0,{duration})")
        if not (0 <= c < n_channels):
            raise DataQualityError(f"row {i}: channel={c} outside [0,{n_channels})")
    return True


def check_encodable(rows):
    """Serialize then parse back — must be byte-for-byte identical (lossless round-trip)."""
    buf = b"".join(struct.pack(_REC, int(t), int(c)) for t, c in rows)
    back = [struct.unpack_from(_REC, buf, k * _SZ) for k in range(len(rows))]
    if back != [(int(t), int(c)) for t, c in rows]:
        raise DataQualityError("events not losslessly encodable (round-trip mismatch)")
    return True


def check_immutable(rows, prior_hash):
    if prior_hash is not None and bronze_hash(rows) != prior_hash:
        raise DataQualityError("Bronze mutated since ingest (immutability violated)")
    return True


def check_no_duplicates(rows):
    """No event (t, channel) may appear twice — a duplicate means double-ingest / corruption.
    (Rows are channel-major, not globally time-sorted, so we do NOT assert time ordering.)"""
    seen = set()
    for i, (t, c) in enumerate(rows):
        key = (int(t), int(c))
        if key in seen:
            raise DataQualityError(f"row {i}: duplicate event {key} (double-ingest)")
        seen.add(key)
    return True


def check_event(ev, n_channels):
    """Streaming parity: validate ONE event dict {'t','channel'} as it lands (no duration
    bound in a stream). Lets dataflow_ingest gate per-message with the same schema rules."""
    t, c = ev.get("t"), ev.get("channel")
    if not (isinstance(t, int) and isinstance(c, int)):
        raise DataQualityError(f"non-int event (t={t!r}, c={c!r})")
    if t < 0:
        raise DataQualityError(f"t={t} negative")
    if not (0 <= c < n_channels):
        raise DataQualityError(f"channel={c} outside [0,{n_channels})")
    return ev


def check_gold(gold):
    """gold: {'icr': float, 'rates': [float], 'synchrony': float}."""
    icr = gold.get("icr")
    if icr is None or math.isnan(icr) or not (0.0 < icr <= 1.0):
        raise DataQualityError(f"ICR out of (0,1]: {icr}")
    for r in gold.get("rates", []):
        if not math.isfinite(r) or r < 0:
            raise DataQualityError(f"rate invalid (NaN/inf/negative): {r}")
    s = gold.get("synchrony")
    if s is None or not math.isfinite(s) or s < 0:
        raise DataQualityError(f"synchrony invalid: {s}")
    return True


def gate(rows, n_channels, duration, gold=None, bronze_prior_hash=None):
    """Run every gate; raise DataQualityError on the first failure, else return a report.
    Call this BEFORE writing the Gold layer — a raise blocks promotion."""
    check_schema(rows, n_channels, duration)
    check_no_duplicates(rows)
    check_encodable(rows)
    check_immutable(rows, bronze_prior_hash)
    if gold is not None:
        check_gold(gold)
    return {"rows": len(rows), "bronze_hash": bronze_hash(rows)[:12],
            "n_channels": n_channels, "duration": duration, "gold_checked": gold is not None}


def main():
    n, dur = 8, 1000
    rows = [(t, t % n) for t in range(0, dur, 5)]                 # valid Bronze
    good_gold = {"icr": 0.3, "rates": [0.1, 0.2, 0.0], "synchrony": 0.5}
    rep = gate(rows, n, dur, gold=good_gold)
    bh = bronze_hash(rows)

    # every defect class must be caught
    defects = [
        ("negative t", lambda: check_schema([(-1, 0)], n, dur)),
        ("t >= duration", lambda: check_schema([(dur, 0)], n, dur)),
        ("channel out of range", lambda: check_schema([(0, n)], n, dur)),
        ("float event (non-int)", lambda: check_schema([(1.5, 0)], n, dur)),
        ("Bronze mutated", lambda: check_immutable(rows + [(1, 1)], bh)),
        ("duplicate event", lambda: check_no_duplicates([(5, 1), (5, 1)])),
        ("stream non-int", lambda: check_event({"t": 1.5, "channel": 0}, n)),
        ("stream channel oob", lambda: check_event({"t": 1, "channel": n}, n)),
        ("NaN ICR", lambda: check_gold({"icr": float("nan"), "rates": [], "synchrony": 0.0})),
        ("ICR > 1", lambda: check_gold({"icr": 1.5, "rates": [], "synchrony": 0.0})),
        ("negative rate", lambda: check_gold({"icr": 0.3, "rates": [-1.0], "synchrony": 0.0})),
        ("inf synchrony", lambda: check_gold({"icr": 0.3, "rates": [], "synchrony": float("inf")})),
    ]

    print("=" * 56)
    print("DATA-QUALITY GATES (block promotion to Gold)")
    print("=" * 56)
    print(f"valid pipeline -> gate PASS: {rep}")
    caught = 0
    for name, fn in defects:
        try:
            fn()
        except DataQualityError:
            caught += 1
            continue
        raise AssertionError(f"defect NOT caught: {name}")
    print(f"defects caught : {caught}/{len(defects)} (schema, immutability, gold sanity)")
    print("=" * 56)

    assert rep["rows"] == len(rows) and rep["gold_checked"], "gate report wrong"
    assert caught == len(defects), "a defect slipped through"
    assert check_encodable(rows), "valid events must round-trip"
    print("self-check OK: valid data passes, every defect class blocks Gold promotion")


if __name__ == "__main__":
    main()
