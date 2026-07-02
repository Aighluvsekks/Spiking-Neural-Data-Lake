"""
v0.39 — end-to-end demo on the builder's sensor (ultrasonic distance + IR temp).

Wires the synthetic sensor dataset (make_sensor_dataset.py, modelling sensor.ino) through
the whole stack on HIS 2-channel domain:
  enroll 5 gestures (mean window per gesture)  ->  recognize held-out windows
  -> Interpreter maps gesture -> a sensor-appropriate command
  -> reflex fires on danger raw samples (too close / too hot), ahead of recognition
  -> a rewarded gesture's value rises (dopamine), under cortisol modulation.

Self-contained (generates its own data) so it runs in CI. Swap in the builder's real
captures (same format) and nothing else changes.

  python sensor_demo.py
"""
from snn_data_lake import signal_loop as S
from snn_data_lake.reflex import Reflex
from snn_data_lake.interpreter import Interpreter
from snn_data_lake import make_sensor_dataset as mk

# his gestures -> commands appropriate to a proximity/temperature arm (illustrative)
SENSOR_COMMANDS = {
    "HAND_APPROACH": "PREPARE_GRASP",
    "HAND_RETREAT":  "RELEASE",
    "HOT_OBJECT":    "WITHDRAW_HEAT",
    "COLD_OBJECT":   "PROCEED",
    "IDLE":          "IDLE_HOLD",
}
# reflex limits on the RAW (distance_m, temp_C) sample: too close (<0.10 m), or too hot (>50 C)
SENSOR_REFLEX = {0: (0.10, 4.0, "STOP"), 1: (-100.0, 50.0, "WITHDRAW")}
REFLEX_THRESHOLD = 0.02


def _fires(sample, n=3):
    r = Reflex(rules=SENSOR_REFLEX, threshold=REFLEX_THRESHOLD)
    for _ in range(n):
        a = r.step(sample)
        if a:
            return a
    return None


def build_library_and_test():
    feats, labels = mk._window_feats(mk.generate())
    norm = [S.normalize(S.resize_to_n(f)) for f in feats]
    by = {}
    for v, lab in zip(norm, labels):
        by.setdefault(lab, []).append(v)
    # enroll = mean of even instances; test = odd instances (hold-out)
    library = {g: [sum(c) / len(vs[0::2]) for c in zip(*vs[0::2])] for g, vs in by.items()}
    test = [(v, g) for g, vs in by.items() for v in vs[1::2]]
    return library, test


def main():
    library, test = build_library_and_test()
    matcher = S.build_matcher(library)
    itp = Interpreter(commands=SENSOR_COMMANDS, assume_success=True)

    # 1. recognition on held-out windows
    correct = 0
    for v, g in test:
        if matcher(v)[0] == g:
            correct += 1
    acc = correct / len(test)

    # 2. recognition -> command (Interpreter on his gestures)
    examples = {}
    for g in library:
        label = matcher(library[g])[0]
        examples[g] = itp.interpret({"match": label})[0]

    # 3. reflex on raw danger samples (ahead of recognition)
    close_hit = _fires([0.0, 25.0])      # contact (0 m) -> STOP
    hot_hit = _fires([0.6, 60.0])        # 60 C -> WITHDRAW
    safe = _fires([0.6, 28.0])           # within limits -> no fire

    # 4. closed-loop: a rewarded HAND_APPROACH raises its value (dopamine)
    val, cort = S.ValenceLearner(), S.Cortisol()
    appr = library["HAND_APPROACH"]
    v0 = val.valence(appr)
    for _ in range(30):
        _, reward = itp.interpret({"match": "HAND_APPROACH"})
        val.learn(appr, reward, lr_scale=cort.learn_rate_scale(reward))
    v1 = val.valence(appr)

    print("=" * 60)
    print("SENSOR DEMO — builder's ultrasonic+IR rig, end to end")
    print("=" * 60)
    print(f"enrolled : {list(library)}")
    print(f"recognition (held-out windows) : {acc:.0%}  (chance {1/len(library):.0%})")
    print("gesture -> command:")
    for g, c in examples.items():
        print(f"  {g:14} -> {c}")
    print(f"reflex   : contact(0.0m)->{close_hit}  hot(60C)->{hot_hit}  safe(0.6m,28C)->{safe}")
    print(f"dopamine : HAND_APPROACH value {v0:+.2f} -> {v1:+.2f} (rewarded -> learned)")
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    assert acc >= 0.9, f"gesture recognition too weak on his domain: {acc:.2f}"
    assert examples["HAND_APPROACH"] == "PREPARE_GRASP", "command map not applied"
    assert close_hit == "STOP" and hot_hit == "WITHDRAW", "reflex missed a danger sample"
    assert safe is None, "reflex fired on a safe sample"
    assert v1 > v0, "dopamine did not raise the rewarded gesture's value"
    print("self-check OK: gestures recognized + commanded, reflex catches close/hot, "
          "dopamine learns — full stack runs on the builder's sensor")


if __name__ == "__main__":
    main()
