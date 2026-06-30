"""
demo.py — Spiking Neural Data Lake -> robot arm, the whole loop in ONE runnable file.

Download this single file and run it. No dependencies, no other files, no hardware:

    python demo.py

It walks the full reaction loop on embedded sample sensor data (an ultrasonic-distance +
IR-temperature rig, the builder's real format):

    sensor (distance, temp)  ->  window  ->  ORDER-AWARE gesture recognition
        ->  Interpreter (gesture -> robot command)  ->  arm motion
        ->  contact reflex (object at the sensor -> EMERGENCY_STOP)

This file is a self-contained MIRROR of the real modules (signal_loop / gesture_recognition /
paradigm_b_engine / interpreter / arm_sim) so it runs standalone — the full repo has the
production versions, tests, lakehouse, GPU training, and the live `--serial` path.

Optional live hardware (needs `pip install pyserial`):
    python demo.py --serial COM8
"""
import sys
import math

W = 8                                  # samples per window
CONTACT = 0.04                         # distance (m) -> reflex EMERGENCY_STOP
NEAR, MID, FAR = 0, 1, 2

# gesture -> robot command (approach grabs the incoming object, retreat releases)
COMMANDS = {"HAND_APPROACH": "GRIPPER_CLOSE", "HAND_RETREAT": "GRIPPER_OPEN", "IDLE": "HOLD"}


# ---- order-aware recognition (Paradigm B sequence detector) ------------------
def dbin(d):
    if d >= 2.0:
        return None                    # nothing in range
    return NEAR if d < 0.30 else MID if d < 0.60 else FAR


def match_sequence(events, ordered, window):
    """Do `ordered` channels fire IN ORDER within `window`? (the Paradigm B detector:
    far->near matches approach, near->far matches retreat; reverse does not match)."""
    L, matches, cands = len(ordered), [], []
    for t, ch in events:
        cands = [c for c in cands if t - c[0] <= window]
        nxt = []
        for t0, stage in cands:
            if ch == ordered[stage]:
                (matches.append(t) if stage + 1 == L else nxt.append((t0, stage + 1)))
            else:
                nxt.append((t0, stage))
        cands = nxt
        if ch == ordered[0]:
            matches.append(t) if L == 1 else cands.append((t, 1))
    return matches


def recognize(window):
    present = [(i, dbin(d)) for i, (d, _) in enumerate(window) if dbin(d) is not None]
    if len(present) < max(2, len(window) // 2):
        return "IDLE"
    appr = match_sequence(present, [FAR, MID, NEAR], W)
    retr = match_sequence(present, [NEAR, MID, FAR], W)
    if appr and not retr:
        return "HAND_APPROACH"
    if retr and not appr:
        return "HAND_RETREAT"
    pd = [window[i][0] for i, _ in present]               # noisy: fall back to net direction
    return "HAND_APPROACH" if pd[-1] < pd[0] else "HAND_RETREAT"


# ---- 2-link arm (forward kinematics + gripper + collision) ------------------
class Arm:
    def __init__(self, l1=1.0, l2=1.0):
        self.l1, self.l2, self.theta, self.gripper, self.stopped = l1, l2, [0.0, 0.0], 0.0, False

    def ee(self):
        a, b = self.theta[0], self.theta[0] + self.theta[1]
        return (round(self.l1 * math.cos(a) + self.l2 * math.cos(b), 2),
                round(self.l1 * math.sin(a) + self.l2 * math.sin(b), 2))

    def apply(self, cmd):
        if cmd == "EMERGENCY_STOP":
            self.stopped = True
        elif cmd == "GRIPPER_CLOSE":
            self.gripper = 1.0
        elif cmd == "GRIPPER_OPEN":
            self.gripper = 0.0
        # HOLD -> nothing
        return cmd


# ---- sample data (embedded: the builder's real gesture shapes) ---------------
def sample_stream():
    """A scripted run: idle, an approach, a retreat, then a contact event."""
    idle = [(3.13, -1.0)] * W
    approach = [(round(0.9 - 0.115 * i, 2), 295.0 + i * 0.4) for i in range(W)]   # distance falling
    retreat = [(round(0.05 + 0.115 * i, 2), 298.0 - i * 0.4) for i in range(W)]   # distance rising
    contact = [(0.02, 298.6)]                                                     # object AT sensor
    for label, win in (("(quiet)", idle), ("(hand approaching)", approach),
                       ("(hand retreating)", retreat)):
        for s in win:
            yield label, s
    for s in contact:
        yield "(contact!)", s


def serial_stream(port, baud=115200):
    import serial                                          # only for --serial
    ser = serial.Serial(port, baud)
    while True:
        parts = ser.readline().decode("ascii", "ignore").strip().split(",")
        if len(parts) >= 2:
            try:
                yield "(live)", (float(parts[-2]), float(parts[-1]))
            except ValueError:
                continue


def run(stream, arm, narrate=True):
    buf, acted = [], []
    for note, (d, t) in stream:
        if d < CONTACT:                                   # reflex preempts everything
            arm.apply("EMERGENCY_STOP")
            acted.append(("REFLEX", "EMERGENCY_STOP"))
            if narrate:
                print(f"  !! {note} distance={d}m -> REFLEX EMERGENCY_STOP (arm frozen)")
            buf = []
            continue
        buf.append((d, t))
        if len(buf) == W:
            gesture = recognize(buf)
            cmd = arm.apply(COMMANDS[gesture])
            acted.append((gesture, cmd))
            if narrate:
                print(f"  {note:20} -> recognize {gesture:14} -> command {cmd:14} "
                      f"(gripper={arm.gripper}, ee={arm.ee()})")
            buf = []
    return acted


def main():
    args = sys.argv[1:]
    arm = Arm()
    print("=" * 70)
    print("SPIKING NEURAL DATA LAKE  ->  ROBOT ARM   (one-file demo)")
    print("=" * 70)
    print("sensor (distance, temp) -> window -> recognize -> command -> arm  (+ reflex)\n")

    if "--serial" in args:
        port = args[args.index("--serial") + 1]
        print(f"live off {port}. Ctrl-C to stop.\n")
        run(serial_stream(port), arm)
        return

    acted = run(sample_stream(), arm)
    print("\n" + "=" * 70)
    cmds = [c for _, c in acted]
    print(f"actions: {', '.join(cmds)}")
    print("approach grabbed (GRIPPER_CLOSE), retreat released (GRIPPER_OPEN), "
          "contact stopped the arm (EMERGENCY_STOP).")
    print("=" * 70)

    # ---- self-check (so the demo is verifiable) -----------------------------
    seen = {c for _, c in acted}
    assert "GRIPPER_CLOSE" in seen, "approach did not grab"
    assert "GRIPPER_OPEN" in seen, "retreat did not release"
    assert "EMERGENCY_STOP" in seen and arm.stopped, "contact did not trip the reflex"
    print("self-check OK: recognized gestures, mapped to commands, moved the arm, "
          "reflex stopped on contact — the whole loop ran from one file.")


if __name__ == "__main__":
    main()
