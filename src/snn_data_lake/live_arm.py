"""
v0.47 — live real-hardware loop: ESP32 sensor -> recognize -> Interpreter -> arm sim.

Closes the loop end to end on the builder's sensor:

  (d, t) samples  ->  buffer W  ->  ORDER-AWARE gesture recognition (Paradigm B, v0.46)
                  ->  Interpreter (gesture -> robot command)  ->  arm_sim (joint / gripper motion)
  + a contact reflex that EMERGENCY_STOPs the arm the instant the object is at the sensor.

Sources (same downstream, sim-to-real):
  python live_arm.py                 # replay the builder's labeled CSVs (hardware-free) + self-check
  python live_arm.py --serial COM8   # live off the ESP32 (needs `pip install pyserial`)
  python live_arm.py --csv FILE      # replay one capture

Gesture -> command map (sensor domain): approach = grab the incoming object, retreat = release.
"""
import os
import sys

import time

from snn_data_lake import signal_loop as S
from snn_data_lake.gesture_recognition import classify, load_csv, windows, W
from snn_data_lake.interpreter import Interpreter
from snn_data_lake.arm_sim import ArmSim
from snn_data_lake.arm_config import CONTACT, STRIDE          # reflex distance + window stride: calibration in arm_config

SENSOR_COMMANDS = {"HAND_APPROACH": "GRIPPER_CLOSE",   # grab the approaching object
                   "HAND_RETREAT":  "GRIPPER_OPEN",    # release as it leaves
                   "IDLE":          "HOLD"}


def serial_samples(port, baud=115200, reconnect=True):
    """(distance, ir) from the ESP32; strips banner / non-numeric lines. Real hardware
    drops the link — reconnect with backoff instead of crashing the arm loop."""
    backoff = 0.5
    while True:
        try:
            for line in S.serial_lines(port, baud):
                v = S.parse_line(line)
                if v and len(v) >= 2:
                    yield v[0], v[1]
            return                                # generator ended cleanly (e.g. EOF)
        except Exception as e:                    # SerialException, decode, device unplug
            if not reconnect:
                raise
            sys.stderr.write(f"serial {port} dropped ({e}); reconnecting in {backoff:.1f}s\n")
            time.sleep(backoff)
            backoff = min(backoff * 2, 5.0)       # cap backoff at 5s


def _serial_samples_from(ser):
    """(d, ir) read from an ALREADY-OPEN serial handle — used when sensor-out and command-in
    share ONE UART (live_arm --serial COMx --actuate COMx opens the port once, reads + writes
    on the same handle). readline() returns b'' on timeout; keep waiting for the next line."""
    while True:
        raw = ser.readline()
        if not raw:
            continue
        v = S.parse_line(raw.decode("utf-8", "ignore"))
        if v and len(v) >= 2:
            yield v[0], v[1]


def replay_samples(paths):
    for p in paths:
        for d, t in load_csv(p):
            yield d, t


def run(samples, arm, interp, emit, stride=STRIDE):
    """Stream (d,ir) -> windowed recognition -> command -> arm. Returns action stats.
    stride < W slides the window (acts every `stride` samples) for lower live latency;
    stride == W (default) is non-overlapping, identical to before."""
    buf = []
    stats = {"windows": 0, "GRIPPER_CLOSE": 0, "GRIPPER_OPEN": 0, "HOLD": 0, "reflex": 0}
    for d, t in samples:
        if d < CONTACT:                                   # reflex: instinct preempts cognition
            cmd, _ = interp.interpret({"reflex": "STOP"})
            arm.apply(cmd)
            stats["reflex"] += 1
            buf = []
            emit({"reflex": "STOP", "command": cmd, "ee": arm.state()["ee"]})
            continue
        buf.append((d, t))
        if len(buf) == W:
            gesture = classify(buf)
            cmd, _ = interp.interpret({"match": gesture})
            arm.apply(cmd)
            stats["windows"] += 1
            stats[cmd] = stats.get(cmd, 0) + 1
            emit({"gesture": gesture, "command": cmd, "gripper": arm.gripper,
                  "ee": arm.state()["ee"]})
            buf = buf[stride:]                             # slide (stride==W -> full reset)
    return stats


def main():
    args = sys.argv[1:]
    interp = Interpreter(commands=SENSOR_COMMANDS)

    def emit(d):
        print(__import__("json").dumps(d)); sys.stdout.flush()

    stride = int(args[args.index("--stride") + 1]) if "--stride" in args else STRIDE
    actuate = args[args.index("--actuate") + 1] if "--actuate" in args else None

    if "--serial" in args:
        port = args[args.index("--serial") + 1]
        baud = int(args[args.index("--baud") + 1]) if "--baud" in args else 115200
        if actuate:
            from snn_data_lake import serial_arm
            serial_arm.assert_calibrated()                # refuse real servos w/ placeholder limits
            if actuate == port:                           # one UART: share ONE handle (read + write)
                import serial
                ser = serial.Serial(port, baud, timeout=1)
                arm = serial_arm.SerialArm(writer=ser)
                samples = _serial_samples_from(ser)
            else:                                         # command on a separate port
                arm = serial_arm.SerialArm(actuate, baud)
                samples = serial_samples(port, baud)
        else:
            arm = ArmSim()
            samples = serial_samples(port, baud)
        sys.stderr.write(f"live: sensor={port}@{baud} actuate={actuate or 'sim'} stride={stride}\n")
        run(samples, arm, interp, emit, stride)           # until interrupted
        return

    arm = ArmSim()                                        # stdlib backend (pybullet if installed)
    if "--csv" in args:
        run(replay_samples([args[args.index("--csv") + 1]]), arm, interp, emit, stride)
        return

    # default: replay the builder's labeled captures (hardware-free) + self-check
    gestures = {g: os.path.join("robot arm", f"{g}.csv") for g in SENSOR_COMMANDS}
    have_real = all(os.path.exists(p) for p in gestures.values())
    quiet = lambda d: None

    if have_real:
        stats_a = run(replay_samples([gestures["HAND_APPROACH"]]), ArmSim(), interp, quiet)
        stats_r = run(replay_samples([gestures["HAND_RETREAT"]]), ArmSim(), interp, quiet)
        a_arm = ArmSim(); run(replay_samples([gestures["HAND_APPROACH"]]), a_arm, interp, quiet)
        source = "real capture (robot arm/*.csv)"
    else:                                                 # CI without data: synth streams
        appr = [(0.9 - 0.11 * i, 295.0) for i in range(W)] * 3 + [(0.02, 298.0)]
        retr = [(0.1 + 0.11 * i, 295.0) for i in range(W)] * 3
        stats_a = run(iter(appr), ArmSim(), interp, quiet)
        stats_r = run(iter(retr), ArmSim(), interp, quiet)
        a_arm = ArmSim(); run(iter(appr), a_arm, interp, quiet)
        source = "synthetic fallback"

    print("=" * 62)
    print("LIVE ARM LOOP — sensor -> recognize -> Interpreter -> arm sim")
    print("=" * 62)
    print(f"source        : {source}")
    print(f"APPROACH file : {stats_a.get('GRIPPER_CLOSE',0)} GRIPPER_CLOSE, "
          f"{stats_a.get('reflex',0)} reflex-STOP (object reached the sensor)")
    print(f"RETREAT file  : {stats_r.get('GRIPPER_OPEN',0)} GRIPPER_OPEN")
    print(f"arm after approach run: gripper={a_arm.gripper} ee={a_arm.state()['ee']} "
          f"stopped={a_arm.stopped}")
    print("=" * 62)

    # ---- self-checks --------------------------------------------------------
    assert stats_a.get("GRIPPER_CLOSE", 0) >= 1, "approach did not drive GRIPPER_CLOSE"
    assert stats_r.get("GRIPPER_OPEN", 0) >= 1, "retreat did not drive GRIPPER_OPEN"
    assert stats_a.get("reflex", 0) >= 1, "contact never tripped the reflex STOP"

    # duplex smoke test: sensor-read + command-write share ONE serial handle (no hardware).
    from snn_data_lake.serial_arm import SerialArm
    lines = ([f"{0.9 - 0.11 * i}, 295.0\n".encode() for i in range(W)] * 3) + [b"0.02, 298.0\n"]

    class _FakeDuplex:
        def __init__(self, ls): self._it = iter(ls); self.sent = []
        def readline(self): return next(self._it, b"")       # b"" at end of the canned stream
        def write(self, b): self.sent.append(b.decode("ascii").strip())
        def flush(self): pass
        def close(self): pass

    fd = _FakeDuplex(lines)
    darm = SerialArm(writer=fd)                              # SerialArm as the arm, writing to fd

    def _bounded():                                          # _serial_samples_from, but stops at EOF
        while True:
            raw = fd.readline()
            if not raw:
                return
            v = S.parse_line(raw.decode("utf-8", "ignore"))
            if v and len(v) >= 2:
                yield v[0], v[1]

    run(_bounded(), darm, interp, quiet)                     # read + recognize + write on one handle
    assert "GRIPPER_CLOSE" in fd.sent, f"duplex: approach did not send GRIPPER_CLOSE ({fd.sent})"
    assert "EMERGENCY_STOP" in fd.sent, "duplex: contact did not send EMERGENCY_STOP over serial"

    print("self-check OK: approach->GRIPPER_CLOSE, retreat->GRIPPER_OPEN, contact->reflex STOP, "
          "one-UART duplex read+write — sensor recognized, Interpreter mapped, arm moved (loop closed)")


if __name__ == "__main__":
    main()
