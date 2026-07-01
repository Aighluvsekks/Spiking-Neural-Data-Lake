"""
v0.60 — real-arm actuator sink. The Phase-2 command-OUT path.

live_arm.py already does: sensor -> recognize -> Interpreter -> command -> arm.apply(cmd).
Until now `arm` was ArmSim (a simulation). SerialArm is a DROP-IN with the same surface
live_arm uses off ArmSim — .apply(cmd), .state(), .gripper — but each command line also goes
out over serial to the ESP32 servo sketch (sketches/command_arm.ino), moving the REAL arm.
See docs/arduino_contract.md ("Command channel").

SAFETY: driving real servos with the PLACEHOLDER limits in arm_config (CONTACT, joint limits,
reflex rules) is unsafe. SerialArm refuses to open a real port until arm_config.SAFETY_CALIBRATED
is True — flip that only after you replace them with your rig's measured values
(docs/arduino_requirements.md #3). For bench/echo tests without hardware, inject a fake `writer=`.

Fail-safe: any serial write error raises, so the caller stops the loop rather than driving blind.
pyserial is imported lazily (only when opening a real port) so this module + its self-check stay
pure-stdlib and run in CI.

  python serial_arm.py        # self-check (fake serial, no hardware, no pyserial)
"""
from arm_sim import ArmSim
import arm_config


class SerialArm:
    """Command sink: mirrors ArmSim (for state()/gripper logging) AND writes each command line
    to the ESP32. writer = an object with .write(bytes)/.flush()/.close() — a real serial.Serial
    on hardware, or a fake for bench tests. writer=None opens a real pyserial port."""

    def __init__(self, port=None, baud=115200, writer=None, allow_uncalibrated=False):
        if writer is None:
            if not (arm_config.SAFETY_CALIBRATED or allow_uncalibrated):
                raise RuntimeError(
                    "SerialArm refuses to drive real servos with placeholder safety limits. "
                    "Set arm_config.SAFETY_CALIBRATED=True after replacing CONTACT/joint limits "
                    "with your rig's real values (docs/arduino_requirements.md #3), or pass a "
                    "fake writer= for a bench/echo test.")
            import serial                        # pyserial — only needed on real hardware
            writer = serial.Serial(port, baud, timeout=1)
        self.w = writer
        self.mirror = ArmSim()                   # local twin so state()/gripper still work
        self.mirror.apply("HOME")

    def apply(self, command):
        """Send the command to the real arm (flushed, so EMERGENCY_STOP is not buffered), then
        mirror it. A serial failure raises -> the caller must stop (fail-safe, not drive blind)."""
        try:
            self.w.write((command + "\n").encode("ascii"))
            self.w.flush()
        except Exception as e:
            raise RuntimeError(f"actuator serial write failed on '{command}': {e}") from e
        return self.mirror.apply(command)

    def state(self):
        return self.mirror.state()

    @property
    def gripper(self):
        return self.mirror.gripper

    def close(self):
        try:
            self.w.close()
        except Exception:
            pass


def main():
    # bench self-check: a fake writer records the lines. No hardware, no pyserial.
    class FakeSerial:
        def __init__(self):
            self.lines = []
            self.fail = False

        def write(self, b):
            if self.fail:
                raise IOError("cable yanked")
            self.lines.append(b.decode("ascii").strip())

        def flush(self):
            pass

        def close(self):
            pass

    fake = FakeSerial()
    arm = SerialArm(writer=fake)                 # fake writer -> calibration gate not exercised

    arm.apply("GRIPPER_CLOSE")
    arm.apply("JOINT_A_ROTATE(+15deg)")
    arm.apply("EMERGENCY_STOP")

    print("=" * 58)
    print("SERIAL ARM — command-out sink (fake serial, no hardware)")
    print("=" * 58)
    print(f"lines sent     : {fake.lines}")
    print(f"mirror gripper : {arm.gripper}   stopped: {arm.state()['stopped']}")

    # calibration gate: opening a REAL port while uncalibrated must be blocked
    try:
        SerialArm(port="COM_TEST")               # writer=None -> would open a real port
        gate_blocked = False
    except RuntimeError:
        gate_blocked = True

    # fail-safe: a serial error must raise, not be swallowed
    fake.fail = True
    try:
        arm.apply("GRIPPER_OPEN")
        failsafe = False
    except RuntimeError:
        failsafe = True

    print(f"calib gate     : real-port open blocked = {gate_blocked}")
    print(f"fail-safe      : serial error raises     = {failsafe}")
    print("=" * 58)

    # ---- self-checks --------------------------------------------------------
    assert fake.lines == ["GRIPPER_CLOSE", "JOINT_A_ROTATE(+15deg)", "EMERGENCY_STOP"], \
        f"commands not sent verbatim: {fake.lines}"
    assert arm.gripper == 1.0, "mirror gripper did not close"
    assert arm.state()["stopped"], "EMERGENCY_STOP did not freeze the mirror"
    assert gate_blocked or arm_config.SAFETY_CALIBRATED, \
        "uncalibrated real-port open must be blocked"
    assert failsafe, "serial write failure must raise (fail-safe)"
    print("self-check OK: commands sent verbatim, mirror synced, calibration gate + fail-safe active")


if __name__ == "__main__":
    main()
