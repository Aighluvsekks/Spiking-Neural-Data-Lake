// command_arm.ino — ESP32 command-IN reference sketch (Phase 2 actuator path).
//
// Pair sketch to the sensor streamer. This one READS one command per line from serial
// (sent by live_arm.py --actuate COMx via serial_arm.SerialArm) and drives the servos.
// Wire protocol: docs/arduino_contract.md ("Command channel") — ASCII, '\n'-terminated,
// 115200 8N1, one command per line.
//
// SAFETY: EMERGENCY_STOP is handled FIRST and detaches the servos (fail-safe). Replace the
// GRIP_* / joint angle constants + clamps with your rig's REAL limits before powering servos.
// Bench-test in ECHO mode first (see ECHO_ONLY) — confirm the right command arrives with the
// servos UNPOWERED, then wire actuation one joint at a time.

#include <ESP32Servo.h>

Servo gripper;
Servo jointA;
Servo jointB;

const int GRIP_OPEN   = 20;      // deg — REPLACE with your gripper's real open/closed angles
const int GRIP_CLOSED = 80;
const int JOINT_MIN   = 10;      // deg — REPLACE with your arm's real per-joint safe limits
const int JOINT_MAX   = 170;
const int HOME_A = 90, HOME_B = 90;

const bool ECHO_ONLY = true;     // true = print the received command, DO NOT move (bench test)

int clampDeg(int v) { return v < JOINT_MIN ? JOINT_MIN : (v > JOINT_MAX ? JOINT_MAX : v); }

// parse "JOINT_A_ROTATE(+15deg)" -> +15
int parseDeg(const String& c) {
  int lp = c.indexOf('('), rp = c.indexOf("deg");
  if (lp < 0 || rp < 0) return 0;
  return c.substring(lp + 1, rp).toInt();
}

void setup() {
  Serial.begin(115200);
  gripper.attach(13);            // REPLACE with your servo pins
  jointA.attach(12);
  jointB.attach(14);
  gripper.write(GRIP_OPEN);
  jointA.write(HOME_A);
  jointB.write(HOME_B);
}

void loop() {
  // (Your sensor streamer prints the sensor line here — sensor OUT + command IN share the UART.)

  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if (cmd.length() == 0) return;

  if (ECHO_ONLY) { Serial.print("ACK "); Serial.println(cmd); return; }

  if (cmd == "EMERGENCY_STOP") {          // fail-safe FIRST — cut motion immediately
    gripper.detach(); jointA.detach(); jointB.detach();
  } else if (cmd == "GRIPPER_CLOSE") {
    gripper.write(GRIP_CLOSED);
  } else if (cmd == "GRIPPER_OPEN") {
    gripper.write(GRIP_OPEN);
  } else if (cmd == "HOME" || cmd == "RETRACT_ALL") {
    jointA.write(HOME_A); jointB.write(HOME_B); gripper.write(GRIP_OPEN);
  } else if (cmd.startsWith("JOINT_A_ROTATE")) {
    jointA.write(clampDeg(jointA.read() + parseDeg(cmd)));
  } else if (cmd.startsWith("JOINT_B_ROTATE")) {
    jointB.write(clampDeg(jointB.read() + parseDeg(cmd)));
  }
  // HOLD (and anything unknown) -> do nothing
}
