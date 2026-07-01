# Hardware Setup — Live Robot Arm (Phase 2)

End-to-end guide: bare ESP32 + sensors + arm → the live loop moving a **real** arm. Read
[`arduino_contract.md`](arduino_contract.md) (wire protocol) and [`arduino_requirements.md`](arduino_requirements.md)
(what's blocking) alongside this.

```
  [ ultrasonic + IR ] --sensor line--> ESP32 --USB serial (one UART)--> PC: live_arm.py
                                          ^                                   |
  [ gripper + 2 joints ] <--PWM--  servos |  <----------command line---------- (recognize -> Interpreter)
```
One USB cable carries **both** directions: sensor lines out, command lines in. `live_arm` opens
the port **once** and reads + writes on the same handle.

---

## 1. Bill of materials
- **ESP32** dev board (USB).
- **Ultrasonic distance** sensor (HC-SR04 or equiv) → `Sensor_1`, metres.
- **IR temperature** sensor (MLX90614, I2C) → `Sensor_2`.
- **3 servos**: gripper + joint A + joint B (hobby servos, e.g. SG90/MG996R).
- **External 5–6 V servo supply** (do **not** power servos off the ESP32/USB — current spikes
  brown-out the board). Tie the supply **ground to the ESP32 ground** (common ground).
- Jumper wires, breadboard, USB cable.

## 2. Wiring
Sensor pins match the shipped sketch ([`sensor_fixed.ino`](sensor_fixed.ino)); servo pins are new
and chosen to **not collide** with the sensor pins.

| Signal | ESP32 pin | Notes |
|---|---|---|
| Ultrasonic TRIG | GPIO 12 | |
| Ultrasonic ECHO | GPIO 11 | 5 V echo → use a divider to 3.3 V if your module is 5 V |
| MLX90614 SDA | GPIO 21 | I2C |
| MLX90614 SCL | GPIO 22 | I2C |
| Gripper servo | GPIO 13 | PWM |
| Joint A servo | GPIO 25 | PWM (NOT 12 — that's TRIG) |
| Joint B servo | GPIO 26 | PWM |
| Servo V+ | external 5–6 V | **not** ESP32 5V |
| Servo GND + ESP32 GND | common | shared ground is required |

## 3. Firmware
1. **Arduino IDE** → add ESP32 board support (Boards Manager → "esp32"). Install libraries:
   `Adafruit MLX90614` and `ESP32Servo` (Library Manager).
2. Flash the **merged sketch** below (one sketch does sensor-out **and** command-in on the one
   UART). It combines `sensor_fixed.ino` + [`sketches/command_arm.ino`](../sketches/command_arm.ino).
3. Keep `ECHO_ONLY = true` for first power-on (servos won't move; the board just ACKs commands).

```cpp
// arm_firmware.ino — ONE ESP32 sketch: sensor lines OUT + command lines IN (shared UART).
#include <Wire.h>
#include <Adafruit_MLX90614.h>
#include <ESP32Servo.h>

const int TRIG_PIN = 12, ECHO_PIN = 11;          // ultrasonic
Adafruit_MLX90614 mlx = Adafruit_MLX90614();     // IR on I2C (SDA 21, SCL 22)

Servo gripper, jointA, jointB;
const int PIN_GRIP = 13, PIN_A = 25, PIN_B = 26; // servo pins (clear of sensor pins)

// ---- REAL limits: REPLACE with your rig's measured values (arduino_requirements.md #3) ----
const int GRIP_OPEN = 20, GRIP_CLOSED = 80;
const int JOINT_MIN = 10, JOINT_MAX = 170, HOME_A = 90, HOME_B = 90;
const bool ECHO_ONLY = true;                      // true = ACK only, DON'T move (bench test)

int clampDeg(int v){ return v<JOINT_MIN?JOINT_MIN:(v>JOINT_MAX?JOINT_MAX:v); }
int parseDeg(const String& c){ int l=c.indexOf('('),r=c.indexOf("deg"); return (l<0||r<0)?0:c.substring(l+1,r).toInt(); }

void setup(){
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT); pinMode(ECHO_PIN, INPUT);
  if(!mlx.begin()){ while(1) delay(1000); }       // no banner on the data port
  gripper.attach(PIN_GRIP); jointA.attach(PIN_A); jointB.attach(PIN_B);
  gripper.write(GRIP_OPEN); jointA.write(HOME_A); jointB.write(HOME_B);
}

float getDistanceM(){
  digitalWrite(TRIG_PIN,LOW); delayMicroseconds(2);
  digitalWrite(TRIG_PIN,HIGH); delayMicroseconds(10); digitalWrite(TRIG_PIN,LOW);
  return (pulseIn(ECHO_PIN,HIGH) * 0.0343) / 200.0;
}

void handleCommand(){
  if(!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n'); cmd.trim();
  if(cmd.length()==0) return;
  if(ECHO_ONLY){ Serial.print("ACK "); Serial.println(cmd); return; }
  if(cmd=="EMERGENCY_STOP"){ gripper.detach(); jointA.detach(); jointB.detach(); }  // fail-safe first
  else if(cmd=="GRIPPER_CLOSE") gripper.write(GRIP_CLOSED);
  else if(cmd=="GRIPPER_OPEN")  gripper.write(GRIP_OPEN);
  else if(cmd=="HOME"||cmd=="RETRACT_ALL"){ jointA.write(HOME_A); jointB.write(HOME_B); gripper.write(GRIP_OPEN); }
  else if(cmd.startsWith("JOINT_A_ROTATE")) jointA.write(clampDeg(jointA.read()+parseDeg(cmd)));
  else if(cmd.startsWith("JOINT_B_ROTATE")) jointB.write(clampDeg(jointB.read()+parseDeg(cmd)));
  // HOLD / unknown -> nothing
}

void loop(){
  float d = getDistanceM();
  Serial.print(d); Serial.print(", ");
  Serial.println(d <= 3.00 ? mlx.readObjectTempC() : -1);   // Sensor_2 (-1 = idle)
  handleCommand();
  delay(100);                                                // ~10 Hz
}
```

## 4. Software (PC side)
```powershell
git clone https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake.git
cd Spiking-Neural-Data-Lake
python demo.py                       # sanity: zero-dep, must print the loop (Python 3.11+)
python -m pip install pyserial       # serial dep for the live path
[System.IO.Ports.SerialPort]::getportnames()   # find your ESP32 port, e.g. COM8
```

## 5. Calibration & safety  *(do BEFORE any servo moves)*
The sim ships **placeholder** limits. Replace them with your rig's real values, then arm the gate:
- **`arm_config.py`** — `CONTACT` (E-STOP distance, m), distance bins, `TRAJ_DEV_MAX`.
- **Sketch** — `GRIP_OPEN/CLOSED`, `JOINT_MIN/MAX`, `HOME_A/B`, servo pins.
- Only after those are real: set **`arm_config.SAFETY_CALIBRATED = True`**.

`live_arm --actuate` **refuses to open the port** until `SAFETY_CALIBRATED` is `True` — a
deliberate gate so placeholder limits can't drive hardware.

## 6. Bring-up sequence  *(do NOT skip — each step de-risks the next)*
1. **Echo test (servos safe).** Sketch `ECHO_ONLY = true`. Open the Arduino **Serial Monitor**
   (115200), type `GRIPPER_CLOSE` → see `ACK GRIPPER_CLOSE`, and confirm `d, t` lines stream. No
   Python, no gate needed.
2. **Sensor stream sanity.** `python live_arm.py --csv captures\IDLE.csv` after a recording (§7),
   or watch the Serial Monitor — distances change as you move your hand.
3. **Set real safety numbers** (§5), flip `SAFETY_CALIBRATED = True`.
4. **One joint.** Sketch `ECHO_ONLY = false`; wire only Joint A. `live_arm --serial COMx --actuate
   COMx`, do the gesture, confirm the joint moves within its clamps.
5. **Gripper**, then Joint B.
6. **E-STOP hardware test (acceptance).** Move a hand under `CONTACT` → the arm must halt
   **before** contact (`EMERGENCY_STOP` detaches the servos).

## 7. Record data + run live
Record labeled captures and run the whole thing with the helper:
```powershell
.\record_and_run.ps1 -Port COM8        # records IDLE/APPROACH/RETREAT, replays, then live loop
```
Or manually:
```powershell
python "robot arm/sensor_reading.py" HAND_APPROACH COM8   # record one gesture (Ctrl-C to stop)
python live_arm.py --serial COM8                          # live, SIM arm (no actuation)
python live_arm.py --serial COM8 --actuate COM8           # live, REAL arm (one UART, gated)
```
`--serial COMx --actuate COMx` (same port) is the single-UART case — one handle, read + write.

## 8. Troubleshooting
| Symptom | Cause / fix |
|---|---|
| `could not open port COMx` | wrong port, or another program (Serial Monitor!) holds it — close it |
| `SerialArm refuses to drive real servos...` | `SAFETY_CALIBRATED` is `False` — set real limits, flip it (§5) |
| `ModuleNotFoundError: serial` | `python -m pip install pyserial` in the Python you run |
| no sensor lines / all idle | check TRIG/ECHO wiring; ECHO may need a 3.3 V divider |
| ESP32 resets when a servo moves | servos on ESP32 power — use an external 5–6 V supply, common ground |
| commands sent but arm still | sketch `ECHO_ONLY` still `true`, or servo pins wrong |
| gesture misread | recapture cleaner data (§7); the order-aware matcher needs a clear far↔near sweep |

**Status:** software path is complete and gated. The blocking hardware pieces are yours: servo
wiring + the real safety numbers (arduino_requirements.md #2, #3).
