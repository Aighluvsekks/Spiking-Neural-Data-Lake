# Arduino ↔ Signal-Loop Wire Contract

The contract between the **Arduino** box (signal source) and the **spike encoding →
data lake → match** part (`signal_loop.py`). Anything that honors this contract works;
`signal_loop.py` is agnostic to what produces the bytes.

## Transport
- **USB serial**, `115200` baud, `8N1` (8 data bits, no parity, 1 stop bit).
- ASCII, **newline-terminated** lines (`\n`; trailing `\r` tolerated).
- Override baud: `--serial COM3 --baud 230400`.

## Line format
One line = comma- **or** space-separated floating-point numbers.
```
0.12, 0.04, 0.98, 0.51, ...
```
- Values may be **any range** — `signal_loop` min-max normalizes each window to `[0,1]`.
- Lines starting with `#` are comments (ignored). Unparseable lines are skipped, never crash the loop.
- `N = 64` features per window (the spike encoder's input width).

## Two modes

### Direct (default, `--window 1`)
Each line **is** one window: send exactly `N=64` floats per line. The Arduino does the
windowing/feature extraction. Lines with ≠64 values are truncated/zero-padded to 64.

```
python signal_loop.py --serial COM3
```

### Windowed (`--window W`)
Each line is `C` **raw channel samples** for one timestep. The loop buffers `W` lines,
flattens them row-major (`C×W`), and resizes to `N=64` → one window. Use when the Arduino
streams raw sensors and you want the loop to do the windowing.

```
python signal_loop.py --serial COM3 --window 8     # 8 timesteps × C channels -> 1 window
```
`C×W` should be ≈ 64 (e.g. 8 channels × 8 steps). Larger → truncated, smaller → zero-padded.

## What comes back
`signal_loop` does **not** write to the Arduino. It emits one JSON line on **stdout** per
window for the Interpreter to consume:
```json
{"t": 42, "match": "GRIPPER_CLOSE", "dist": 2.99, "confident": true}
```
`match` is `null` + `confident:false` when the signal is novel/ambiguous (no command fired).

## Feedback channel (Interpreter → loop, `--feedback`)
With `--feedback` the loop closes: after the Interpreter executes a command and observes the
result, it sends an **outcome** back into the loop's input stream (same channel as signals):
```
OUTCOME 1.0      # that action was good   (reward in [-1, +1], applied to the last acted signal)
OUTCOME -1.0     # that action was bad
```
Outcomes drive **dopamine** learning (reward-prediction-error) and a **cortisol** stress
level; cortisol then modulates the reflex threshold and the matcher's caution bias live. The
loop emits the neuromodulator state per event:
```json
{"outcome": -1.0, "dopamine": -1.0, "stress": 0.3}
{"t": 51, "match": "HOME", "instinct": "AVOID", "valence": -0.42, "stress": 0.28}
```
Mix `OUTCOME` lines and signal lines on the same stream. (Serial reads outcomes the same way;
the Interpreter or a small bridge injects them onto the loop's input.)

## Command channel (loop → Arduino, the Phase-2 actuator path)
The sensor stream above is Arduino → loop. The **actuator path is the reverse**: the loop sends
one **command per line** back to the ESP32, which drives the servos (`deploy/command_arm.ino`).
Same transport — ASCII, `\n`-terminated, `115200 8N1`, one command per line. Sensor-out and
command-in share the one UART.

Command vocabulary (exactly what `interpreter.py` emits):
```
GRIPPER_CLOSE
GRIPPER_OPEN
HOLD                      # no-op (hold pose)
EMERGENCY_STOP            # fail-safe — the sketch detaches servos FIRST
JOINT_A_ROTATE(+15deg)    # signed integer degrees, relative
JOINT_B_ROTATE(-10deg)
HOME
RETRACT_ALL
```
Driven by `python live_arm.py --serial COMx --actuate COMx` (`serial_arm.SerialArm`). SerialArm
refuses to open a real port until `arm_config.SAFETY_CALIBRATED = True` — set that only after the
placeholder limits are replaced with your rig's real values. Bench-test with the sketch's
`ECHO_ONLY = true` first (servos unpowered) to confirm the right command arrives.

## Throughput
One matched command per window. Window rate = (Arduino line rate) ÷ `W`. Match latency is
dominated by the Van Rossum gate + (hybrid) the spiking forward pass — sub-millisecond per
window at `N=64`, so the Arduino's `delay()` sets the real cadence.

## Test without hardware
Same contract over stdin — pipe a CSV of windows:
```
cat windows.csv | python signal_loop.py --stdin            # hybrid
cat windows.csv | python signal_loop.py --stdin --fast     # template baseline
```

## Example Arduino sketch — direct mode (64 features/line)
```cpp
// Emits one 64-value feature window per line @115200. Replace readFeature()
// with your real feature extraction (joint encoders, current, IMU, FFT bins, ...).
const int N = 64;
void setup() { Serial.begin(115200); }
void loop() {
  for (int i = 0; i < N; i++) {
    Serial.print(readFeature(i), 4);     // a float feature
    if (i < N - 1) Serial.print(',');
  }
  Serial.println();                       // newline = end of window
  delay(50);                              // ~20 windows/s
}
float readFeature(int i) { /* your sensor/feature code */ return analogRead(A0) / 1023.0; }
```

## Example Arduino sketch — windowed mode (C raw channels/line)
```cpp
// Streams C raw channel samples per timestep; the loop buffers W of these
// (run: python signal_loop.py --serial COM3 --window 8).
const int C = 8;                          // channels (e.g. 6 joints + grip + current)
void setup() { Serial.begin(115200); }
void loop() {
  for (int c = 0; c < C; c++) {
    Serial.print(analogRead(A0 + c));     // raw int is fine — auto-normalized
    if (c < C - 1) Serial.print(',');
  }
  Serial.println();
  delay(6);                               // sample rate; W lines -> one window
}
```

## Enrolling references
Each command needs one reference signature in `data/signatures.json`. On hardware:
hold the arm in the gesture, then `python signal_loop.py --enroll GRIPPER_CLOSE` captures
a window. Labels in `signatures.json` are exactly what `match` emits to the Interpreter.
