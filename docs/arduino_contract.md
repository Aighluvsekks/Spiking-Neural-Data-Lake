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
