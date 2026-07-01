# Phase 2 — Go Live: Builder Brief

> For the embedded/Arduino builder (MakiKuri00). Self-contained — you don't need the rest
> of the system. Supersedes the original 8-channel brief; this matches your **real** 2-channel
> ultrasonic + IR hardware.

## Where we are
**Phase 1 (clean, labeled data) is done — thank you.** Your labeled per-gesture captures
(`HAND_APPROACH`, `HAND_RETREAT`, `IDLE`) drove **95% live recognition**. Everything downstream
is built and tested against your CSVs on replay:

```
sensor (distance, IR) → recognize gesture → command → arm + a contact reflex that E-STOPs
```

Phase 2 is turning that from **replay into a real, live loop on your hardware**.

## Your hardware (confirmed — correct anything wrong)
- ESP32, USB serial @ **115200**, ~**10 Hz**, `", "` separator.
- **Sensor_1 = ultrasonic distance (metres)**, `-1` when nothing in range.
- **Sensor_2 = IR (MLX90614), Kelvin** (~295 at room temp).

---

## What we need from you (6 items, blocking first)

### 1. A live run  *(blocking)*
Everything works on replay; we need one real real-time run. With the ESP32 + arm connected:
```bash
python live_arm.py --serial COM8      # your port
```
**Deliver:** confirmation it runs end-to-end, plus the console log (or a short video). If it
errors, send the error — that's just as useful.

### 2. Actuator interface — how a command reaches the real arm  *(blocking)*
Right now our loop emits commands (`GRIPPER_CLOSE`, `GRIPPER_OPEN`, `JOINT_A_ROTATE(+15deg)`,
`EMERGENCY_STOP`) and drives a **simulated** arm. To move the **real** arm we need the actuator
path. Tell us **how your servos are driven**, then either:
- accept our commands **back over serial** to the ESP32 (a second sketch that reads a command
  line and moves the servo), or
- point us at your servo controller and we send commands to it.

**Deliver:** the command→servo path + a sketch (or driver) that accepts one command per line and
actuates. We'll agree the exact command strings with you.

### 3. Real safety limits — replace our placeholders  *(safety-critical, before #2 drives anything)*
Our danger thresholds are **guesses**. Give the real numbers for your rig:

| Limit | Ours (placeholder) | We need your real value |
|---|---|---|
| Contact distance → E-STOP | `CONTACT = 0.04 m` | at what distance is a collision imminent? |
| Joint limits / reach | ±0.4 rad (sim) | your arm's real min/max per joint |
| Over-current / force danger | `reflex.py` ch6/ch7 rules = placeholder | any reading that means "stop now" |

**Deliver:** short written values. These wire straight into `arm_config.py` + `reflex.py`.

### 4. IR channel calibration  *(upgrade)*
Sensor_2 (IR) is captured but currently **ignored** — we don't know its polarity/scale, so IR
fusion is off (`arm_config.IR_PRESENT = None`). Record a short capture:
- ~20 s with a **hand held in range**, then ~20 s with **nothing in range**.

**Deliver:** one CSV of that. We'll set the presence threshold and use both sensors (better idle
vs far-object separation).

### 5. Reward / success signal  *(upgrade — enables real learning)*
The loop can **learn** from outcomes (dopamine/cortisol), but only with a real "did it work?"
signal. Today it's a placeholder (`--assume-success`). If the arm has a **gripper-force or
object-present sensor**, stream it as a 3rd channel or a success flag.

**Deliver (optional):** a success indicator per grasp. Without it, learning stays simulated.

### 6. Confirm + extend  *(quick)*
- Confirm **Sensor_2 is intentionally Kelvin** (fine — we normalize; just confirming).
- If the arm needs **more actions** than APPROACH / RETREAT / IDLE, record those the same way
  (one labeled CSV per gesture, ~30–60 s each — same as Phase 1).

---

## What we handle (you don't)
Recognition, encoding, the data lake, the reflex/learning logic, the arm simulation, and the
command mapping. Your Phase 2 surface is only: **run the live loop (#1), give commands a way to
the servos (#2), and hand us the real safety numbers (#3)**. #4–#6 are upgrades.

## Acceptance (how we'll know Phase 2 works)
- `live_arm.py --serial COM8` runs live: your gesture in front of the sensor → correct command →
  the **real** arm moves, and a too-close reading triggers `EMERGENCY_STOP` before contact.
- The E-STOP distance and joint limits are **your** real numbers, not our placeholders.

## Wire protocol
Sensor-out format is unchanged and already correct — see [`arduino_contract.md`](arduino_contract.md).
The new piece is the **command-in** channel (#2); we'll pin its format with you before you build it.
