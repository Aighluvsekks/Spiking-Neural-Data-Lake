import serial, csv, sys

label = sys.argv[1] if len(sys.argv) > 1 else "capture"   # gesture name = the file name
PORT = sys.argv[2] if len(sys.argv) > 2 else "COM8"       # 2nd arg overrides; else Arduino IDE port
out = f"{label}.csv"

ser = serial.Serial(PORT, 115200)
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["label", "Sensor_1", "Sensor_2"])         # labeled, numbers only
    print(f"Recording '{label}' -> {out}. Do the gesture repeatedly. Ctrl-C to stop.")
    try:
        while True:
            line = ser.readline().decode("utf-8", "ignore").strip()
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue                                   # skip boot banner / junk
            try:
                d, t = float(parts[0]), float(parts[1])
            except ValueError:
                continue                                   # skip any non-numeric line
            w.writerow([label, d, t])
            f.flush()
    except KeyboardInterrupt:
        print("stopped.")
        ser.close()