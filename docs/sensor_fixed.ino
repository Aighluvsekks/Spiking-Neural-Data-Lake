// sensor.ino — corrected for the signal-loop wire contract.
// Changes from the original:
//   1. idle prints a NUMBER (-1), not "___"  -> idle lines now parse instead of being dropped
//   2. no banner text on the data port        -> startup "ultrasonic, temp" line removed
//   3. (kept) 115200 baud, comma-separated, newline-terminated, 10 Hz
// Output per line:  "<distance_m>, <temp_C>"   (temp = -1 when nothing within 3 m = idle)

#include <Wire.h>
#include <Adafruit_MLX90614.h>

const int C = 2;
int TRIG_PIN = 12;
int ECHO_PIN = 11;

int THERMO_SCL = 21;
int THERMO_SDA = 22;

Adafruit_MLX90614 mlx = Adafruit_MLX90614();

void setup() {
  Serial.begin(115200);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // If the temp sensor is missing, hold — but do NOT spam the data port with text.
  if (!mlx.begin()) {
    while (1) { delay(1000); }   // (debug on a separate channel if you need one)
  }
  // no banner line: the consumer expects only numeric "d, t" lines
}

float getDistanceM() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH);

  float distanceM = (duration * 0.0343) / 200.0;

  return distanceM;
}

void loop() {
  float d = getDistanceM();          // object's distance from sensor (metres)
  Serial.print(d);
  Serial.print(", ");
  if (d <= 3.00) {
    Serial.print(mlx.readObjectTempC());   // object temperature (C)
  } else {
    Serial.print(-1);                       // idle: nothing in range (was "___" — unparseable)
  }
  Serial.println();
  delay(100);                          // ~10 Hz
}
