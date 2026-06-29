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

  if (!mlx.begin()) {
    while (1) { delay(1000); }
  }
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
  float d = getDistanceM(); // measure distance
  Serial.print(d);
  Serial.print(", ");
  if (d <= 3.00) {
    // measure temperature (K)
    Serial.print(mlx.readObjectTempC() + 273.15);
  } else {
    // idle
    Serial.print(-1);
  }
  Serial.println();
  delay(100);
}
