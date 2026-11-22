// Battery-days model on the ESP32-S3 (Arduino IDE; board "ESP32S3 Dev Module", OPI PSRAM not needed).
// No library to install: battery_model.h (same folder) holds the weights as const arrays in flash.
//
// This sketch is NOT yet run on a board by the authors of this repo. Run it, and replace the
// estimate in MISSION4.md with YOUR measured microseconds (Class 15: "measure the real answer").
#include "battery_model.h"

void setup() {
  Serial.begin(115200);
  delay(1000);

  volatile float in[BATTERY_N_FEATURES] = {3.86f, 28.0f, 5.0f, -66.0f, 22.0f};  // volatile: stop the compiler hoisting the call
  const int REPS = 2000;
  float r[BATTERY_N_FEATURES], sink = 0;

  for (int mode = 0; mode < 2; mode++) {
    uint32_t t = micros();
    for (int k = 0; k < REPS; k++) {
      for (int i = 0; i < BATTERY_N_FEATURES; i++) r[i] = in[i];
      sink += battery_days(r, mode);
    }
    uint32_t us = micros() - t;
    for (int i = 0; i < BATTERY_N_FEATURES; i++) r[i] = in[i];
    Serial.printf("%s: %.1f days left, %.2f us per prediction\n", mode ? "int8   " : "float32",
                  battery_days(r, mode), (float)us / REPS);
  }
  // A glitch must be refused, not guessed (0 V = dead ADC):
  r[0] = 0.0f;
  Serial.printf("battery_v = 0 V -> %.1f (-1 means refused)\n", battery_days(r, 0));
  if (sink < 0) Serial.println();  // keep 'sink' alive
}

void loop() {}
