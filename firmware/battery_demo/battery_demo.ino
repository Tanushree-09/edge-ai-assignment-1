/* Battery-days demo for ESP32-S3.
 *
 * Class 15: measure the real µs/prediction — replace MISSION4.md's estimate with yours.
 * Class 5:  cascade is now inside battery_days(): voltage-only rule fires first for
 *            battery_v <= 3.50 V (data-validated, MAE 0.75d); model runs above that.
 * Class 18: a bursty edge device lives or dies by its sleep current. Between hourly
 *            readings the board enters light sleep (~2 mA vs ~50 mA awake). The demo
 *            measures one reading, prints it, then sleeps until the next period.
 *
 * Board settings: ESP32S3 Dev Module. OPI PSRAM not needed.
 * No libraries to install: battery_model.h holds all weights as const flash arrays.
 */
#include "battery_model.h"
#include "esp_sleep.h"   /* light sleep + timer wakeup */

/* ── Duty-cycle parameters (Class 18: bursty workload) ─────────────────────── */
#define MEASURE_PERIOD_S   3600ULL   /* predict once per hour in production */
#define DEMO_PERIOD_S         10ULL  /* short period for the demo bench run */

/* ── Benchmark: how long does one prediction actually take? ─────────────────── */
static void benchmark() {
  /* A healthy mid-range reading — model path, not cascade path */
  volatile float in[BATTERY_N_FEATURES] = {3.86f, 28.0f, 5.0f, -66.0f, 22.0f};
  const int REPS = 2000;
  float r[BATTERY_N_FEATURES];
  volatile float sink = 0;

  Serial.println("=== Latency benchmark (Class 15: measure, don't estimate) ===");
  uint32_t t, us;
  for (int mode = 0; mode < 2; mode++) {
    t = micros();
    for (int k = 0; k < REPS; k++) {
      for (int i = 0; i < BATTERY_N_FEATURES; i++) r[i] = in[i];
      sink += battery_days(r, mode);
    }
    us = micros() - t;
    Serial.printf("  %s: %.1f days, %.2f us/prediction\n",
                  mode ? "int8   " : "float32", battery_days(r, mode), (float)us / REPS);
  }

  /* Cascade path: battery_v <= 3.50 V -> rule fires, model skipped (Class 5).
   * Rule derived from training data: 360 rows, 100% have days_left <= 8.2 days. */
  float low[BATTERY_N_FEATURES] = {3.45f, 28.0f, 5.0f, -66.0f, 22.0f};
  t = micros();
  for (int k = 0; k < REPS; k++) sink += battery_days(low, 0);
  us = micros() - t;
  Serial.printf("  cascade (3.45 V): %.1f days, %.2f us/prediction (rule, no model)\n",
                battery_days(low, 0), (float)us / REPS);

  /* Physical impossibility: 0 V = dead ADC, refused immediately (Class 5 graceful degradation) */
  float bad[BATTERY_N_FEATURES] = {0.0f, 28.0f, 5.0f, -66.0f, 22.0f};
  Serial.printf("  battery_v = 0 V -> %.1f (-1 = refused, never a guess)\n",
                battery_days(bad, 0));

  if (sink < 0) Serial.println();  /* keep sink alive so the compiler doesn't hoist loops */
}

/* ── One production measurement cycle ───────────────────────────────────────── */
static void measure_and_print() {
  /* In production: read real sensors here.
   * For the demo we use a fixed reading so the output is deterministic. */
  float reading[BATTERY_N_FEATURES] = {3.86f, 28.0f, 5.0f, -66.0f, 22.0f};

  uint32_t t = micros();
  float days = battery_days(reading, /*use_int8=*/0);
  uint32_t us = micros() - t;

  if (days < 0.0f) {
    Serial.println("SENSOR FAULT — reading refused (physically impossible)");
  } else if (days < 10.0f) {
    /* Class 5 graceful degradation: surface the urgency, not just a number */
    Serial.printf("WARNING: battery critical — %.1f days left (%.1f us)\n", days, (float)us);
  } else {
    Serial.printf("Battery: %.1f days left (%.1f us per prediction)\n", days, (float)us);
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  /* On first boot: run the full benchmark so the user can fill in MISSION4.md */
  if (esp_sleep_get_wakeup_cause() == ESP_SLEEP_WAKEUP_UNDEFINED) {
    benchmark();
    Serial.println();
    Serial.printf("Switching to duty-cycle mode: measure every %llu s, sleep between.\n",
                  DEMO_PERIOD_S);
    Serial.printf("(Set MEASURE_PERIOD_S to %llu for production — Class 18 duty cycle)\n",
                  MEASURE_PERIOD_S);
  }

  measure_and_print();

  /* Class 18: sleep until next measurement period.
   * Light sleep keeps RAM alive (model weights stay in flash; activations gone anyway).
   * Deep sleep saves more power but loses Serial; use it in production. */
  esp_sleep_enable_timer_wakeup(DEMO_PERIOD_S * 1000000ULL);
  Serial.printf("Sleeping for %llu s (light sleep, ~2 mA vs ~50 mA awake)...\n\n",
                DEMO_PERIOD_S);
  Serial.flush();
  esp_light_sleep_start();   /* returns here on wakeup; loop() calls setup()-equivalent */
}

void loop() {
  /* After light sleep, execution resumes here.
   * We re-enter the measurement cycle directly — no full setup() needed. */
  measure_and_print();
  esp_sleep_enable_timer_wakeup(DEMO_PERIOD_S * 1000000ULL);
  Serial.printf("Sleeping for %llu s...\n\n", DEMO_PERIOD_S);
  Serial.flush();
  esp_light_sleep_start();
}
