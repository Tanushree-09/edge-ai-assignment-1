/* Host-side check of battery_model.h: reads "battery_v temp_c rings wifi_rssi age" per line on
 * stdin, prints "<float32 days> <int8 days>". Used to prove the C port matches Python.
 * Build (any C99 compiler):  cc -O2 -std=c99 -I battery_demo host_test.c -o host_test -lm */
#include <stdio.h>
#include "battery_model.h"

int main(void) {
  float r[BATTERY_N_FEATURES];
  while (scanf("%f %f %f %f %f", &r[0], &r[1], &r[2], &r[3], &r[4]) == 5)
    printf("%.6f %.6f\n", battery_days(r, 0), battery_days(r, 1));
  return 0;
}
