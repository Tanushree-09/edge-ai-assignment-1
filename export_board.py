"""
export_board.py -- "Would it fit inside the doorbell?" made executable (Mission 4).

    python export_board.py                # reads model/, writes firmware/battery_demo/battery_model.h

What it does, in the order the course teaches it (Class 7, 15, 24, 25):
  1. Footprint: weights in flash, activations in SRAM (input AND output of a layer are live
     together), everything else the board must also carry (mu/sd, scales).
  2. Post-training int8 quantisation (PTQ), simulated in integer arithmetic exactly as the C
     code will do it, and its accuracy cost on the SAME validation rows train.py uses.
  3. A dependency-free C header: weights as const arrays (they live in flash, no file system),
     the normalisation constants (training/serving skew is THE classic bug, Class 6),
     the same physical range checks as the API ("never a guess"), and float32 + int8 inference.
     A 385-parameter MLP needs no runtime library: no TFLite Micro, no ESP-NN. The whole
     "library" is three nested loops (Class 4: the model is one of seven layers; here the
     other six are tiny).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from app.model import FEATURES, BatteryNet

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="model")
ap.add_argument("--out", default="firmware/battery_demo")
ap.add_argument("--dump-ref", default=None,
                help="write validation rows + Python float32/int8 predictions to this CSV (to cross-check the C port)")
a = ap.parse_args()

# ── load model + the exact validation split train.py used ────────────────────
cfg = json.load(open(f"{a.model}/config.json"))
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load(f"{a.model}/model.pt", weights_only=True))
net.eval()
MU = np.array(cfg["mu"], np.float32)
SD = np.array(cfg["sd"], np.float32)
LOG_T = bool(cfg.get("log_target", False))

df = pd.read_csv(f"data/team_{cfg['team']:02d}_train.csv")
df = df[~((df.battery_v < 2.5) | (df.temp_c < -40) | (df.wifi_rssi_dbm >= -10))].reset_index(drop=True)
rng = np.random.default_rng(0)
idx = rng.permutation(len(df)); n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
Xtr = ((tr[FEATURES].to_numpy(np.float32) - MU) / SD).astype(np.float64)
Xv = ((val[FEATURES].to_numpy(np.float32) - MU) / SD).astype(np.float64)
yv = val.days_left.to_numpy(np.float64)
mae = lambda p, t: float(np.mean(np.abs(p - t)))


def to_days(raw):
    d = np.expm1(raw) if LOG_T else raw
    return np.maximum(d, 0.0)


lin = [m for m in net.net if isinstance(m, nn.Linear)]
W = [m.weight.detach().numpy().astype(np.float64) for m in lin]
B = [m.bias.detach().numpy().astype(np.float64) for m in lin]
L = len(W)
dims = [W[0].shape[1]] + [w.shape[0] for w in W]          # 5, 16, 16, 1

# ── 1. footprint ─────────────────────────────────────────────────────────────
n_w = sum(w.size for w in W); n_b = sum(b.size for b in B)
f32_bytes = (n_w + n_b) * 4
int8_readme = n_w + n_b                                    # README's definition: 1 byte per parameter
int8_real = n_w + 4 * n_b + 4 * sum(w.shape[0] for w in W)  # int8 weights, int32 biases, f32 scale per channel
norm_bytes = 4 * 2 * len(FEATURES)
peak_pair = max(dims[i] + dims[i + 1] for i in range(L))   # layer input + output live together
macs = sum(w.size for w in W)
print(f"model {tuple(cfg['hidden'])}: {n_w + n_b} params, {macs} MACs per prediction (biases not counted)")
print(f"weights f32 {f32_bytes} B | int8 (README: 1 B/param) {int8_readme} B | int8 as TFLite really stores it "
      f"(int8 W + int32 b + f32 scale/channel) {int8_real} B | mu+sd {norm_bytes} B")
print(f"peak activation memory (input+output of the widest layer): f32 {peak_pair * 4} B, int8 {peak_pair} B")

# ── 2. int8 PTQ, simulated in integer arithmetic ─────────────────────────────
def float_forward(x, Ws, Bs):
    h = x
    for k in range(L):
        h = h @ Ws[k].T + Bs[k]
        if k < L - 1:
            h = np.maximum(h, 0)
    return h[:, 0]


# calibration on TRAINING rows (Class 24: bad calibration data is the commonest cause of a bad PTQ model)
acts = [Xtr]
h = Xtr
for k in range(L - 1):
    h = np.maximum(h @ W[k].T + B[k], 0); acts.append(h)
s_act = [max(np.abs(x).max(), 1e-9) / 127.0 for x in acts]       # symmetric, per-tensor, one scale per layer input

s_w = [np.maximum(np.abs(w).max(1), 1e-12) / 127.0 for w in W]   # symmetric, per output channel
QW = [np.clip(np.rint(W[k] / s_w[k][:, None]), -127, 127).astype(np.int8) for k in range(L)]
QB = [np.rint(B[k] / (s_act[k] * s_w[k])).astype(np.int32) for k in range(L)]
QM = []                                                           # float32 multipliers, as in the C code
for k in range(L):
    m = s_act[k] * s_w[k] / (s_act[k + 1] if k < L - 1 else 1.0)
    QM.append(m.astype(np.float32))


def int8_forward(x):
    q = np.clip(np.rint(x / s_act[0]), -127, 127).astype(np.int32)
    for k in range(L):
        acc = q @ QW[k].astype(np.int32).T + QB[k]
        v = acc.astype(np.float32) * QM[k]
        if k < L - 1:
            q = np.clip(np.rint(np.maximum(v, 0)), 0, 127).astype(np.int32)
        else:
            return v[:, 0].astype(np.float64)


Wdq = [QW[k].astype(np.float64) * s_w[k][:, None] for k in range(L)]
p_f = to_days(float_forward(Xv, W, B))
p_wo = to_days(float_forward(Xv, Wdq, B))
p_i8 = to_days(int8_forward(Xv))
print(f"\nvalidation MAE ({len(yv)} rows, days):  float32 {mae(p_f, yv):.3f} | int8 weights only {mae(p_wo, yv):.3f} "
      f"| full int8 {mae(p_i8, yv):.3f}")
print(f"int8 vs float32 predictions: mean |diff| {np.mean(np.abs(p_i8 - p_f)):.3f} d, worst {np.max(np.abs(p_i8 - p_f)):.3f} d")
if LOG_T:
    print("(log-target model: the final int8 error is multiplied by exp(), so it matters more than for a linear head)")
if a.dump_ref:
    ref = val[FEATURES].copy()
    with torch.no_grad():                      # the real float32 network, not the float64 numpy re-implementation
        ref["py_f32"] = to_days(net(torch.from_numpy(Xv.astype(np.float32))).numpy().astype(np.float64))
    ref["py_i8"] = p_i8
    ref.to_csv(a.dump_ref, index=False)


# ── 3. C header ──────────────────────────────────────────────────────────────
def carr(ctype, name, arr, fmt):
    flat = np.asarray(arr).ravel()
    rows = []
    for i in range(0, len(flat), 8):
        rows.append("  " + ", ".join(fmt(v) for v in flat[i:i + 8]))
    return f"static const {ctype} {name}[{len(flat)}] = {{\n" + ",\n".join(rows) + "\n};\n"


def ff(v):
    """C float literal: always has a '.' or an exponent before the 'f' (so 0 -> 0.0f, not the invalid 0f)."""
    s = f"{float(v):.9g}" if np.isfinite(v) else "0.0"
    return s + ("f" if ("." in s or "e" in s) else ".0f")


fi = lambda v: str(int(v))
ranges = {  # keep identical to app/main.py (physical sensor limits, never the data's min/max)
    "battery_v": (2.5, 4.3), "temp_c": (-55.0, 125.0), "rings_per_day": (0.0, 1000.0),
    "wifi_rssi_dbm": (-100.0, -10.0), "age_months": (0.0, 120.0)}
max_w = max(dims)

h = []
h.append("/* GENERATED by export_board.py. Do not edit by hand: re-run the script after retraining.\n"
         f" * model 5 -> {' -> '.join(str(d) for d in dims[1:])}, {n_w + n_b} parameters, {macs} MACs, "
         f"team {cfg['team']}, validation MAE {cfg['val_mae']} days (float32).\n"
         " * const arrays live in flash on the ESP32-S3 (no file system, no pip, no runtime library needed). */\n"
         "#pragma once\n#include <math.h>\n#include <stdint.h>\n\n"
         f"#define BATTERY_N_FEATURES {len(FEATURES)}\n#define BATTERY_N_LAYERS {L}\n"
         f"#define BATTERY_MAX_WIDTH {max_w}\n#define BATTERY_LOG_TARGET {1 if LOG_T else 0}\n\n")
h.append("/* feature order: " + ", ".join(FEATURES) + " */\n")
h.append(carr("float", "BATTERY_MU", MU, ff))
h.append(carr("float", "BATTERY_SD", SD, ff))
h.append(f"static const float BATTERY_Q_IN_SCALE = {ff(s_act[0])};\n\n")
h.append("static const int BATTERY_IN[BATTERY_N_LAYERS]  = {" + ", ".join(str(d) for d in dims[:-1]) + "};\n")
h.append("static const int BATTERY_OUT[BATTERY_N_LAYERS] = {" + ", ".join(str(d) for d in dims[1:]) + "};\n\n")
for k in range(L):
    h.append(carr("float", f"F_W{k}", W[k], ff)); h.append(carr("float", f"F_B{k}", B[k], ff))
for k in range(L):
    h.append(carr("int8_t", f"Q_W{k}", QW[k], fi)); h.append(carr("int32_t", f"Q_B{k}", QB[k], fi))
    h.append(carr("float", f"Q_M{k}", QM[k], ff))
h.append("static const float *const F_W[BATTERY_N_LAYERS] = {" + ", ".join(f"F_W{k}" for k in range(L)) + "};\n")
h.append("static const float *const F_B[BATTERY_N_LAYERS] = {" + ", ".join(f"F_B{k}" for k in range(L)) + "};\n")
h.append("static const int8_t *const Q_W[BATTERY_N_LAYERS] = {" + ", ".join(f"Q_W{k}" for k in range(L)) + "};\n")
h.append("static const int32_t *const Q_B[BATTERY_N_LAYERS] = {" + ", ".join(f"Q_B{k}" for k in range(L)) + "};\n")
h.append("static const float *const Q_M[BATTERY_N_LAYERS] = {" + ", ".join(f"Q_M{k}" for k in range(L)) + "};\n\n")

h.append(r'''/* Same physical limits as the API. A reading outside them is a sensor fault (0 V = dead ADC,
 * -127 C = failed probe, 0 dBm = no scan): the answer is "no answer", never a guess. */
static inline int battery_reading_ok(const float r[BATTERY_N_FEATURES]) {
  static const float LO[BATTERY_N_FEATURES] = {''' + ", ".join(ff(ranges[f][0]) for f in FEATURES) + r'''};
  static const float HI[BATTERY_N_FEATURES] = {''' + ", ".join(ff(ranges[f][1]) for f in FEATURES) + r'''};
  for (int i = 0; i < BATTERY_N_FEATURES; i++)
    if (!isfinite(r[i]) || r[i] < LO[i] || r[i] > HI[i]) return 0;
  return 1;
}

static inline float battery_post(float y) {          /* undo the training-time target transform */
#if BATTERY_LOG_TARGET
  y = expm1f(y);
#endif
  return y < 0.0f ? 0.0f : y;                          /* negative days make no sense */
}

/* float32 path. Activation memory: two ping-pong buffers of BATTERY_MAX_WIDTH floats. */
static float battery_forward_f32(const float *x) {
  float a[BATTERY_MAX_WIDTH], b[BATTERY_MAX_WIDTH];
  const float *src = x; float *dst = a;
  for (int l = 0; l < BATTERY_N_LAYERS; l++) {
    const int in = BATTERY_IN[l], out = BATTERY_OUT[l];
    for (int o = 0; o < out; o++) {
      float acc = F_B[l][o];
      for (int i = 0; i < in; i++) acc += F_W[l][o * in + i] * src[i];
      if (l < BATTERY_N_LAYERS - 1 && acc < 0.0f) acc = 0.0f;   /* ReLU */
      dst[o] = acc;
    }
    src = dst; dst = (dst == a) ? b : a;
  }
  return src[0];
}

/* int8 path: int8 weights/activations, int32 accumulators, one float multiply per output. */
static float battery_forward_i8(const float *xn) {
  int8_t a[BATTERY_MAX_WIDTH], b[BATTERY_MAX_WIDTH], q0[BATTERY_MAX_WIDTH];
  for (int i = 0; i < BATTERY_N_FEATURES; i++) {
    long q = lrintf(xn[i] / BATTERY_Q_IN_SCALE);
    q0[i] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
  }
  const int8_t *src = q0; int8_t *dst = a;
  for (int l = 0; l < BATTERY_N_LAYERS; l++) {
    const int in = BATTERY_IN[l], out = BATTERY_OUT[l];
    for (int o = 0; o < out; o++) {
      int32_t acc = Q_B[l][o];
      for (int i = 0; i < in; i++) acc += (int32_t)Q_W[l][o * in + i] * (int32_t)src[i];
      float v = (float)acc * Q_M[l][o];
      if (l == BATTERY_N_LAYERS - 1) return v;
      if (v < 0.0f) v = 0.0f;                                   /* ReLU */
      long r = lrintf(v);
      dst[o] = (int8_t)(r > 127 ? 127 : r);
    }
    src = dst; dst = (dst == a) ? b : a;
  }
  return 0.0f;
}

/* Public API: returns days left, or -1.0f if the reading is physically impossible. */
static float battery_days(const float raw[BATTERY_N_FEATURES], int use_int8) {
  if (!battery_reading_ok(raw)) return -1.0f;
  float x[BATTERY_N_FEATURES];
  for (int i = 0; i < BATTERY_N_FEATURES; i++) x[i] = (raw[i] - BATTERY_MU[i]) / BATTERY_SD[i];
  return battery_post(use_int8 ? battery_forward_i8(x) : battery_forward_f32(x));
}
''')

os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, "battery_model.h")
open(path, "w", newline="\n").write("".join(h))
print(f"\nwrote {path} ({os.path.getsize(path)} bytes of source; the compiled constants are the sizes printed above)")
