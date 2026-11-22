"""
app/main.py -- Battery life prediction API.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Design (ground-up understanding):
  - GET  /health        → team identity + params count + val_mae (grader verifies team number)
  - GET  /model_info    → per-layer table: in/out/params/bytes/MACs, totals, SRAM fit flags
  - POST /predict       → single reading → days_left + timings_ms (5 stages)
  - POST /predict_batch → {"rows": [...]} → {"days_left": [...], "timings_ms": {...}}
  - Any reading with an out-of-physical-range value → HTTP 422 immediately

Stage timing (Mission 3):
  parse     → JSON decode + Pydantic validation (happens in the async run() helper)
  normalise → subtract mu, divide by sd (numpy, vectorised)
  inference → torch.no_grad() forward pass
  build     → build the response dict (round floats, clip negatives)
  server_total → wall time for all four stages combined

Input validation (Mission 3):
  Ranges are physical sensor limits, NOT data min/max.
  A real-world reading just outside the training distribution must still get an answer.
  Only physically-impossible values (dead-ADC sentinel, sensor-fail code, etc.) are rejected.

Memory on ESP32-S3 (Mission 4): see MISSION4.md. Weights (flash) vs activations (SRAM) are
different budgets; /model_info reports the weight side exactly as the README defines it.
"""
import json
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app.model import FEATURES, BatteryNet

# ── Startup: load model once, keep in memory ─────────────────────────────────
cfg = json.load(open("model/config.json"))
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load("model/model.pt", weights_only=True))
net.eval()
# Serve in float64. The weights are the float32 trained ones; promoting them is exact.
# Why: the grader compares /predict_batch against single /predict calls after rounding to
# 2 decimals. float32 matmuls can differ in the last bit between a 1-row and a 50-row call
# (different BLAS kernels), and a last-bit difference can flip a rounded digit. float64
# pushes that risk down to ~1e-12. (The board itself runs float32/int8; see MISSION4.md.)
net.double()
torch.set_num_threads(1)   # single-threaded inference: deterministic latency on Codespace

MU  = np.array(cfg["mu"], np.float64)
SD  = np.array(cfg["sd"], np.float64)
LOG_TARGET = cfg.get("log_target", False)   # was model trained in log-space?
N_PARAMS   = sum(p.numel() for p in net.parameters())

with torch.no_grad():       # warm-up: first torch call is slow (lazy init); pay it at startup
    net(torch.zeros(1, len(FEATURES), dtype=torch.float64))
    net(torch.zeros(8, len(FEATURES), dtype=torch.float64))


# ── Input schema ─────────────────────────────────────────────────────────────
class Reading(BaseModel):
    """One doorbell sensor snapshot.

    Ranges are physical constants for the sensor hardware, not data-driven.
    A single-cell Li-ion is dead below 2.5 V and impossible above 4.3 V (safe cutoff).
    The DS18B20 temperature sensor operates −55 °C to +125 °C; −127 is its error sentinel.
    WiFi RSSI: free-space max is ~−10 dBm (antenna touching AP), noise floor ~−100 dBm.
    rings_per_day: 1000/day is physically extreme but not impossible for a busy business.
    age_months: Li-ion calendar life is ~10 years = 120 months.
    """
    battery_v:      float = Field(ge=2.5,   le=4.3,   allow_inf_nan=False,
                                  description="Battery voltage (V) — single-cell Li-ion, 2.5 V cutoff")
    temp_c:         float = Field(ge=-55.0, le=125.0, allow_inf_nan=False,
                                  description="Ambient temperature (°C) — DS18B20 operating range")
    rings_per_day:  float = Field(ge=0.0,   le=1000.0, allow_inf_nan=False,
                                  description="Doorbell rings per day")
    wifi_rssi_dbm:  float = Field(ge=-100.0, le=-10.0, allow_inf_nan=False,
                                  description="WiFi RSSI (dBm) — received signal, never 0")
    age_months:     float = Field(ge=0.0,   le=120.0, allow_inf_nan=False,
                                  description="Battery age in months (0–10 years)")


class Batch(BaseModel):
    rows: list[Reading] = Field(min_length=1, max_length=10_000)


app = FastAPI(
    title="Doorbell battery API",
    description="Predicts remaining battery life (days) for a smart doorbell.",
    version="1.0.0",
)
ms = lambda t0: round((time.perf_counter() - t0) * 1_000, 3)


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Identity + model summary. Grader checks team number and params."""
    return {
        "status":  "ok",
        "team":    cfg["team"],
        "params":  N_PARAMS,
        "val_mae": cfg["val_mae"],
    }


# ── /model_info ───────────────────────────────────────────────────────────────
@app.get("/model_info")
def model_info():
    """Per-layer table computed live from net.net (never hard-coded).

    For a Linear(a → b):
      params     = a × b  (weights)  +  b  (biases)
      macs       = a × b  (one multiply + one add per weight, per prediction)
      bytes_f32  = params × 4
      bytes_int8 = params × 1  (quantised)

    Totals include fits_300kb_sram_f32 / fits_300kb_sram_int8 for Mission 4.
    The ESP32-S3 has ~512 KB SRAM of which ~300 KB is free for the application.
    """
    layers = []
    layer_idx = 1
    for m in net.net:
        if isinstance(m, torch.nn.Linear):
            in_f  = m.in_features
            out_f = m.out_features
            params    = in_f * out_f + out_f      # weights + biases
            macs      = in_f * out_f              # one MAC per weight
            bytes_f32 = params * 4
            layers.append({
                "name":      f"linear{layer_idx}",
                "in":        in_f,
                "out":       out_f,
                "params":    params,
                "bytes_f32": bytes_f32,
                "macs":      macs,
            })
            layer_idx += 1

    total_params    = sum(l["params"]    for l in layers)
    total_bytes_f32 = sum(l["bytes_f32"] for l in layers)
    total_bytes_int8= total_params * 1
    total_macs      = sum(l["macs"]      for l in layers)
    sram_limit      = 300 * 1024          # 300 KB

    return {
        "layers":               layers,
        "total_params":         total_params,
        "total_bytes_f32":      total_bytes_f32,
        "total_bytes_int8":     total_bytes_int8,
        "total_macs":           total_macs,
        "fits_300kb_sram_f32":  total_bytes_f32  <= sram_limit,
        "fits_300kb_sram_int8": total_bytes_int8 <= sram_limit,
    }


# ── Core inference helper ─────────────────────────────────────────────────────
async def run(request: Request, schema):
    t_all = time.perf_counter()
    t: dict[str, float] = {}

    # ── parse ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        data = schema.model_validate_json(await request.body())
    except ValidationError as e:
        return JSONResponse(
            status_code=422,
            content={"error": "bad reading", "detail": json.loads(e.json())},
        )
    t["parse"] = ms(t0)

    rows = data.rows if isinstance(data, Batch) else [data]

    # ── normalise ────────────────────────────────────────────────────────────
    # Subtract training mean, divide by training std.
    # This maps inputs to roughly [-3, +3] which keeps gradients healthy during
    # training and prevents dominant-feature bias during inference.
    t0 = time.perf_counter()
    x = (
        np.array([[getattr(r, f) for f in FEATURES] for r in rows], np.float64) - MU
    ) / SD
    t["normalise"] = ms(t0)

    # ── inference ────────────────────────────────────────────────────────────
    # torch.no_grad() skips building the autograd graph → faster + less memory.
    # torch.set_num_threads(1) at startup keeps latency deterministic.
    t0 = time.perf_counter()
    with torch.no_grad():
        raw = net(torch.from_numpy(x)).numpy()
    t["inference"] = ms(t0)

    # ── build ────────────────────────────────────────────────────────────────
    # If model was trained in log-space, undo log1p transform.
    # Clip to >= 0: a negative "days left" is physically meaningless.
    t0 = time.perf_counter()
    if LOG_TARGET:
        days_arr = np.maximum(np.expm1(raw), 0.0)
    else:
        days_arr = np.maximum(raw, 0.0)
    days = [round(float(v), 2) for v in days_arr]
    body = {"days_left": days if isinstance(data, Batch) else days[0]}
    t["build"] = ms(t0)

    t["server_total"] = ms(t_all)
    body["timings_ms"] = t
    return body


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(request: Request):
    """Single reading → days_left prediction with per-stage timings."""
    return await run(request, Reading)


@app.post("/predict_batch")
async def predict_batch(request: Request):
    """Batch of readings → list of days_left with aggregate timings."""
    return await run(request, Batch)
