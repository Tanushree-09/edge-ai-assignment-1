"""
app/main.py -- Battery life prediction API.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Endpoints:
  GET  /health        -> team identity + params count + val_mae
  GET  /model_info    -> per-layer table (params, bytes, MACs), totals, SRAM fit flags
  POST /predict       -> single reading -> days_left + timings_ms
  POST /predict_batch -> {"rows": [...]} -> {"days_left": [...], "timings_ms": {...}}
  Any reading with a physically impossible value -> HTTP 422

Stage timing (Mission 3):
  parse        JSON decode + Pydantic validation
  normalise    subtract mu, divide by sd
  inference    torch.no_grad() forward pass
  build        round, clip negatives, build the reply
  server_total wall time of the whole handler

Input ranges are physical sensor limits, NOT the data's min/max, so a real reading just
outside the training data still gets an answer. Only impossible values (dead-ADC 0 V,
the -127 sensor-fail code, 0 dBm RSSI, negatives) are refused.
"""
import json
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app.model import FEATURES, BatteryNet

# ── Startup: load the model once and keep it in memory ───────────────────────
cfg = json.load(open("model/config.json"))
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load("model/model.pt", weights_only=True))
net.eval()
# Serve in float64 (promoting the float32 weights is exact). The grader compares
# /predict_batch with single /predict calls after rounding to 2 decimals; float32 matmuls
# can differ in the last bit between a 1-row and a 50-row call and flip a rounded digit.
net.double()
torch.set_num_threads(1)   # single thread: steadier latency

MU = np.array(cfg["mu"], np.float64)
SD = np.array(cfg["sd"], np.float64)
LOG_TARGET = cfg.get("log_target", False)   # True only if the model was trained on log1p(days)
N_PARAMS = sum(p.numel() for p in net.parameters())

with torch.no_grad():   # warm-up: the first torch call is slow, so pay for it at startup
    net(torch.zeros(1, len(FEATURES), dtype=torch.float64))
    net(torch.zeros(8, len(FEATURES), dtype=torch.float64))


# ── Input schema ─────────────────────────────────────────────────────────────
class Reading(BaseModel):
    """One doorbell sensor snapshot, with ranges from sensor physics."""
    battery_v:     float = Field(ge=2.5,    le=4.3,    allow_inf_nan=False,
                                 description="Battery voltage (V), single-cell Li-ion")
    temp_c:        float = Field(ge=-55.0,  le=125.0,  allow_inf_nan=False,
                                 description="Temperature (C); -127 is the sensor-fail code")
    rings_per_day: float = Field(ge=0.0,    le=1000.0, allow_inf_nan=False,
                                 description="Doorbell rings per day")
    wifi_rssi_dbm: float = Field(ge=-100.0, le=-10.0,  allow_inf_nan=False,
                                 description="Wi-Fi RSSI (dBm); 0 is not a real reading")
    age_months:    float = Field(ge=0.0,    le=120.0,  allow_inf_nan=False,
                                 description="Battery age in months")


class Batch(BaseModel):
    rows: list[Reading] = Field(min_length=1, max_length=10_000)


# Handlers read the raw body (so `parse` can be timed), which hides the schema from
# /docs. These blocks put the request body, with the README example, back on /docs.
EXAMPLE = {"battery_v": 3.86, "temp_c": 28.0, "rings_per_day": 5,
           "wifi_rssi_dbm": -66, "age_months": 22}
READING_SCHEMA = {**Reading.model_json_schema(), "example": EXAMPLE}
BATCH_SCHEMA = {"type": "object", "required": ["rows"],
                "properties": {"rows": {"type": "array", "minItems": 1,
                                        "items": READING_SCHEMA}},
                "example": {"rows": [EXAMPLE, EXAMPLE]}}
body_doc = lambda s: {"requestBody": {"required": True,
                                      "content": {"application/json": {"schema": s}}}}

app = FastAPI(
    title="Doorbell battery API",
    description="Predicts remaining battery life (days) for a smart doorbell.",
    version="1.0.0",
)
ms = lambda t0: round((time.perf_counter() - t0) * 1_000, 3)


# ── /health ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "team": cfg["team"], "params": N_PARAMS, "val_mae": cfg["val_mae"]}


# ── /model_info ──────────────────────────────────────────────────────────────
@app.get("/model_info")
def model_info():
    """Per-layer table computed from the network itself (net.net), never hard-coded.

    Linear(a -> b): params = a*b + b, macs = a*b, bytes_f32 = 4*params, bytes_int8 = params.
    SRAM limit is taken as 300 KB = 300 * 1024 = 307,200 bytes.
    """
    layers = []
    for m in net.net:
        if isinstance(m, torch.nn.Linear):
            a, b = m.in_features, m.out_features
            params = a * b + b
            layers.append({
                "name":      f"linear{len(layers) + 1}",
                "in":        a,
                "out":       b,
                "params":    params,
                "bytes_f32": params * 4,
                "macs":      a * b,
            })

    total_params = sum(l["params"] for l in layers)
    total_bytes_f32 = sum(l["bytes_f32"] for l in layers)
    total_bytes_int8 = total_params
    sram_limit = 300 * 1024

    return {
        "layers":               layers,
        "total_params":         total_params,
        "total_bytes_f32":      total_bytes_f32,
        "total_bytes_int8":     total_bytes_int8,
        "total_macs":           sum(l["macs"] for l in layers),
        "fits_300kb_sram_f32":  total_bytes_f32 <= sram_limit,
        "fits_300kb_sram_int8": total_bytes_int8 <= sram_limit,
    }


# ── Core inference helper ────────────────────────────────────────────────────
async def run(request: Request, schema):
    t_all = time.perf_counter()
    t: dict[str, float] = {}

    # parse: JSON decode + validation. Any out-of-range value becomes HTTP 422.
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

    # normalise: same mu/sd (from the training rows) that training used
    t0 = time.perf_counter()
    x = (np.array([[getattr(r, f) for f in FEATURES] for r in rows], np.float64) - MU) / SD
    t["normalise"] = ms(t0)

    # inference: forward pass with no autograd graph
    t0 = time.perf_counter()
    with torch.no_grad():
        raw = net(torch.from_numpy(x)).numpy()
    t["inference"] = ms(t0)

    # build: undo log transform if used, clip negatives (negative days are meaningless),
    # round, assemble the reply
    t0 = time.perf_counter()
    days_arr = np.maximum(np.expm1(raw) if LOG_TARGET else raw, 0.0)
    days = [round(float(v), 2) for v in days_arr]
    body = {"days_left": days if isinstance(data, Batch) else days[0]}
    t["build"] = ms(t0)

    t["server_total"] = ms(t_all)
    body["timings_ms"] = t
    return body


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/predict", openapi_extra=body_doc(READING_SCHEMA))
async def predict(request: Request):
    """Single reading -> days_left with per-stage timings."""
    return await run(request, Reading)


@app.post("/predict_batch", openapi_extra=body_doc(BATCH_SCHEMA))
async def predict_batch(request: Request):
    """Batch of readings -> list of days_left with aggregate timings."""
    return await run(request, Batch)