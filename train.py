"""
train.py -- Clean the data, fit baselines, train the network, save it for the server.

    python train.py --team 7 --hidden 16,16 --log-target        # the model that is committed in model/
Writes model/model.pt and model/config.json, which app/main.py loads
(--out DIR writes somewhere else, --seed N changes the init; see MISSION4.md for the 3-seed comparison).

Design decisions:
  1. Glitch rules: physical impossibility rules, not data-min/max clipping.
     battery_v=0 → dead ADC; temp=-127 → failed sensor sentinel; wifi=0 → no reading.
  2. Normalisation: mu/sd from TRAINING rows only. Validation is held-out future data —
     using its statistics would leak information and give optimistic metrics.
     mu/sd are saved in config.json because the device must apply the same two numbers per
     feature (training/serving skew is the classic edge bug).
  3. Log-space target (--log-target): days_left is right-skewed (p50=51, max=322). Training on
     log(days_left+1) balances urgent low-battery cases against rare 300-day readings.
     The server/board must undo it with expm1 and clip at 0 (recorded as log_target in config.json).
  4. Cosine-annealing LR; the best checkpoint (validation MAE, checked every 50 epochs) is kept.
     No early stopping: the best weights arrive late, when the learning rate is small.
  5. L2 weight decay (1e-5): light regularisation to prevent overfitting on 1976 rows.
  6. The model is intentionally small (385 params, 1.5 KB as float32) so it fits the ESP32-S3
     with room to spare.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from app.model import FEATURES, BatteryNet

ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
ap.add_argument("--hidden", default="32,32")
ap.add_argument("--epochs", type=int, default=2000)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--seed", type=int, default=0, help="weight-init / batch-order seed")
ap.add_argument("--out", default="model", help="output folder (default: model, which the server loads)")
ap.add_argument("--no-norm", action="store_true", help="skip normalisation (for the comparison)")
ap.add_argument("--log-target", action="store_true",
                help="train on log1p(days_left) and expm1 on output — better for skewed targets")
a = ap.parse_args()
hidden = tuple(int(h) for h in a.hidden.split(","))

# ── 1. Load & clean ──────────────────────────────────────────────────────────
df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")

# Mission 1: glitch rules — each catches a known sensor failure code or physical
# impossibility. We use physical constants, not data-driven thresholds, so a real
# reading that happens to be at the edge of the training range is never dropped.
g_batt = df.battery_v < 2.5         # Li-ion cutoff; 0.0 = dead ADC sentinel
g_temp = df.temp_c < -40            # below sensor spec; -127 = DS18B20 failure code
g_rssi = df.wifi_rssi_dbm >= -10    # 0 dBm impossible for a received signal; 0 = no scan
bad = (g_batt | g_temp | g_rssi).to_numpy()
print(f"{len(df)} rows, {bad.sum()} glitch rows dropped "
      f"(battery_v {g_batt.sum()}, temp_c {g_temp.sum()}, wifi_rssi {g_rssi.sum()})")
df = df[~bad].reset_index(drop=True)

# ── 2. Train/validation split ────────────────────────────────────────────────
rng = np.random.default_rng(0)
idx = rng.permutation(len(df)); n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
X,  y  = tr[FEATURES].to_numpy(np.float32),  tr.days_left.to_numpy(np.float32)
Xv, yv = val[FEATURES].to_numpy(np.float32), val.days_left.to_numpy(np.float32)

# ── 3. Baselines ─────────────────────────────────────────────────────────────
mae = lambda p, t: float(np.mean(np.abs(p - t)))
print(f"baseline 1 (predict the mean): {mae(np.full_like(yv, y.mean()), yv):.2f} days")

# Mission 1: baseline 2 — linear regression via least squares on [X, 1]
A = np.c_[X, np.ones(len(X))]
w, *_ = np.linalg.lstsq(A, y, rcond=None)
lin_pred = np.c_[Xv, np.ones(len(Xv))] @ w
print(f"baseline 2 (linear regression): {mae(lin_pred, yv):.2f} days")

# ── 4. Normalisation ─────────────────────────────────────────────────────────
# Mission 2: mu/sd from training rows only — validation rows simulate unseen data.
if a.no_norm:
    mu, sd = np.zeros(X.shape[1], np.float32), np.ones(X.shape[1], np.float32)
else:
    mu, sd = X.mean(0).astype(np.float32), (X.std(0) + 1e-8).astype(np.float32)

Xn  = (X  - mu) / sd
Xvn = (Xv - mu) / sd

# ── 5. Training ──────────────────────────────────────────────────────────────
# Log-space target: days_left is right-skewed. log1p maps [0, 322] → [0, 5.78].
# The network then has a balanced regression target and handles low-battery (urgent)
# cases without the loss being dominated by rare 300-day readings.
if a.log_target:
    y_train = np.log1p(y).astype(np.float32)
    print(f"[log-target] training on log1p(days_left): range [{y_train.min():.2f}, {y_train.max():.2f}]")
else:
    y_train = y

torch.manual_seed(a.seed)
net = BatteryNet(hidden)
n_params = sum(p.numel() for p in net.parameters())
print(f"network {hidden}: {n_params} parameters")

opt   = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-4)

Xt  = torch.from_numpy(Xn)
yt  = torch.from_numpy(y_train)
Xvt = torch.from_numpy(Xvn)

best_val_mae = float("inf")
best_state   = None

# No early stopping: with a cosine schedule the best weights arrive late, when the
# learning rate is small. We just keep the best checkpoint seen (checked every 50 epochs).
for ep in range(a.epochs):
    net.train()
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 128):
        b = perm[i:i + 128]
        loss = nn.functional.mse_loss(net(Xt[b]), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()

    if ep % 50 == 0 or ep == a.epochs - 1:
        net.eval()
        with torch.no_grad():
            raw = net(Xvt).numpy()
            val_pred = np.expm1(raw) if a.log_target else raw
            val_pred = np.maximum(val_pred, 0)
            vm = mae(val_pred, yv)

            raw_tr = net(Xt).numpy()
            tr_pred = np.expm1(raw_tr) if a.log_target else raw_tr
            tr_pred = np.maximum(tr_pred, 0)
            tm = mae(tr_pred, y)

        if vm < best_val_mae:
            best_val_mae = vm
            best_state   = {k: v.clone() for k, v in net.state_dict().items()}
        if ep % 250 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep:4d}  val MAE {vm:.2f}  train MAE {tm:.2f}  (best so far {best_val_mae:.2f})")

# Restore best checkpoint
net.load_state_dict(best_state)
net.eval()
with torch.no_grad():
    raw_v = net(Xvt).numpy()
    raw_t = net(Xt).numpy()
    val_pred  = np.maximum(np.expm1(raw_v) if a.log_target else raw_v, 0)
    train_pred = np.maximum(np.expm1(raw_t) if a.log_target else raw_t, 0)

val_mae   = mae(val_pred,  yv)
train_mae = mae(train_pred, y)
print(f"network {hidden}: train MAE {train_mae:.2f}, val MAE {val_mae:.2f} days")

# ── 6. Save ──────────────────────────────────────────────────────────────────
os.makedirs(a.out, exist_ok=True)
torch.save(net.state_dict(), f"{a.out}/model.pt")
json.dump(dict(
    team=a.team,
    hidden=list(hidden),
    mu=mu.tolist(),
    sd=sd.tolist(),
    features=FEATURES,
    val_mae=round(val_mae, 2),
    log_target=a.log_target,
), open(f"{a.out}/config.json", "w"), indent=2)
print(f"saved {a.out}/model.pt and {a.out}/config.json")
