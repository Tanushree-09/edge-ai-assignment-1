"""
train.py -- Clean the data, fit baselines, train the network, save it for the server.

    python train.py --team 7 --hidden 32,32
Writes model/model.pt and model/config.json, which app/main.py loads.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from app.model import FEATURES, BatteryNet

ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
ap.add_argument("--hidden", default="32,32")
ap.add_argument("--epochs", type=int, default=1500)
ap.add_argument("--no-norm", action="store_true", help="skip normalisation (for the comparison)")
a = ap.parse_args()
hidden = tuple(int(h) for h in a.hidden.split(","))

df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")

# Mission 1: glitch rules. Each is a value a real sensor cannot report / a known
# "sensor failed" code, so removing these rows cannot remove real readings.
g_batt = df.battery_v < 2.5          # a Li-ion cell is never below ~2.5 V; 0.0 = dead ADC
g_temp = df.temp_c < -40             # below the sensor's range; -127 = failed-sensor code
g_rssi = df.wifi_rssi_dbm >= -10     # 0 dBm is not a real received signal; 0 = "no reading"
bad = (g_batt | g_temp | g_rssi).to_numpy()
print(f"{len(df)} rows, {bad.sum()} glitch rows dropped "
      f"(battery_v {g_batt.sum()}, temp_c {g_temp.sum()}, wifi_rssi {g_rssi.sum()})")
df = df[~bad]

rng = np.random.default_rng(0)
idx = rng.permutation(len(df)); n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
X, y = tr[FEATURES].to_numpy(np.float32), tr.days_left.to_numpy(np.float32)
Xv, yv = val[FEATURES].to_numpy(np.float32), val.days_left.to_numpy(np.float32)

# Mission 2: normalisation. mu/sd come from the TRAINING rows only: the validation rows
# stand in for unseen data, so using them would leak information about them into training.
if a.no_norm:
    mu, sd = np.zeros(X.shape[1], np.float32), np.ones(X.shape[1], np.float32)
else:
    mu, sd = X.mean(0), X.std(0) + 1e-8

mae = lambda p, t: float(np.mean(np.abs(p - t)))
print(f"baseline 1 (predict the mean): {mae(np.full_like(yv, y.mean()), yv):.2f} days")

# Mission 1: baseline 2, linear regression via least squares on [X, 1]
A = np.c_[X, np.ones(len(X))]
w, *_ = np.linalg.lstsq(A, y, rcond=None)
lin_pred = np.c_[Xv, np.ones(len(Xv))] @ w
print(f"baseline 2 (linear regression): {mae(lin_pred, yv):.2f} days")

torch.manual_seed(0)
net = BatteryNet(hidden)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)
Xvt = torch.tensor((Xv - mu) / sd)
for ep in range(a.epochs):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 128):
        b = perm[i:i + 128]
        loss = torch.nn.functional.mse_loss(net(Xt[b]), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 300 == 0 or ep == a.epochs - 1:
        with torch.no_grad():
            print(f"  epoch {ep:4d}  val MAE {mae(net(Xvt).numpy(), yv):.2f} days")
with torch.no_grad():
    val_mae = mae(net(Xvt).numpy(), yv)
    train_mae = mae(net(Xt).numpy(), y)
print(f"network {hidden}: train MAE {train_mae:.2f}, val MAE {val_mae:.2f} days")

os.makedirs("model", exist_ok=True)
torch.save(net.state_dict(), "model/model.pt")
json.dump(dict(team=a.team, hidden=list(hidden), mu=mu.tolist(), sd=sd.tolist(),
               features=FEATURES, val_mae=round(val_mae, 2)), open("model/config.json", "w"), indent=2)
print("saved model/model.pt and model/config.json")