# Mission 4: Would it fit inside the doorbell?

Team 7 (SynaptiX). Every rule below comes from the course decks (adithyamd.com/courses/edge-ai, classes 1 to 27); the class is
named next to each. Every number is reproducible with the commands in section 9.

## 0. The rules we are using (and where they come from)

| Rule | Class |
|---|---|
| Parameters of `Linear(a,b)` = a·b + b. MACs = a·b (biases are additions, left out). float32 = 4 B, int8 = 1 B | 7 |
| **Weights live in flash, activations live in SRAM.** A model can fit in flash and still fail to run | 4, 7 |
| While a layer runs its **input and its output** must both sit in SRAM; the peak over layers is the working memory | 7 |
| ~300 KB of the 512 KB SRAM is left for the model after Wi-Fi and firmware ("fits" means fits alongside everything else) | 5, 7 |
| Green < 40 % of what the board has, amber < 80 %, red beyond | 27 |
| There is no OS, no file system, no `pip install`. The model is a byte/const array compiled into the firmware | 4, 7 |
| Time estimates come from a **measured** MAC/s rate, not from cycle folklore | 15 |
| Edge vs cloud is decided by BLERP: bandwidth, latency, economics, reliability, privacy. "Edge AI is a design choice, not a default" | 2 |
| Same scaling in training and on the device, with the training set's mean/sd stored on the device | 6, 19 |

## 1. Our model

`5 → 16 → 16 → 1`, ReLU, trained on `log1p(days_left)` (`python train.py --team 7 --hidden 16,16 --log-target`).

| | value |
|---|---|
| baseline 1 (predict the mean) / baseline 2 (linear regression), validation MAE | 46.32 d / 21.10 d |
| network, validation MAE | **3.79 d** (train 3.05 d) |
| parameters / MACs per prediction | **385 / 352** |
| weights float32 | **1,540 B (1.50 KB)** |
| weights int8, README definition (1 B per parameter) | **385 B** |
| int8 as TFLite really stores it (int8 weights, int32 biases, one float32 scale per output channel) | 616 B |
| normalisation constants the board must also carry (mu and sd, 5 floats each) | 40 B |
| peak activation memory, layer by layer (input + output) | L1 21 floats, **L2 32 floats**, L3 17 floats → **128 B float32, 32 B int8** |
| share of the 300 KB budget (weights + peak activations + constants) | 1,708 B ≈ **0.6 %**, deep green |
| where it lives | weights + constants in flash (`const` arrays), activations in SRAM, nothing in PSRAM |

Both precisions fit with room to spare, so **we ship float32 on the board**. Quantising costs accuracy here (section 7), and class 24's rule is
to shrink only as far as a budget forces you.

## 2. `/model_info`

`app/main.py` builds the table by walking `net.net` (never hard-coded), so it stays right if the layer sizes change. It returns per-layer
`params`, `bytes_f32`, `macs`, plus `total_params`, `total_bytes_f32`, `total_bytes_int8`, `total_macs`, `fits_300kb_sram_f32`,
`fits_300kb_sram_int8`. For our model: 385 / 1,540 / 385 / 352 / true / true. `total_params` equals `/health`'s `params`.

## 3. The table for `5 → w → w → 1`

Parameters = w² + 8w + 1 (checked against real `BatteryNet((w, w))` objects), MACs = w² + 6w, peak activation = 8w bytes in float32.
KB = 1024 B. Time uses the course's **20 M MAC/s** (class 27, "have ~20 M MAC/s measured") for float32 and **40 M MAC/s** (class 24, optimised
int8 kernels, illustrative) for int8.

| w | params | float32 | int8 | peak act. f32 | f32 ≤ 300 KB | int8 ≤ 300 KB | f32 ≤ 8 MB PSRAM | int8 ≤ 8 MB PSRAM | MACs | time f32 @20 M | time int8 @40 M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 1,281 | 5.0 KB | 1.3 KB | 0.25 KB | yes | yes | yes | yes | 1,216 | 0.06 ms | 0.03 ms |
| 128 | 17,409 | 68.0 KB | 17.0 KB | 1.0 KB | yes | yes | yes | yes | 17,152 | 0.86 ms | 0.43 ms |
| 256 | 67,585 | 264.0 KB | 66.0 KB | 2.0 KB | yes | yes | yes | yes | 67,072 | 3.4 ms | 1.7 ms |
| 512 | 266,241 | 1,040.0 KB | 260.0 KB | 4.0 KB | **NO** | yes | yes | yes | 265,216 | 13.3 ms | 6.6 ms |
| 2048 | 4,210,689 | 16,448 KB | 4,112 KB | 16.0 KB | **NO** | **NO** | **NO** | **yes** | 4,206,592 | 210 ms | 105 ms |

Notes: w = 2048 float32 (16.06 MiB) does not even fit the 16 MB flash; w = 512 float32 is 4× over the SRAM budget. Weights read from PSRAM or flash go
through a cache and run slower than the SRAM figure (class 15, lab part B), so the w = 512 and w = 2048 times are optimistic.

## 4. Exactly where the weights stop fitting in 300 KB

| convention | float32 fits up to | stops at | int8 (1 B/param) fits up to | stops at |
|---|---|---|---|---|
| 300 KB = 300 × 1024 B | **w = 273** (306,856 B) | **w = 274** (309,076 B) | **w = 550** (306,901 B) | **w = 551** (308,010 B) |
| 300 KB = 300,000 B | w = 269 | w = 270 | w = 543 | w = 544 |

Stricter and more honest limits for the same family:
* weights **plus** peak activations: float32 up to w = 272, int8 up to w = 549;
* int8 with its real layout (int32 biases + scales): up to w = 543;
* the class 27 **green** rule (< 40 % of 300 KB = 120 KB, weights + activations): float32 up to **w = 170**, int8 up to **w = 345**;
* PSRAM (8 MiB): float32 up to w = 1,444, int8 up to w = 2,892; flash (16 MiB): float32 up to w = 2,044.

Weights can also stay in flash and be read through the cache (class 24: "doesn't fit SRAM is a warning, not always a stop"), so the 300 KB line is a
guideline for weights, but a hard line for activations.

## 5. One prediction on the board

Assumption, stated: the course's **20 M MAC/s** float32 rate (class 27's worked board budget; class 24 uses 10 M MAC/s float32 and 40 M MAC/s int8
with ESP-NN as illustrative). Nobody on the team has measured it yet.

| rate | time for 352 MACs |
|---|---|
| float32 @ 10 M MAC/s (class 24, pessimistic) | 35 µs |
| **float32 @ 20 M MAC/s (class 27)** | **≈ 18 µs** |
| int8 @ 40 M MAC/s (class 24, optimised kernels) | ≈ 9 µs |

Plus about 5 multiplies, 5 divides, one `expm1f` and the range checks: negligible. Energy at the course's 0.3 W "awake" estimate (class 17/18):
≈ 5 µJ per prediction. **To replace the estimate with a measurement** (class 15's whole point): flash `firmware/battery_demo/battery_demo.ino`; it prints
the real µs per prediction for float32 and int8. It has not been run on a board by us.

## 6. Doorbell or server? (BLERP, class 2)

Honesty first: not every letter applies. The payload is 5 numbers (≈ 100 B), nothing needs an answer in milliseconds, and nobody streams video, so
**B and L are weak**. The strong ones are:

* **R, reliability (decisive).** The prediction matters most when the battery is nearly flat, and a flat battery is plausibly when a Wi-Fi radio (the biggest current
  draw on the board; class 4: power constrains everything, class 9: the battery budget) is least able to transmit. We have not measured that brown-out behaviour, but a design that
  needs Wi-Fi to say "battery is dying" is built on it. On the board the answer is available with the network down. When a reading is impossible (0 V, −127 °C) the board says
  "sensor fault", never a guess (class 5, graceful degradation).
* **E, economics.** One prediction per hour is 8,760 calls per doorbell per year. At class 2's made-up ₹0.01 per call that is ₹87.6 per doorbell per year,
  forever, against a model that costs ≈ 1.5 KB of flash we already own.
* **Power.** Inference ≈ 5 µJ on the CPU. Keeping the radio on for even one second at the course's 0.3 W awake figure (the course gives no separate radio number; measure it) is
  300,000 µJ: about 57,000× more (class 18: a bursty device lives or dies by what it does while awake).
* **P, privacy.** `rings_per_day`, Wi-Fi strength and time are a household's habits (class 26: minimise what leaves the device). On-device, only the result
  needs to leave, if at all.

**Where the server wins** (class 2, the row that favours the cloud: updating the model; class 9: updates are hard): retraining on the whole fleet's data, monitoring drift,
and rolling out a better model over the air. So the design is a hybrid, class 5 pattern 2 ("edge filter, cloud analysis"): infer on the doorbell, upload a daily
summary or the unsure cases (class 6: "keep results, not raw data"), retrain on the server, push the new weights with a staged rollout (class 9). The endpoint we hand
in is the development twin of the on-board model.

## 7. Blind spots from missions 1 to 3 that change the answer here

| Blind spot | What we did |
|---|---|
| **Training/serving skew** (class 6 "the classic bug"). The board does not run Python | `mu`/`sd` and the `expm1` + clip-at-0 step are saved in `config.json` and exported into the C header; the C output matches PyTorch to 2e-6 relative on all 395 validation rows |
| **Log target.** The best change in the project (validation MAE 4.59 → 3.79) | but a log head magnifies int8 error through `exp`: validation MAE float32 3.79, int8 weights only 3.81, **full int8 3.98**, worst single row 6.4 days off. That is why we ship float32 |
| **Model size.** Mission 2's sweep used one seed | 3 seeds each, log target, 2,000 epochs: `8,8` mean 4.59 (range 4.26 to 5.11), **`16,16` mean 3.71 (3.64 to 3.79)**, `32,32` mean 3.81 (3.68 to 3.96). Bigger is not better, so 385 parameters |
| **Glitch rules must be physical** (class 10, datasheet "range") | the API ranges and the C range check are the same sensor limits (e.g. battery 2.5 to 4.3 V, RSSI −100 to −10 dBm), not the data's min/max; the five sentinel glitches (0 V, −127 °C, 0 dBm, negative rings, huge age) are refused with 422, and so is NaN/Infinity |
| **Selection bias.** The saved checkpoint is the best validation epoch | the last epoch scores 3.86 d, so expect **about 3.8 d (± 0.3)** on the company's 400 unseen readings, not 3.79 to the decimal |
| **Batch equals singles** (the grader checks it) | float32 matmuls can differ in the last bit between a 1-row and a 50-row call; a flipped rounded digit would fail the check. The server runs in float64, tested exactly equal on 3,000 rows |
| **Cold start** | the first torch call is slow; the server warms the model at startup so the first graded request is not an outlier |

## 8. "Can we store libraries on the board?" (class 4, 7, 25)

No `pip`, no file system, usually no OS (class 7, "no pip install"). Every library is compiled into the firmware and lives in flash; the model is a `const` array in the same
image. The usual stack is TFLite Micro/LiteRT Micro plus ESP-NN (int8 kernels using the S3 vector instructions), with a tensor arena in SRAM. **We do not need any of it:** a 385-parameter
MLP is three nested loops. `export_board.py` writes `firmware/battery_demo/battery_model.h` (weights, scales, mu/sd, range checks, float32 and int8 inference, no
dependencies beyond `math.h`); we compiled it with MSVC and checked it against Python (float32, int8 and five glitch rows). An int8 TFLite flow only starts to pay off
when the model approaches the class 27 amber line (~240 KB of weights) or when we need ESP-NN speed.

## 9. Reproduce

```
python train.py --team 7 --hidden 16,16 --log-target      # writes model/ (commit it)
python export_board.py                                    # footprint, int8 accuracy, writes firmware/battery_demo/battery_model.h
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 10. What is still unverified

* Nothing here has run on a physical ESP32-S3. The 18 µs figure is an estimate from the course's rate; flash `battery_demo.ino` to get ours.
* The company's 400 hidden readings are unseen: our expectation is about 3.8 d, not a measurement.
* The 4.3 V upper limit for `battery_v` is a Li-ion charge limit we chose; the README only asks for ranges based on what a real sensor can report.
