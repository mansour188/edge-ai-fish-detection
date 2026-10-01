# Fish detection from sonar echoes — project README

Last updated: 1 October 2026

This file explains everything we did, why we did it, what the results mean, and what to do next.
Read it top to bottom once; later use the sections as a reference.

---

## 1. The goal

A small battery/solar device on a fishing boat (Raspberry Pi + GPS + sonar) that answers, offline and in near real time:

> "Does this spot show signs of fish — yes or no, and how sure are you?"

It logs GPS + time + confidence when the answer is "fish". This README covers the **machine-learning part only**
(data → model → maps). The hardware side (reading the real sonar and GPS) is not built yet.

**Why not an LLM?** The input is a stream of numbers (echo strength by depth), not text. A small classifier
(Random Forest or a small CNN) is 100–1000× smaller, faster and more power-efficient than any LLM.

---

## 2. Words you need

| Word | Meaning |
|---|---|
| **Ping** | One sound pulse sent by the sonar. About 1 ping per second in our dataset. |
| **Echo** | The sound that comes back. Late echo = deep object. |
| **Sv (dB)** | Echo strength in decibels. About -90 = nothing/noise, about -60 = something weak, -30 or higher = very strong (seabed). |
| **Echogram** | Many pings side by side: a picture with depth (rows) and time/pings (columns). Fish schools look like bright blobs. |
| **Seabed / bottom** | The strongest echo of all. It is *not* fish but looks bright. |
| **Patch** | A small square picture cut from an echogram (we use 64 x 64 pixels: 64 depth rows x 64 pings). |
| **Label** | The correct answer for a patch: 1 = fish school, 0 = other. |
| **Trawl** | One fishing haul / survey leg. Our data has trawl numbers 4, 14 and 17 with echograms. |
| **Baseline** | A simple model that a fancier model must beat. |
| **Precision** | When the model says "fish", how often is it right? |
| **Recall** | Of all real fish, how many did the model find? |
| **F1** | One number combining precision and recall (1.0 is perfect). |
| **AUC** | Can the model rank fish above non-fish? 1.0 = perfect, 0.5 = coin flip, below 0.5 = worse than random. |
| **Threshold** | Probability above which we say "fish". It changes precision vs recall. |
| **Leave-one-trawl-out** | Train on 2 trawls, test on the 3rd, repeat 3 times. The honest way to test with so little data. |
| **Confound / shortcut** | The model finds an easy clue that is *not* the real thing (example below: depth). |

---

## 3. The data

### Dataset used
"Volume backscattering strength samples and echograms (38 kHz) associated to small pelagic fish schools in the
Gulf of California, Mexico" — Villalobos, López-Serrano, Nevárez-Martínez (2018), SEANOE.
Page: https://www.seanoe.org/data/00419/53034/
Recorded with a scientific Simrad EK60 echosounder, 38 kHz, May 2013.

Downloaded file: `LopezSerrano_et_al_data.RData` (9 MB). We saved it as `data/dataset.rdata`.
(There is also `LopezSerrano_et_al_code.R`, the authors' R code — useful to understand their columns.)

### What is inside the file
- `eco.L04`, `eco.L14`, `eco.L17`: three echograms (R objects; keys `depth` and `Sv`; `Sv` is a depth x ping matrix with NaN where there is no data).
  The number in the name is the trawl number (4, 14, 17).
- `SvData`: a table with **1209 labeled points**. Columns: `id, x, y, pingNumber, pingTime, depth, Sv038, trawlNo, category`.
  `category` is `schools` (575) or `other` (634). Each row is a single point, **not** an outlined area.
  (The column is called `Sv038` — not `S_v038`.)

### What we can use
Only trawls 4, 14 and 17 have echograms, so **536 labeled points** are usable (269 fish = 50%).
Trawls 8, 11, 12 and 15 have table rows but no echogram in this file, so they are skipped.

### Useful facts we measured
- Depth step is about 0.098 m per row, so a 64-row patch is about 6.3 m tall; 64 pings is about 64 seconds.
- We checked that the table matches the echogram: sample 1 has depth 6.407 m, and the echogram depth list has exactly that value at row 65.
  `2_prepare.py` prints a "value match" percentage for each trawl (it should be near 100%).

### Other datasets we looked at (and why we did not use them)
| Dataset | Verdict |
|---|---|
| NOAA NCEI Water Column Sonar Archive (170+ TB, free, S3 bucket `noaa-wcsd-pds`) | Real raw EK60/EK80 data, **no labels**. Good for volume/pretraining later. Read with Echopype or pyEcholab. |
| Malaspina EK60 dataset | Open raw echosounder files, no fish labels. |
| NOAA Atlantic herring echograms (861 annotated) | Annotated, but download terms not confirmed. |
| DIDSON / Caltech Fish Counting / Alaska sonar | Imaging sonar (acoustic camera) — a different kind of picture than a single-beam echosounder. |
| Kaggle fish images, Fish4Knowledge, FishNet, OzFish, Norfisk | Camera photos/video, not sonar. |
| FishSounds | Passive audio, a different task. |

---

## 4. Files in this project

```
fish-track/start/
  data/
    dataset.rdata            downloaded data (you provide)
    patches.npz              made by 2_prepare.py
    preview.png              made by 2_prepare.py (check it!)
  1_inspect.py               look inside the .rdata file
  2_prepare.py               cut 64x64 patches around every labeled point
  train.py                   train + honest test (Random Forest and 2D-CNN)
  3_predict.py               slide the model over an echogram -> probability map
  check_depth_shortcut.py    test whether depth alone explains the labels
  modal_train.py             optional long training on Modal (later)
  requirements.txt
  models/                    output of normal training
  models_matched/            output of training with --match-depth
  prediction_*.png           maps made by 3_predict.py
```

`patches.npz` contains: `X` (N, 64, 64) raw dB patches, `y` labels, `g` trawl number, `rows`/`cols` (position of each
point), and the cleaned echograms `sv_4`, `sv_14`, `sv_17`.

---

## 5. How to run everything from zero

```bash
cd fish-track/start
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy pandas scikit-learn scipy matplotlib rdata tensorflow

python3 1_inspect.py data/dataset.rdata       # 1. look inside the file
python3 2_prepare.py data/dataset.rdata       # 2. make patches (+ data/preview.png)
python3 train.py --quick                      # 3. 1-minute check that the code runs
python3 train.py --epochs 30 --no-bn          # 4. real training (CNN without BatchNorm)
python3 check_depth_shortcut.py               # 5. is the label just depth?
python3 train.py --epochs 30 --no-bn --match-depth   # 6. honest test with depth removed -> models_matched/
python3 3_predict.py 14 rf models_matched     # 7. map: trawl, model (rf|cnn), folder
```

Notes:
- Linux file names are case sensitive (`dataset.rdata` is not `dataset.RData`).
- The lines printed as "saved models/..." are a fixed text; with `--match-depth` the files really go to `models_matched/`.
- CUDA / "cuInit" / "retracing" / "shuffle=True" warnings are harmless (no GPU is used).
- `AUC nan` means the test set had only one class, so AUC cannot be computed.

---

## 6. What each script does

### 1_inspect.py
Reads the `.rdata` file with the `rdata` package and prints every object. (`pyreadr` failed with
"Invalid file, or file has unsupported features" because the echograms are R matrices/objects it cannot read.)

### 2_prepare.py
For each echogram (`eco.Lxx`) and each table row of the same trawl:
1. find the echogram row whose depth is closest to the sample depth;
2. find the column (ping) by matching the table's `Sv038` value to the echogram (tries offsets -3..+3, prints the match rate);
3. cut a **64 x 64 patch centered on that point** (NaN and outside-the-edge filled with -90 dB);
4. label = 1 if the point's category is `schools`, else 0.
Saves `data/patches.npz` and `data/preview.png` (echograms with red = schools, white = other points).

### train.py
- **Normalizing:** clip dB to -90..-30, scale to 0..1.
- **Random Forest:** 6 simple numbers per patch (mean, std, max, 90th percentile, fraction of pixels > 0.5, fraction > 0.7), 300 trees, balanced class weights. ~0.5 MB.
- **2D-CNN:** 3 blocks of Conv2D (16, 32, 64 filters) + ReLU + MaxPool, global average pooling, dropout 0.4, sigmoid output.
  Augmentation: left-right flip, small gain shift, small noise. Exported to TFLite (~30 KB).
- **Testing:** leave-one-trawl-out; threshold chosen on the pooled out-of-fold predictions.
- **Flags:** `--quick` (2 epochs, code check only), `--no-bn` (no BatchNorm — needed, see lessons), `--match-depth` (honest test, see below), `--epochs N`.
- **Saves:** `rf.joblib`, `fish_cnn.keras`, `fish_cnn_int8.tflite`, `config.json` (thresholds, patch size, dB range).

### 3_predict.py
Slides a 64 x 64 window (stride 16) over one echogram and draws a fish-probability map under the echogram,
with the labeled points on top (green = schools, cyan = other). It also masks the seabed (details in section 9).

### check_depth_shortcut.py
Trains a model that sees **only the depth** of each point (no echo at all) and tests it leave-one-trawl-out.

---

## 7. Results (in the order we got them)

All numbers come from your real runs, tested on trawls the model never saw.

| Step | Random Forest | 2D-CNN | Note |
|---|---|---|---|
| First run (with BatchNorm) | F1 0.94, AUC 0.99 | F1 0.53, AUC 0.70 | CNN fails: thresholds shift between trawls (recall 0.00 on two trawls) |
| CNN without BatchNorm (`--no-bn`) | F1 0.98 (thr 0.20), AUC 0.99 | F1 0.96 (thr 0.10), AUC 0.99 | CNN fixed, but trawl 14 still weak (F1 0.65) |
| Depth only (no echo) | **F1 0.87, AUC 0.93** | | The labels are confounded with depth |
| Depth-matched test (169 patches: 126 fish, 43 other) | F1 0.98 (thr 0.40), AUC 0.99 | F1 0.88 (thr 0.78), AUC 0.82 | Depth-only drops to AUC 0.67 here |

Depth-matched, per trawl (small test sets!):
- Random Forest: AUC 1.00 on trawl 14 (27 patches) and 1.00 on trawl 17 (105 patches); trawl 4 has one class only (AUC nan).
- 2D-CNN: AUC 0.99 on trawl 17, but **0.16 on trawl 14** (worse than random) — unstable.

**Current verdict:** the Random Forest passes the honest test; the 2D-CNN does not (too little data).
The pipeline works end to end. The detector is **not yet proven** on your sonar or in your waters.

---

## 8. Lessons learned (why the scripts look the way they do)

1. **Split by trawl, never randomly.** Neighboring pings are almost identical; a random split leaks and gives fake 99% scores.
   (Our first synthetic data gave 100% for exactly this reason: too easy.)
2. **BatchNorm hurt the CNN.** With ~500 patches from 3 echograms, BatchNorm made the output probabilities shift on an unseen echogram
   (AUC 1.00 but recall 0.00 at a fixed threshold). Removing it (`--no-bn`) fixed it. We believe this is the cause; it is not proven beyond the result.
3. **The labels are confounded with depth.** In all three trawls the labeled schools are shallower than the "other" points
   (e.g. trawl 14: schools rows 34–216, other rows 197–601). A depth-only model reached AUC 0.93. So the first scores looked much better than they were.
   Fix: `--match-depth` drops patches that contain padding (which reveals the surface) and keeps only depth bands that contain **both** classes
   (>= 5 of each in 40-row bands). This left 169 of 536 patches.
4. **The seabed looks like a school.** The models never saw a seabed in training, so they flag it as fish. We mask it in `3_predict.py`.
   On the boat, use the sonar's own depth reading instead (see section 10).
5. **Test scores only check the labeled points.** The maps check everything else (seabed, empty water, unlabeled areas). Use both.
6. **The Random Forest is the right first model** (as the original project brief said). Use the CNN once you have much more data.

---

## 9. How to read the maps, and the seabed mask

`prediction_<trawl>_<model>[_folder].png`: top = echogram (red line = detected seabed), bottom = fish probability
(bright = high, light grey = excluded). Green dots = labeled schools, cyan dots = labeled other.

**What to look for:** the bright area should sit on real bright blobs. The same depth should get *different* answers where the echo differs.
With `models_matched` this is true on trawl 4 (top-right at the same depth stays dark) and trawl 17 (left half dark, right half bright around the school columns).

**Seabed mask rule (3_predict.py):** a column has a seabed if the last 10 rows are "no data" and a strong echo (> -45 dB) exists.
From the deepest strong echo we walk **up** through the thick echo body (> -60 dB, gaps <= 3 rows) to its top edge. Windows must end 10 rows above that edge.
Values: `BOTTOM_DB = -45`, `BODY_DB = -60`, `SEABED_MARGIN = 10`, smoothing = median filter over 15 pings.

**Known problems of the mask:**
- Trawl 17: the far left (columns 0–190) is not masked (seabed echo there is followed by data, not "no data"); a small leak remains at the right edge.
- Trawl 4: a few small pale specks at the seabed edge (columns 50–150, 500–700).
- Fish within a few meters above the seabed are hidden by the margin.
- Trawl 14 has no seabed in view; the detector correctly finds none.

---

## 10. Limits — what is NOT proven

- Only **3 echograms / 536 labeled points** (169 after depth matching, of which about 43 are "other"). Some test sets are tiny.
- Thresholds are chosen on the same predictions that are scored, so the pooled F1 is slightly optimistic.
- "Other" means "a surrounding echo the authors sampled", not "verified empty water".
- Trawl 14: the model marks a wide band (rows 50–200) over almost the whole width, not only the blobs. It may be real dispersed fish/plankton, or still partly a depth effect. We cannot tell from labels. A person who knows these echograms should check.
- The top layer on the right of trawl 4 (columns 600–1020) has no labels; the CNN flags it, the Random Forest mostly does not.
- Different sonar: this is a scientific 38 kHz EK60 in the Gulf of California. Your transducer and Tunisian waters will look different. Expect the threshold and accuracy to change.
- One window needs 64 pings (about a minute here) before it gives an answer.
- **Do not use this model for real fishing decisions yet.** It is a prototype that proves the pipeline.

---

## 11. Next steps

### A. Raspberry Pi side (not built yet)
1. Decide the sonar/transducer: model, output type (serial, analog, NMEA), and whether it gives raw pings.
2. Write a logger that saves, for every ping: time, GPS lat/lon, the ping vector, and the sonar's **bottom depth**.
3. Run the Random Forest live on a sliding window; log GPS + time + confidence when above threshold.
4. Test the TFLite CNN speed on the Pi (`models*/fish_cnn_int8.tflite`), and power use.
5. Use the sonar's depth reading to exclude the seabed and everything below it.

### B. Collect your own training data (the most important step)
The depth trap must not happen again. Rules:
1. Record "no fish" at **all depths**, including depths where fish are found.
2. Label from a **second source** (catch, a reference fish finder, a person's check) — never from position or depth.
3. Include **hard negatives** as their own examples: seabed, plankton layers, surface noise/bubbles, engine noise.
4. Split tests by **trip or day**.
5. Run `check_depth_shortcut.py` on the new data. If depth alone scores high, fix the labels first.
6. Fine-tune on your data; set the threshold on your data.

### C. Later improvements
- More data, then retry the CNN (or a 1D-CNN on single pings, or several pings stacked).
- Add automatic "no fish" patches (seabed, empty deep water) to training.
- Use NOAA raw files for volume if you can label them.

### D. Modal (only when training becomes slow)
With ~500 patches, training takes minutes on a laptop, so Modal gives no benefit yet. Later:
```bash
pip install modal && modal setup
modal volume create fish-data
modal volume put fish-data data/patches.npz /patches.npz
modal run modal_train.py --epochs 40
modal volume get fish-data /models ./models_modal
```
(`modal_train.py` uses CPU; it has a comment showing how to switch to a GPU. Check Modal's docs if a command errors.)

---

## 12. Troubleshooting

| Problem | Cause / fix |
|---|---|
| `File ... does not exist` | Typo or wrong case in the file name (Linux is case sensitive). |
| `pyreadr ... unsupported features` | Use `rdata` (already in `1_inspect.py`). |
| Low "value match" in 2_prepare | The table and echogram do not line up; send the printed output for debugging. |
| TensorFlow will not install | Try Python 3.11 or 3.12 in a new venv. (It worked for you on 3.13.) |
| `Too few patches left after matching depth` | The two classes barely overlap in depth; the dataset cannot support that test. |
| `AUC nan` | Test set has only one class. |
| CNN has AUC near 1 but recall near 0 | Probability scale shifted between trawls; train with `--no-bn`, pick the threshold on pooled predictions. |

---

## 13. Decision log (short)

- LLM rejected: wrong input type, too big/slow/power-hungry.
- Dataset chosen: Gulf of California 38 kHz set (only labeled single-beam echosounder data we found); NOAA archive kept for later volume.
- Model chosen by you: 2D-CNN. Result: not reliable on this small data; Random Forest is the working model.
- Test method: leave-one-trawl-out, then depth-matched test after we found the depth confound.
- Seabed: masked in prediction; real device should use the sonar's depth.
- Public data is for proving the pipeline; the real model will come from your own recordings.

Note: scripts were tested by the assistant on fake data for errors; all real results above come from your own runs.
