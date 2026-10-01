# 🐟 Edge AI Fish Detection

**A low-cost, offline device that tells fishermen whether a spot shows signs of fish — in real time, on battery and solar power.**
Sonar echo + GPS → small on-device model → fish / no-fish with a model score, logged on a map.

> **Trained on real sonar data.** The model is trained on real echosounder recordings (38 kHz, Gulf of California) from a published scientific dataset,
> labeled by the dataset's authors — **not on synthetic data**. It has not yet been trained or tested on our own boat or sonar; that is the next funded step.

## ▶️ Demo video


https://github.com/user-attachments/assets/dff2e4f1-6874-4af9-8334-ae121d396ab3



## The problem
Small-scale fishing crews burn time and fuel searching open water without knowing if fish are there. Industrial fish finders are expensive and
need cloud or ship electronics. There is no cheap, offline, GPS-tagged "is there fish here?" tool for small boats (our target: fishing vessels in Tunisia).

## The solution
| Stage | What it does |
|---|---|
| Sonar transducer + GPS | Echo strength by depth, plus position and time |
| Preprocessing | Seabed exclusion, normalization |
| Model | Lightweight classifier (Random Forest, ~0.3–0.5 MB; 2D-CNN exported to a 30 KB TFLite file) |
| Logger | GPS + time + confidence saved when probability crosses a threshold |
| Power | Battery + small solar panel; Raspberry Pi class board (~1–5 W) |

Everything runs **offline on the device**. No LLM is used: this is a signal-classification task, and a small classifier is 100–1000× smaller and cheaper to run.

## Live demo (this repo)
This repository hosts the model behind an API and streams **real public echograms** as if they came from a sonar, with a live dashboard:
scrolling echogram, per-depth fish probability, decision, detection log, and a GPS track map.

> ⚠️ The echoes are real (public Gulf of California 38 kHz data, replayed). **The GPS track is simulated.** This is a prototype, not validated on a boat.

```bash
pip install -r requirements.txt
uvicorn server:app --port 7860          # open http://localhost:7860
```
The trained model (`model/`) and the replay data (`data/replay.npz`) are included. To rebuild the replay file from the training data:
`python3 make_replay.py path/to/patches.npz`.
API: `GET /` dashboard · `GET /stream?trawl=17&speed=20` live events (SSE) · `POST /predict` (64×64 dB patch → probability) · `GET /api/model`.

**Run in Docker / deploy a public link** (GitHub Pages cannot run Python):
`docker build -t fish-demo . && docker run -p 7860:7860 fish-demo`, or deploy this repo on Hugging Face Spaces (Docker SDK, port 7860), Render or Fly.io.
Pin the **same `scikit-learn` version** as in training in `requirements.txt` (the model is a pickled file).

## What we measured (honestly)
- Data: 3 labeled echograms, 536 labeled points. Tested **leave-one-trawl-out** (train on 2 trawls, test on the 3rd).
- We found the labels were **confounded with depth** (a depth-only model reached AUC 0.93), so we re-tested on depth-matched data (169 patches).
  Depth-only fell to AUC 0.67; the Random Forest stayed at **AUC 0.99, F1 0.98**; the 2D-CNN was unstable (AUC 0.82) and needs more data.
- Limits: very small dataset, different sonar and sea than our target, tiny test sets, thresholds tuned on the same predictions. Details: [`docs/TECHNICAL.md`](docs/TECHNICAL.md).

## Roadmap and what funding enables
1. **Sensor bring-up** — Raspberry Pi + GPS + sonar with raw ping output; log every ping.
2. **Own data collection at sea** — labeled with catches / a reference fish finder, including fish-free water at all depths, seabed, plankton and noise.
3. **Retrain and validate** on our own data, split by trip; fine-tune the threshold.
4. **Power and enclosure** — battery + solar sizing, weatherproofing, field trial.
5. **Optional connectivity** — LoRa/GSM sync of detections.

Estimated prototype hardware (from our brief): board $15–50, GPS $5–10, sonar $30–100, battery $15–25, solar + controller $15–30, optional radio $10–30.
Own data collection ≈ $100. **Funding ask: _[amount / purpose — fill in]_**

## Repository layout
```
server.py            API + live stream (hosts the model)
dashboard.html       live dashboard
make_replay.py       builds data/replay.npz from the training data
model/               trained Random Forest + config
data/replay.npz      replayed real echograms (public dataset)
Dockerfile, requirements.txt
docs/TECHNICAL.md    full technical write-up: data, method, results, lessons
```
The training pipeline (data preparation, leave-one-trawl-out test, depth-matched test) is described in [`docs/TECHNICAL.md`](docs/TECHNICAL.md).

## Credits
Echogram data: Villalobos, López-Serrano, Nevárez-Martínez (2018), *Volume backscattering strength samples and echograms (38 kHz) associated to
small pelagic fish schools in the Gulf of California, Mexico*, SEANOE — https://www.seanoe.org/data/00419/53034/ (check the dataset's license before redistributing).
