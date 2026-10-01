"""Fish-detection demo server: hosts the model, replays real echograms as a live stream (SSE).
Run: uvicorn server:app --port 7860   (from the demo/ folder)"""
import asyncio, base64, json, os, time
from pathlib import Path
import numpy as np, joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

BASE = Path(__file__).parent
MD = Path(os.getenv("MODEL_DIR", BASE / "model"))
rf = joblib.load(MD / "rf.joblib")
rf.n_jobs = 1                              # tiny batches: threads only add overhead
cfg = json.load(open(MD / "config.json"))
P, LO, HI, THR = cfg["patch"], cfg["db_low"], cfg["db_high"], cfg.get("rf_threshold", 0.5)
REPLAY = np.load(BASE / "data" / "replay.npz")
STEP = 4                                    # pings per stream tick
app = FastAPI(title="Fish detection demo")


def feats(x):                               # x: (n, P, P) in dB -> same 6 features as train.py
    x = np.clip((x - LO) / (HI - LO), 0, 1).reshape(len(x), -1)
    return np.c_[x.mean(1), x.std(1), x.max(1), np.percentile(x, 90, axis=1), (x > .5).mean(1), (x > .7).mean(1)]


def seabed_top(col):                        # same rule as 3_predict.py; None = no seabed in this ping
    if not (col[-10:] <= LO + 1e-3).all():
        return None
    s = np.where(col > -45)[0]
    if not len(s):
        return None
    top, gap = int(s[-1]), 0
    while top > 0 and gap <= 3:
        top -= 1
        gap = 0 if col[top] > -60 else gap + 1
    return top + gap


def bottom_track(sv, max_gap=300):
    """Seabed top row per ping. On a real boat this comes from the sonar's depth reading (e.g. NMEA DBT).
    Pings where our rule fails are filled from the nearest detected ping (max_gap pings away)."""
    W = sv.shape[1]
    raw = np.array([np.nan if (v := seabed_top(sv[:, c])) is None else v for c in range(W)], float)
    ok = np.where(np.isfinite(raw))[0]
    if len(ok) < 0.2 * W:
        return np.full(W, np.nan)           # no seabed in this recording
    idx = np.arange(W)
    filled = np.interp(idx, ok, raw[ok])
    near = np.abs(idx[:, None] - ok[None, :]).min(1) <= max_gap
    b = np.where(near, filled, np.nan)
    pad = np.pad(b, 7, mode="edge")
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmedian(np.lib.stride_tricks.sliding_window_view(pad, 15), axis=1)


def gps(t):                                 # SIMULATED boat track (1 ping/s, ~2.5 m/s)
    d = t * 2.5
    north, east = d * .6 + 40 * np.cos(t / 150), d * .8 + 40 * np.sin(t / 120)
    return 24.60 + north / 111320, -110.40 + east / (111320 * np.cos(np.radians(24.6)))


class Patch(BaseModel):
    patch: list[list[float]]                # 64 x 64 echo values in dB


@app.get("/api/model")
def model_info():
    return {"model": "RandomForest (6 features)", "patch": P, "threshold": THR, "db_range": [LO, HI],
            "trained_on": "Gulf of California 38 kHz echograms (public), depth-matched", "status": "prototype"}


@app.post("/predict")
def predict(p: Patch):
    a = np.array(p.patch, np.float32)
    if a.shape != (P, P):
        raise HTTPException(422, f"patch must be {P}x{P}")
    pr = float(rf.predict_proba(feats(a[None]))[0, 1])
    return {"probability": round(pr, 3), "fish": pr >= THR, "threshold": THR}


@app.get("/stream")
async def stream(trawl: int = 17, speed: float = 20):
    if f"sv_{trawl}" not in REPLAY.files:
        raise HTTPException(404, "unknown trawl")
    sv = REPLAY[f"sv_{trawl}"].astype(np.float32)
    H, W = sv.shape
    centers = np.arange(P // 2, H - P // 2 + 1, 16)
    bottom = bottom_track(sv)

    async def gen():
        for t in range(P, W + 1, STEP):
            t0 = time.perf_counter()
            win = np.stack([sv[r - P // 2:r + P // 2, t - P:t] for r in centers])
            probs = (await asyncio.to_thread(lambda: rf.predict_proba(feats(win))[:, 1]))
            blk = bottom[t - P:t]
            top = int(np.nanmin(blk)) if np.isfinite(blk).any() else None
            if top is not None:
                probs = np.where(centers + P // 2 < top - 10, probs, 0.0)
            i = int(probs.argmax())
            hit = centers[probs >= THR]
            row = int(np.median(hit)) if len(hit) else int(centers[i])
            lat, lon = gps(t - P // 2)
            cols = (np.clip((sv[:, t - STEP:t] - LO) / (HI - LO), 0, 1) * 255).astype(np.uint8).T
            ev = {"t": t, "total": W, "H": H, "step": STEP, "cols": base64.b64encode(cols.tobytes()).decode(),
                  "centers": centers.tolist(), "probs": np.round(probs, 2).tolist(), "pmax": round(float(probs[i]), 3),
                  "row": row, "fish": bool(probs[i] >= THR), "thr": THR, "bottom": top,
                  "lat": round(float(lat), 6), "lon": round(float(lon), 6),
                  "ms": round((time.perf_counter() - t0) * 1000, 1)}
            yield f"data: {json.dumps(ev)}\n\n"
            await asyncio.sleep(max(0.0, STEP / speed - (time.perf_counter() - t0)))
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/")
def index():
    return FileResponse(BASE / "dashboard.html")
