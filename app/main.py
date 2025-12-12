# app/main.py
import os
import time
import warnings
from datetime import datetime, timezone
from typing import Optional, List

import numpy as np
from joblib import load
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# اخفاء تحذير sklearn الخاص بـ feature names (مو خطأ)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names*",
    category=UserWarning,
)

# ==========================================
# MongoDB setup
# ==========================================
try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

MONGO_URI = os.getenv("MONGO_URI", "").strip()
MONGO_DB = os.getenv("MONGO_DB", "sg")
MONGO_COLL = os.getenv("MONGO_COLL", "readings")

mongo_ok = False
coll = None
coll_alerts = None

if MongoClient and MONGO_URI:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        db = client[MONGO_DB]
        coll = db[MONGO_COLL]
        coll_alerts = db["alerts"]
        client.admin.command("ping")
        mongo_ok = True
    except Exception:
        coll = None
        coll_alerts = None
        mongo_ok = False

# ==========================================
# ML model
# ==========================================
START_TIME = time.time()
model = load("best_model_fe.joblib")

app = FastAPI(title="SmartGrid Fault Detection", version="1.0.3")


class Sample(BaseModel):
    x: List[float]
    device_id: Optional[str] = None
    ts: Optional[str] = None


# ==========================================
# Feature engineering (مطابق للتدريب قدر الإمكان)
# ==========================================
def build_features_from_raw(arr6: List[float]) -> np.ndarray:
    Va, Vb, Vc, Ia, Ib, Ic = arr6

    Va_abs, Vb_abs, Vc_abs = abs(Va), abs(Vb), abs(Vc)
    Ia_abs, Ib_abs, Ic_abs = abs(Ia), abs(Ib), abs(Ic)

    Vab, Vbc, Vca = Va - Vb, Vb - Vc, Vc - Va
    Iab, Ibc, Ica = Ia - Ib, Ib - Ic, Ic - Ia

    V_sum = Va + Vb + Vc
    I_sum = Ia + Ib + Ic

    V_rss = float(np.sqrt(Va**2 + Vb**2 + Vc**2))
    I_rss = float(np.sqrt(Ia**2 + Ib**2 + Ic**2))

    V_mean_abs = (Va_abs + Vb_abs + Vc_abs) / 3.0
    I_mean_abs = (Ia_abs + Ib_abs + Ic_abs) / 3.0

    # IMPORTANT: مثل التدريب (std على الثلاث فازات)
    V_std = float(np.std([Va, Vb, Vc], ddof=0))
    I_std = float(np.std([Ia, Ib, Ic], ddof=0))

    eps = 1e-9
    V_max = max(Va_abs, Vb_abs, Vc_abs) + eps
    V_min = min(Va_abs, Vb_abs, Vc_abs) + eps
    I_max = max(Ia_abs, Ib_abs, Ic_abs) + eps
    I_min = min(Ia_abs, Ib_abs, Ic_abs) + eps

    V_imbalance = V_max / V_min
    I_imbalance = I_max / I_min

    Sa = Va_abs * Ia_abs
    Sb = Vb_abs * Ib_abs
    Sc = Vc_abs * Ic_abs
    S_total = Sa + Sb + Sc

    I_total = Ia_abs + Ib_abs + Ic_abs + eps
    Ia_share = Ia_abs / I_total
    Ib_share = Ib_abs / I_total
    Ic_share = Ic_abs / I_total

    V_total = Va_abs + Vb_abs + Vc_abs + eps
    Va_share = Va_abs / V_total
    Vb_share = Vb_abs / V_total
    Vc_share = Vc_abs / V_total

    feats = [
        Va_abs, Vb_abs, Vc_abs, Ia_abs, Ib_abs, Ic_abs,
        Vab, Vbc, Vca, Iab, Ibc, Ica,
        V_sum, I_sum, V_rss, I_rss, V_mean_abs, I_mean_abs,
        V_std, I_std, V_imbalance, I_imbalance,
        Sa, Sb, Sc, S_total,
        Ia_share, Ib_share, Ic_share, Va_share, Vb_share, Vc_share,
    ]
    return np.array(feats, dtype=np.float32).reshape(1, -1)


def save_alert(device_id: str, phase: str, reason: str, value: float):
    if coll_alerts is None:
        return
    try:
        coll_alerts.insert_one(
            {
                "ts": datetime.utcnow().isoformat(),
                "device_id": device_id,
                "phase": phase,
                "reason": reason,
                "value": float(value),
                "level": "CRITICAL",
            }
        )
    except Exception:
        pass


# ==========================================
# Endpoints
# ==========================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "sg-fault-api",
        "uptime_s": int(time.time() - START_TIME),
        "mongo": "ok" if mongo_ok else "disabled",
    }


@app.get("/version")
def version():
    return {
        "service_version": app.version,
        "model": {
            "name": "ExtraTrees_engineered_fe",
            "version": "2025-11-07",
            "accuracy_test": 0.9949,
        },
    }


@app.post("/predict")
def predict(sample: Sample):
    # 422 واضح بدل assert/500
    if sample.x is None or len(sample.x) != 6:
        raise HTTPException(
            status_code=422,
            detail="x must contain exactly 6 values in order: [Va, Vb, Vc, Ia, Ib, Ic]",
        )

    feats = build_features_from_raw(sample.x)
    pred = model.predict(feats)[0]
    probs = model.predict_proba(feats)[0].tolist()

    Va, Vb, Vc, Ia, Ib, Ic = sample.x
    device_id = sample.device_id or "unknown"

    alerts_created = 0
    try:
        if Va > 260:
            save_alert(device_id, "A", "OVER_VOLTAGE", Va)
            alerts_created += 1
        if Ia > 4.0:
            save_alert(device_id, "A", "OVERCURRENT", Ia)
            alerts_created += 1
    except Exception:
        pass

    ts_server = datetime.now(timezone.utc).isoformat()

    doc = {
        "ts": ts_server,
        "device_id": device_id,
        "raw": sample.x,
        "prediction": str(pred),
        "probs": probs,
    }

    saved = False
    if coll is not None:
        try:
            coll.insert_one(doc)
            saved = True
        except Exception:
            saved = False

    return {
        "pred_class": str(pred),
        "probs": probs,
        "saved": saved,
        "alerts_created": alerts_created,
        "ts": ts_server,
    }


@app.get("/last_readings")
def last_readings(limit: int = 200, device_id: Optional[str] = None):
    if coll is None:
        return {"mongo": "disabled", "items": []}

    query = {}
    if device_id:
        query["device_id"] = device_id

    try:
        cursor = (
            coll.find(query, {"_id": 0, "ts": 1, "device_id": 1, "raw": 1, "prediction": 1, "probs": 1})
            .sort("_id", -1)
            .limit(int(limit))
        )
        items = list(cursor)
        return {"mongo": "ok", "count": len(items), "items": items}
    except Exception as e:
        return {"mongo": "error", "error": str(e), "items": []}


@app.get("/alerts")
def get_alerts(limit: int = 200, device_id: Optional[str] = None):
    if coll_alerts is None:
        return {"mongo": "disabled", "items": []}

    query = {}
    if device_id:
        query["device_id"] = device_id

    try:
        cur = (
            coll_alerts.find(query, {"_id": 0})
            .sort("_id", -1)
            .limit(int(limit))
        )
        items = list(cur)
        return {"mongo": "ok", "count": len(items), "items": items}
    except Exception as e:
        return {"mongo": "error", "error": str(e), "items": []}
