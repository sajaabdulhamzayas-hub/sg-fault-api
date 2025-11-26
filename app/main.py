# app/main.py
import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load

# ====== إعداد MongoDB ======
try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None  # لو المكتبة غير منصّبة بعد أثناء build

MONGO_URI  = os.getenv("MONGO_URI", "").strip()
MONGO_DB   = os.getenv("MONGO_DB", "sg")
MONGO_COLL = os.getenv("MONGO_COLL", "readings")

mongo_ok = False
coll = None

from typing import Optional  # ضيفي هذا في أعلى الملف لو بعده مو مضاف
from fastapi import Query

if MongoClient and MONGO_URI:
    try:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000)
        _db = _client[MONGO_DB]
        coll = _db[MONGO_COLL]
        # ping سريع
        _client.admin.command("ping")
        mongo_ok = True
    except Exception:
        mongo_ok = False
        coll = None

# ====== نموذج التصنيف ======
START_TIME = time.time()
model = load("best_model_fe.joblib")

app = FastAPI(title="SmartGrid Fault Detection (RF + FE)", version="1.0.2")

class Sample(BaseModel):
    # 6 قراءات خام: [Va, Vb, Vc, Ia, Ib, Ic]
    x: list
    device_id: str | None = None
    ts: str | None = None  # ISO-8601 (اختياري)

def build_features_from_raw(arr6):
    Va, Vb, Vc, Ia, Ib, Ic = arr6
    Va_abs, Vb_abs, Vc_abs = abs(Va), abs(Vb), abs(Vc)
    Ia_abs, Ib_abs, Ic_abs = abs(Ia), abs(Ib), abs(Ic)

    Vab, Vbc, Vca = Va - Vb, Vb - Vc, Vc - Va
    Iab, Ibc, Ica = Ia - Ib, Ib - Ic, Ic - Ia

    V_sum = Va + Vb + Vc
    I_sum = Ia + Ib + Ic

    V_rss = (Va**2 + Vb**2 + Vc**2) ** 0.5
    I_rss = (Ia**2 + Ib**2 + Ic**2) ** 0.5

    V_mean_abs = (Va_abs + Vb_abs + Vc_abs) / 3.0
    I_mean_abs = (Ia_abs + Ib_abs + Ic_abs) / 3.0

    V_std = 0.0
    I_std = 0.0

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
    Ia_share, Ib_share, Ic_share = Ia_abs / I_total, Ib_abs / I_total, Ic_abs / I_total

    V_total = Va_abs + Vb_abs + Vc_abs + eps
    Va_share, Vb_share, Vc_share = Va_abs / V_total, Vb_abs / V_total, Vc_abs / V_total

    feats = [
        Va_abs, Vb_abs, Vc_abs, Ia_abs, Ib_abs, Ic_abs,
        Vab, Vbc, Vca, Iab, Ibc, Ica,
        V_sum, I_sum, V_rss, I_rss, V_mean_abs, I_mean_abs,
        V_std, I_std, V_imbalance, I_imbalance,
        Sa, Sb, Sc, S_total,
        Ia_share, Ib_share, Ic_share, Va_share, Vb_share, Vc_share
    ]
    return np.array(feats, dtype=np.float32).reshape(1, -1)

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "sg-fault-api",
        "uptime_s": int(time.time() - START_TIME),
        "mongo": "ok" if mongo_ok else "disabled"
    }

@app.get("/version")
def version():
    return {
        "service_version": app.version,
        "model": {
            "name": "rf_engineered_fe",
            "version": "2025-11-07",
            "accuracy_test": 0.9949
        }
    }

@app.get("/last_readings")
def last_readings(limit: int = Query(100, ge=1, le=1000)):
    """
    إرجاع آخر القراءات المخزّنة من MongoDB
    """
    if coll is None:
        return {"error": "mongo_disabled", "items": []}

    try:
        cursor = (
            coll.find({})
            .sort("ts", -1)      # الأحدث أولاً
            .limit(limit)
        )
        docs = list(cursor)

        # تحويل ObjectId إلى نص
        for d in docs:
            d["_id"] = str(d.get("_id", ""))

        return {
            "count": len(docs),
            "items": docs,
        }
    except Exception as e:
        return {
            "error": f"mongo_error: {e.__class__.__name__}",
            "items": [],
        }


@app.post("/predict")
def predict(sample: Sample):
    assert len(sample.x) == 6, "Send exactly 6 values: [Va,Vb,Vc,Ia,Ib,Ic]"
    feats = build_features_from_raw(sample.x)
    pred = model.predict(feats)[0]
    probs = model.predict_proba(feats)[0].tolist() if hasattr(model, "predict_proba") else None

    # تحضير وثيقة التخزين
    doc = {
        "ts": sample.ts or datetime.now(timezone.utc).isoformat(),
        "device_id": sample.device_id or "unknown",
        "raw": sample.x,
        "features": feats.flatten().tolist(),
        "prediction": str(pred),
        "probs": probs,
        "service_version": app.version,
    }

    saved = False
    inserted_id = None
    if coll is not None:
        try:
            res = coll.insert_one(doc)
            saved = True
            inserted_id = str(res.inserted_id)
        except Exception:
            saved = False

    return {
        "pred_class": str(pred),
        "probs": probs,
        "saved": saved,
        "id": inserted_id
    }
