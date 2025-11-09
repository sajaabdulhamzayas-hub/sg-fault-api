# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load

import time
START_TIME = time.time()

# تحميل الموديل المدرَّب على الميزات المشتقة
model = load("best_model_fe.joblib")

app = FastAPI(title="SmartGrid Fault Detection (RF + FE)")

class Sample(BaseModel):
    # 6 قراءات خام: [Va, Vb, Vc, Ia, Ib, Ic]
    x: list


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "sg-fault-api",
        "uptime_s": int(time.time() - START_TIME)
    }

@app.get("/version")
def version():
    return {
        "service_version": "v1.0.2",  # ← غيّري الرقم هنا
        "model": {
            "name": "rf_engineered_fe",
            "version": "2025-11-09",
            "accuracy_test": 0.9949
        }
    }


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

    # ما عدنا نافذة زمنية هنا → std = 0
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


@app.post("/predict")
def predict(sample: Sample):
    assert len(sample.x) == 6, "Send exactly 6 values: [Va,Vb,Vc,Ia,Ib,Ic]"
    feats = build_features_from_raw(sample.x)
    pred = model.predict(feats)[0]
    probs = model.predict_proba(feats)[0].tolist() if hasattr(model, "predict_proba") else None
    return {"pred_class": str(pred), "probs": probs}
