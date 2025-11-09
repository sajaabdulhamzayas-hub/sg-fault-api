import time, json, requests
from datetime import datetime, timezone

URL = "https://sg-fault-api.onrender.com/predict"
DEVICE_ID = "esp32-sim-01"

def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')

def sample():
    # قياسات تجريبية ثابتة/واقعية بسيطة
    Va, Vb, Vc = 230.5, 229.9, 231.1
    Ia, Ib, Ic = 3.21, 3.18, 3.27
    return [Va, Vb, Vc, Ia, Ib, Ic]

if __name__ == "__main__":
    print(f"POSTing to {URL} as {DEVICE_ID} (Ctrl+C to stop)")
    while True:
        payload = {"device_id": DEVICE_ID, "ts": now_iso(), "x": sample()}
        try:
            r = requests.post(URL, json=payload, timeout=10)
            print(f"{payload['ts']} -> code={r.status_code} resp={r.text[:120]}")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(1)  # واحدة بالثانية
