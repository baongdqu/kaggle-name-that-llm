import time
import subprocess
import datetime
import os

SUB_ID = "57623685"
LOG_FILE = "scratch/monitor_v24.log"

def log(msg):
    ts = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

log(f"=== BẮT ĐẦU GIÁM SÁT BÀI NỘP CHAMPION V24 CLEAN SOVEREIGN (Ref: {SUB_ID}) ===")

start_time = time.time()
while True:
    elapsed_min = (time.time() - start_time) / 60.0
    try:
        res = subprocess.run(["kaggle", "competitions", "submissions", "-c", "name-that-llm"], capture_output=True, text=True, timeout=30)
        out = res.stdout
    except Exception as e:
        log(f"Lỗi truy vấn Kaggle: {e}")
        time.sleep(60)
        continue

    for line in out.splitlines():
        if SUB_ID in line:
            status = "PENDING"
            if "COMPLETE" in line:
                status = "COMPLETE"
            elif "ERROR" in line:
                status = "ERROR"
            
            log(f"[{elapsed_min:.1f}m] Status: {status} | Full: {line}")
            
            if status in ["COMPLETE", "ERROR"]:
                log(f"=== KẾT THÚC GIÁM SÁT V24: {status} sau {elapsed_min:.1f} phút ===")
                exit(0)
            break
    time.sleep(60)
