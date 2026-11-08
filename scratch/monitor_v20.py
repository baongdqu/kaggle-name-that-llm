import time
import subprocess
import datetime
import os

SUB_ID = "57595989"
LOG_FILE = "scratch/monitor_v20.log"

def log(msg):
    ts = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

log(f"=== Bắt đầu giám sát bài nộp Champion V20 (Ref: {SUB_ID}) ===")

start_time = time.time()
while True:
    elapsed_min = (time.time() - start_time) / 60.0
    res = subprocess.run(["kaggle", "competitions", "submissions", "-c", "name-that-llm"], capture_output=True, text=True)
    out = res.stdout
    for line in out.splitlines():
        if SUB_ID in line:
            status = "PENDING"
            if "COMPLETE" in line:
                status = "COMPLETE"
            elif "ERROR" in line:
                status = "ERROR"
            
            # Extract scores if available
            parts = [p.strip() for p in line.split() if p.strip()]
            log(f"[{elapsed_min:.1f}m] Status: {status} | Full: {line}")
            
            if status in ["COMPLETE", "ERROR"]:
                log(f"=== KẾT THÚC GIÁM SÁT V20: {status} sau {elapsed_min:.1f} phút ===")
                exit(0)
            break
    time.sleep(60)
