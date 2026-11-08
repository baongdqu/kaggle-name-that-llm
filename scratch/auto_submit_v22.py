import subprocess
import time
import os
import datetime

SUBMIT_CMD = [
    "kaggle", "competitions", "submit", "name-that-llm",
    "-k", "baonguyenduyquoc/name-that-llm-champion-v22-pre-sinkhorn-sovereign",
    "-f", "submission.parquet",
    "-v", "1",
    "-m", "Champion V22 Pre-Sinkhorn Sovereign (Pre-Sinkhorn Micro-Gates + Terminal Sinkhorn Purity + SDPA)"
]

LOG_FILE = "scratch/auto_submit_v22.log"

def log(msg):
    ts = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

log("=== KHỞI ĐỘNG HỆ THỐNG ĐÓN ĐẦU RESET QUOTA (00:00 UTC) CHO V22 ===")

while True:
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    log(f"[{now_utc.strftime('%H:%M:%S')} UTC] Đang kiểm tra quota & gửi bài...")
    res = subprocess.run(SUBMIT_CMD, capture_output=True, text=True)
    out = (res.stdout + "\n" + res.stderr).strip()
    log(f"Kaggle response:\n{out}")
    
    if "Successfully submitted" in out or "submissions remaining" in out:
        log("🎉 NỘP BÀI THÀNH CÔNG CHAMPION V22 LÊN KAGGLE!")
        # Lấy submission id
        sub_list = subprocess.run(["kaggle", "competitions", "submissions", "-c", "name-that-llm"], capture_output=True, text=True)
        log(f"Danh sách bài nộp mới nhất:\n{sub_list.stdout[:600]}")
        break
    
    log("Chưa tới giờ reset (hoặc Kaggle chưa mở khóa quota). Chờ 30 giây...")
    time.sleep(30)
