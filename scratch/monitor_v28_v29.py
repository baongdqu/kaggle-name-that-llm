import subprocess
import time
import sys
from datetime import datetime

LOG_FILE = "scratch/monitor_v28_v29.log"

def log(msg):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now_str}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === MONITOR V28 & AUTO-SUBMIT V29 LAUNCHED ===\n")

# Bước 1: Giám sát V29 kernel commit
v29_submitted = False
v29_kernel_id = "baonguyenduyquoc/name-that-llm-champion-v29-frontier-orthogonal"

log("Bắt đầu giám sát V29 kernel commit...")
for i in range(60): # chờ tối đa 30 phút
    res = subprocess.run(["kaggle", "kernels", "status", v29_kernel_id], capture_output=True, text=True)
    status_str = res.stdout.strip()
    log(f"Trạng thái V29 kernel: {status_str}")
    
    if "KernelWorkerStatus.COMPLETE" in status_str:
        log(">>> V29 KERNEL COMMIT COMPLETE THÀNH CÔNG! Bắt đầu nộp bài...")
        submit_cmd = [
            "kaggle", "competitions", "submit",
            "-c", "name-that-llm",
            "-f", "submission.parquet",
            "-k", v29_kernel_id,
            "-v", "2",
            "-m", "Champion V29 Frontier Orthogonal (ModernBERT + Two-Stage LoRA24 Fold-Ensemble + Gauss + 120k Pool, Proven LB 0.651)"
        ]
        sub_res = subprocess.run(submit_cmd, capture_output=True, text=True)
        log(f"Kết quả nộp V29:\nSTDOUT: {sub_res.stdout}\nSTDERR: {sub_res.stderr}")
        if sub_res.returncode == 0 or "remaining today" in sub_res.stdout:
            log(">>> V29 ĐÃ ĐƯỢC NỘP THÀNH CÔNG VÀO COMPETITION!")
            v29_submitted = True
        break
    elif "KernelWorkerStatus.ERROR" in status_str:
        log("!!! V29 KERNEL GẶP LỖI (ERROR)!")
        break
    time.sleep(30)

log(">>> Bắt đầu giám sát tiến trình chấm điểm ẩn trên Leaderboard...")
# Bước 2: Giám sát bảng submissions
for cycle in range(360): # 6 tiếng
    res = subprocess.run(["kaggle", "competitions", "submissions", "-c", "name-that-llm"], capture_output=True, text=True)
    lines = res.stdout.splitlines()
    header = lines[0] if lines else ""
    log(f"--- Chu kỳ kiểm tra {cycle + 1} ---")
    for l in lines[1:8]:
        log(l)
    
    # Kiểm tra xem V28 đã hoàn tất chưa
    v28_done = any("57656311" in l and "COMPLETE" in l for l in lines)
    if v28_done:
        log(">>> V28 (57656311) ĐÃ CHẤM XONG!")
    
    time.sleep(60)
