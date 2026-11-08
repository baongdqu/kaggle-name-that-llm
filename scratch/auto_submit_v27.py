import subprocess
import time
import sys
import os
from datetime import datetime, timezone

KERNEL_ID = "baonguyenduyquoc/name-that-llm-champion-v27-sovereign-dqa"
COMPETITION = "name-that-llm"
MESSAGE = "Champion V27 Sovereign DQA (Robust ATLAS Likelihood + Full Open Ambiguity Scope + Golden Pool_v4)"
LOG_FILE = "scratch/monitor_v27.log"

def log(msg):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now_str}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# Clear or start new log
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === KHỞI ĐỘNG AUTO-SUBMIT & MONITOR CHO CHAMPION V27 ===\n")

# Bước 1: Đảm bảo Kernel commit đã hoàn tất (KernelWorkerStatus.COMPLETE)
log("Bước 1: Kiểm tra trạng thái kernel commit...")
res = subprocess.run(
    ["kaggle", "kernels", "status", KERNEL_ID],
    capture_output=True,
    text=True
)
status_text = res.stdout.strip()
log(f"Trạng thái kernel: {status_text}")
if "KernelWorkerStatus.COMPLETE" not in status_text:
    log("Kernel chưa COMPLETE, thoát!")
    sys.exit(1)
log(">>> Kernel đã hoàn tất commit xuất sắc (KernelWorkerStatus.COMPLETE)!")

# Bước 2: Chờ chính xác đến 07:00:05 GMT+7 (2026-10-26 00:00:05 UTC)
target_utc = datetime(2026, 10, 26, 0, 0, 5, tzinfo=timezone.utc)
log(f"Bước 2: Chờ Quota reset vào đúng 07:00:05 GMT+7 ({target_utc.strftime('%Y-%m-%d %H:%M:%S UTC')})...")

while True:
    now_utc = datetime.now(timezone.utc)
    rem_sec = (target_utc - now_utc).total_seconds()
    if rem_sec <= 0:
        log(f">>> ĐÃ CHÍNH THỨC SANG NGÀY MỚI 2026-10-26 UTC (Hiện tại: {now_utc.strftime('%H:%M:%S UTC')} / {datetime.now().strftime('%H:%M:%S')} Local). Quota 5 lượt nộp đã reset!")
        break
    
    log(f"Chờ Quota reset: Còn {rem_sec:.0f}s ({rem_sec/60:.1f} phút)... [Local: {datetime.now().strftime('%H:%M:%S')}]")
    sleep_time = min(20, max(2, int(rem_sec / 2)))
    time.sleep(sleep_time)

# Bước 3: Nộp bài lên Kaggle Competition
log("Bước 3: Thực hiện nộp bài lên Kaggle Competition...")
submit_cmd = [
    "kaggle", "competitions", "submit",
    "-c", COMPETITION,
    "-f", "submission.parquet",
    "-k", KERNEL_ID,
    "-v", "1",
    "-m", MESSAGE
]

sub_success = False
for attempt in range(6):
    try:
        log(f"Thực hiện lệnh nộp bài (lần thử {attempt + 1}): {' '.join(submit_cmd)}")
        res = subprocess.run(submit_cmd, capture_output=True, text=True)
        log(f"Output nộp bài:\n{res.stdout}\n{res.stderr}")
        if "Successfully submitted" in res.stdout or res.returncode == 0:
            log(">>> NỘP BÀI THÀNH CÔNG RỰC RỠ!")
            sub_success = True
            break
        else:
            log(f"Chưa thành công, đợi 5 giây thử lại... (lỗi: {res.stderr.strip()})")
            time.sleep(5)
    except Exception as e:
        log(f"Exception khi nộp bài: {e}")
        time.sleep(5)

if not sub_success:
    log("!!! KHÔNG THỂ NỘP BÀI TỰ ĐỘNG SAU 6 LẦN THỬ.")
    sys.exit(1)

# Bước 4: Lấy mã tham chiếu Ref ID và bắt đầu giám sát chấm điểm ẩn
log("Bước 4: Bắt đầu chu trình giám sát chấm điểm ẩn trên 4,000 dòng test...")
ref_id = None
start_scoring_time = time.time()

while True:
    try:
        res = subprocess.run(
            ["kaggle", "competitions", "submissions", "-c", COMPETITION],
            capture_output=True,
            text=True,
            check=True
        )
        lines = res.stdout.strip().splitlines()
        for line in lines[2:8]:
            if "Champion V27" in line:
                parts = line.split()
                cur_ref = parts[0]
                status = "PENDING"
                if "COMPLETE" in line:
                    status = "COMPLETE"
                elif "ERROR" in line:
                    status = "ERROR"
                
                score = parts[-2] if status == "COMPLETE" else "SCORING"
                elapsed_min = (time.time() - start_scoring_time) / 60
                log(f"[{elapsed_min:.1f}m] Ref: {cur_ref} | Status: {status} | Score: {score}")

                if status == "COMPLETE":
                    log(f"🎉🎉🎉 CHẤM ĐIỂM HOÀN TẤT! Điểm Public Score của Champion V27: {score}")
                    sys.exit(0)
                elif status == "ERROR":
                    log("!!! BÀI NỘP BỊ LỖI CHẤM ĐIỂM (ERROR).")
                    sys.exit(1)
                break
    except Exception as e:
        log(f"Lỗi truy vấn submissions: {e}")
    time.sleep(60)
