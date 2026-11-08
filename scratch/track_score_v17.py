import subprocess
import time
import re
from datetime import datetime

SUB_ID = "57580018"

print(f"[{datetime.now().strftime('%H:%M:%S')}] Bắt đầu theo dõi chấm điểm Submission {SUB_ID}...", flush=True)

attempt = 0
while True:
    attempt += 1
    try:
        res = subprocess.run(["kaggle", "competitions", "submissions", "-c", "name-that-llm"], capture_output=True, text=True, timeout=30)
        lines = res.stdout.splitlines()
        for line in lines:
            if SUB_ID in line:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] (Lần {attempt}) Trạng thái: {line.strip()}", flush=True)
                if "SubmissionStatus.COMPLETE" in line:
                    parts = re.split(r'\s{2,}', line.strip())
                    print(f"\n==========================================", flush=True)
                    print(f"🎉 CHẤM ĐIỂM HOÀN TẤT! SCORE: {parts[-2] if len(parts)>=2 else 'XEM TRÊN LEADERBOARD'}", flush=True)
                    print(f"==========================================", flush=True)
                    exit(0)
                elif "SubmissionStatus.ERROR" in line:
                    print(f"\n[!] LỖI CHẤM ĐIỂM!", flush=True)
                    exit(1)
                break
    except Exception as e:
        print(f"Error checking submissions: {e}", flush=True)
    time.sleep(20)
