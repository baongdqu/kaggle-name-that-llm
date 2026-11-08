import subprocess
import time
import os
import sys
from datetime import datetime

SLUG = "baonguyenduyquoc/name-that-llm-champion-v16-apex-sovereign"
MSG = "Champion V16 Apex Sovereign 0.630+ (Two-Stage 14B + 35k Pool SVC + Injective Decoding)"
LOG_FILE = "scratch/monitor_v16.log"
VERSION = "1"

def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def get_kernel_status():
    try:
        res = subprocess.run(["kaggle", "kernels", "status", SLUG], capture_output=True, text=True, timeout=30)
        return res.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"

def submit_kernel(version=VERSION):
    cmd = ["kaggle", "competitions", "submit", "name-that-llm", "-k", SLUG, "-f", "submission.parquet", "-v", str(version), "-m", MSG]
    log(f"Submitting: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        log(f"Submit stdout: {res.stdout.strip()}")
        if res.stderr:
            log(f"Submit stderr: {res.stderr.strip()}")
        return res.returncode == 0
    except Exception as e:
        log(f"Submit exception: {e}")
        return False

def get_submissions():
    try:
        res = subprocess.run(["kaggle", "competitions", "submissions", "name-that-llm"], capture_output=True, text=True, timeout=30)
        return res.stdout
    except Exception as e:
        return f"ERROR: {e}"

def main():
    log(f"=== V16 APEX SOVEREIGN MONITOR STARTED (VERSION {VERSION}) ===")
    submitted = False
    check_interval = 60 # seconds

    while True:
        status_line = get_kernel_status()
        log(f"Kernel Status: {status_line}")

        if "KernelWorkerStatus.COMPLETE" in status_line:
            log("KERNEL RUN COMPLETED SUCCESSFULLY!")
            
            # Download kernel outputs
            os.makedirs("champion_v16_output", exist_ok=True)
            log("Downloading kernel outputs...")
            subprocess.run(["kaggle", "kernels", "output", SLUG, "-p", "champion_v16_output"], timeout=120)
            
            # Submit to competition
            if not submitted:
                log(f"Triggering competition submission for Version {VERSION}...")
                ok = submit_kernel(VERSION)
                if ok:
                    submitted = True
                    log("Submission submitted successfully! Now waiting for scoring...")
                else:
                    log("Submission failed, will retry next cycle.")

        elif "KernelWorkerStatus.FAILED" in status_line or "KernelWorkerStatus.CANCELLED" in status_line:
            log(f"KERNEL FAILED OR CANCELLED: {status_line}")
            os.makedirs("champion_v16_output", exist_ok=True)
            subprocess.run(["kaggle", "kernels", "output", SLUG, "-p", "champion_v16_output"], timeout=120)
            break

        if submitted:
            sub_output = get_submissions()
            lines = [l for l in sub_output.splitlines() if "Champion V16" in l or "Apex Sovereign" in l]
            if lines:
                log(f"Submission Entry: {lines[0]}")
                if "complete" in lines[0].lower():
                    log(f"FINAL SUBMISSION RESULT:\n{lines[0]}")
                    log("=== MONITORING COMPLETE ===")
                    break
            else:
                log("Submission not yet visible in submissions list, waiting...")

        time.sleep(check_interval)

if __name__ == "__main__":
    main()
