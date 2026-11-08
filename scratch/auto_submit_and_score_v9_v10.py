import subprocess
import time
import json
import os
import sys
from datetime import datetime

V9_SLUG = "baonguyenduyquoc/name-that-llm-champion-v9-pseudo-refit"
V10_SLUG = "baonguyenduyquoc/name-that-llm-champion-v10-balanced-refit"

V9_MSG = "Champion V9 Transductive Pseudo-Refit (Global Margin)"
V10_MSG = "Champion V10 Transductive Pseudo-Refit (Class-Balanced Margin)"

LOG_FILE = "scratch/auto_monitor.log"
RESULTS_FILE = "scratch/v9_v10_results.json"

def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def get_kernel_status(slug):
    try:
        res = subprocess.run(["kaggle", "kernels", "status", slug], capture_output=True, text=True, timeout=30)
        return res.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"

def submit_kernel(slug, message):
    cmd = ["kaggle", "competitions", "submit", "name-that-llm", "-k", slug, "-f", "submission.parquet", "-m", message]
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
    log("=== AUTO SUBMIT & SCORE DAEMON STARTED (V9 & V10) ===")
    v9_submitted = False
    v10_submitted = False
    v9_scored = False
    v10_scored = False
    v9_score = None
    v10_score = None
    
    check_interval = 60 # check every 60 seconds

    while True:
        # 1. Check V9 kernel status
        if not v9_submitted:
            st9 = get_kernel_status(V9_SLUG)
            log(f"V9 kernel status: {st9}")
            if "COMPLETE" in st9:
                log(">>> V9 Kernel is COMPLETE! Triggering competition submission...")
                ok = submit_kernel(V9_SLUG, V9_MSG)
                if ok:
                    v9_submitted = True
                    log(">>> V9 submitted successfully!")
            elif "ERROR" in st9 or "CANCEL" in st9:
                log(f"!!! V9 Kernel encountered issue: {st9}")

        # 2. Check V10 kernel status
        if not v10_submitted:
            st10 = get_kernel_status(V10_SLUG)
            log(f"V10 kernel status: {st10}")
            if "COMPLETE" in st10:
                log(">>> V10 Kernel is COMPLETE! Triggering competition submission...")
                ok = submit_kernel(V10_SLUG, V10_MSG)
                if ok:
                    v10_submitted = True
                    log(">>> V10 submitted successfully!")
            elif "ERROR" in st10 or "CANCEL" in st10:
                log(f"!!! V10 Kernel encountered issue: {st10}")

        # 3. Check scoring if any submitted
        if v9_submitted or v10_submitted:
            sub_output = get_submissions()
            log("Current Submissions Table:")
            for line in sub_output.split("\n")[:12]:
                if line.strip():
                    log("  " + line)
            
            # Parse scores for our recent submissions
            for line in sub_output.split("\n"):
                if V9_MSG in line or "V9" in line:
                    parts = [p.strip() for p in line.split() if p.strip()]
                    for p in parts:
                        try:
                            val = float(p)
                            if 0.4 < val < 0.9:
                                v9_score = val
                                v9_scored = True
                        except ValueError:
                            pass
                if V10_MSG in line or "V10" in line:
                    parts = [p.strip() for p in line.split() if p.strip()]
                    for p in parts:
                        try:
                            val = float(p)
                            if 0.4 < val < 0.9:
                                v10_score = val
                                v10_scored = True
                        except ValueError:
                            pass

            if (v9_scored or not v9_submitted) and (v10_scored or not v10_submitted) and (v9_submitted and v10_submitted):
                log("==================================================")
                log("🎉 BOTH V9 AND V10 SUBMISSIONS SCORED!")
                log(f"V9 Score: {v9_score}")
                log(f"V10 Score: {v10_score}")
                log("==================================================")
                
                results = {
                    "v9_slug": V9_SLUG,
                    "v9_score": v9_score,
                    "v10_slug": V10_SLUG,
                    "v10_score": v10_score,
                    "timestamp": datetime.now().isoformat()
                }
                with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                break

        time.sleep(check_interval)

if __name__ == "__main__":
    main()
