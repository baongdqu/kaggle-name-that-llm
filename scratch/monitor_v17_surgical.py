import subprocess
import time
import os
import sys
from datetime import datetime

SLUG = "baonguyenduyquoc/name-that-llm-champion-v17-apex-crown"
MSG = "Champion V17 Surgical Apex (Hungarian Injective Matching + Margin-Gated Sinkhorn + 0.603 Proven Core)"
LOG_FILE = "scratch/monitor_v17_surgical.log"
VERSION = "7"

def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def get_kernel_status():
    try:
        res = subprocess.run(["kaggle", "kernels", "status", SLUG], capture_output=True, text=True, timeout=60)
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
    log(f"=== V17 SURGICAL APEX AUTO-MONITOR & SUBMISSION DAEMON STARTED (VERSION {VERSION}) ===")
    submitted = False
    check_interval = 60 # seconds
    start_time = time.time()
    max_duration = 5400 # 90 minutes max limit

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_duration:
            log(f"ALERT: Kernel has been running for {elapsed/60:.1f} minutes (> 90m limit)! Timeout reached.")
            break

        status_line = get_kernel_status()
        log(f"[{elapsed/60:.1f}m elapsed] Kernel Status: {status_line}")

        if "KernelWorkerStatus.COMPLETE" in status_line:
            log("KERNEL RUN COMPLETED SUCCESSFULLY!")
            
            # Download kernel outputs
            os.makedirs("v17_output", exist_ok=True)
            log("Downloading kernel outputs...")
            subprocess.run(["kaggle", "kernels", "output", SLUG, "-p", "v17_output"], capture_output=True, text=True)
            
            # Check if submission.parquet exists
            sub_path = os.path.join("v17_output", "submission.parquet")
            if os.path.exists(sub_path):
                log(f"Found {sub_path} ({os.path.getsize(sub_path)} bytes)")
            else:
                log(f"WARNING: {sub_path} not found in output directory, proceeding with submit anyway.")

            # Submit to competition
            log("Attempting submission to competition...")
            success = submit_kernel(version=VERSION)
            if success:
                log("Submission command succeeded! Now tracking submission scoring status...")
                submitted = True
                break
            else:
                log("Submission command failed. Retrying in 30 seconds...")
                time.sleep(30)
                continue

        elif "KernelWorkerStatus.ERROR" in status_line:
            log("KERNEL RUN FAILED WITH ERROR!")
            subprocess.run(["kaggle", "kernels", "output", SLUG, "-p", "v17_output"], capture_output=True, text=True)
            log("Downloaded failure logs to v17_output.")
            break

        time.sleep(check_interval)

    # If submitted, monitor until scored
    if submitted:
        log("Monitoring competition submission scoring...")
        for _ in range(60): # up to 60 minutes
            time.sleep(60)
            subs = get_submissions()
            log("Recent submissions:\n" + "\n".join(subs.splitlines()[:8]))
            if "Champion V17" in subs:
                for line in subs.splitlines():
                    if "Champion V17" in line:
                        if "COMPLETE" in line:
                            log(f"*** V17 SUBMISSION COMPLETE! Final Result: {line} ***")
                            return
                        elif "ERROR" in line:
                            log(f"*** V17 SUBMISSION ERROR! {line} ***")
                            return

if __name__ == "__main__":
    main()
