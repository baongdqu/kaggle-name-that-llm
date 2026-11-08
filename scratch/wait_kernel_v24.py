import subprocess
import time
import sys

kernel = "baonguyenduyquoc/name-that-llm-champion-v24-clean-sovereign"
print(f"Waiting for {kernel} commit to complete...")

for i in range(24):
    res = subprocess.run(["kaggle", "kernels", "status", kernel], capture_output=True, text=True)
    out = res.stdout.strip()
    print(f"[{i*10}s] {out}", flush=True)
    if "COMPLETE" in out:
        print("Kernel commit finished successfully!")
        sys.exit(0)
    elif "ERROR" in out or "FAILED" in out:
        print("Kernel commit FAILED!")
        sys.exit(1)
    time.sleep(10)

print("Timeout waiting for kernel commit.")
sys.exit(2)
