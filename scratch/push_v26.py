import subprocess
import time
import json

print("=== PUSHING KERNEL CHAMPION V26 SOVEREIGN CALIBRATED ===")
res = subprocess.run(["kaggle", "kernels", "push", "-p", "champion_v26"], capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
assert res.returncode == 0, f"Kernel push failed with code {res.returncode}"

kernel_id = "baonguyenduyquoc/name-that-llm-champion-v26-sovereign-calibrated"
print(f"Monitoring commit status for {kernel_id}...")
t0 = time.time()
while True:
    status_res = subprocess.run(["kaggle", "kernels", "status", kernel_id], capture_output=True, text=True)
    status_line = status_res.stdout.strip()
    print(f"[{time.time()-t0:.1f}s] {status_line}")
    if "complete" in status_line.lower():
        print("Kernel commit COMPLETE!")
        break
    elif "error" in status_line.lower() or "failed" in status_line.lower():
        print("Kernel commit FAILED!")
        exit(1)
    time.sleep(4)
