import subprocess, time, sys

kernel = "baonguyenduyquoc/ntl-champion-v74-tokattn-apex"
print(f"Tracking kernel {kernel}...")

for i in range(40):
    res = subprocess.run(["kaggle", "kernels", "status", kernel], capture_output=True, text=True)
    status_line = [l for l in res.stdout.split('\n') if "status" in l.lower()]
    print(f"[{i*10:3d}s] {status_line[0] if status_line else res.stdout.strip()}")
    if "complete" in res.stdout.lower():
        print("KERNEL FINISHED SUCCESSFULLY!")
        break
    if "error" in res.stdout.lower() or "cancel" in res.stdout.lower():
        print("KERNEL FAILED OR CANCELLED!")
        break
    time.sleep(10)

