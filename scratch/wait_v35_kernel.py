import subprocess, time

slug = "baonguyenduyquoc/name-that-llm-champion-v35-gemma2-hierarchical"
print(f"Polling kernel status for {slug}...")

for attempt in range(25):
    res = subprocess.run(["kaggle", "kernels", "status", slug], capture_output=True, text=True)
    out = res.stdout.strip()
    print(f"[{attempt+1}] {out}")
    if "complete" in out.lower():
        print("Kernel V35 is COMPLETE!")
        break
    if "error" in out.lower() or "failed" in out.lower():
        print("Kernel V35 FAILED!")
        break
    time.sleep(10)
