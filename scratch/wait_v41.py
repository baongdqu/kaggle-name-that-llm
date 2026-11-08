import subprocess, time, sys

k41 = 'baonguyenduyquoc/name-that-llm-champion-v41-apex-titan-sovereign'

for i in range(25):
    res = subprocess.run(['kaggle', 'kernels', 'status', k41], capture_output=True, text=True)
    out = res.stdout + res.stderr
    print(f"[{i*15}s] {out.strip()}", flush=True)
    if 'COMPLETE' in out:
        print("V41 kernel commit completed successfully!", flush=True)
        sys.exit(0)
    elif 'ERROR' in out or 'FAILED' in out:
        print("V41 kernel commit failed!", flush=True)
        sys.exit(1)
    time.sleep(15)

print("Timeout waiting for V41", flush=True)
sys.exit(2)

