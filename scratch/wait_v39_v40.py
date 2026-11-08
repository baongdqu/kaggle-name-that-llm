import subprocess, time, sys

def check_status(kernel_id):
    res = subprocess.run(['kaggle', 'kernels', 'status', kernel_id], capture_output=True, text=True)
    out = res.stdout + res.stderr
    if 'COMPLETE' in out:
        return 'COMPLETE'
    elif 'ERROR' in out or 'FAILED' in out:
        return 'ERROR: ' + out.strip()
    return 'RUNNING'

k39 = 'baonguyenduyquoc/name-that-llm-champion-v39-llama-amplifier'
k40 = 'baonguyenduyquoc/name-that-llm-champion-v40-sovereign-trinity-lean'

for i in range(20):
    s39 = check_status(k39)
    s40 = check_status(k40)
    print(f"[{i*15}s] V39: {s39} | V40: {s40}", flush=True)
    if s39 == 'COMPLETE' and s40 == 'COMPLETE':
        print("Both kernels completed successfully!", flush=True)
        sys.exit(0)
    if 'ERROR' in s39 or 'ERROR' in s40:
        print("Error detected!", flush=True)
        sys.exit(1)
    time.sleep(15)

print("Timeout waiting for kernels in script", flush=True)

