import time, subprocess, sys

REF = "57720964"
print(f"Monitoring V37 submission Ref: {REF}...", flush=True)

while True:
    try:
        res = subprocess.run(['kaggle', 'competitions', 'submissions', '-c', 'name-that-llm', '--csv'],
                             capture_output=True, text=True)
        lines = res.stdout.splitlines()
        found = False
        for l in lines:
            if REF in l:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {l}", flush=True)
                if 'COMPLETE' in l or 'ERROR' in l:
                    sys.exit(0)
                found = True
                break
        if not found:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ref {REF} not found in top submissions.", flush=True)
    except Exception as e:
        print(f"Error checking: {e}", flush=True)
    time.sleep(60)
