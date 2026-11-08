import time, subprocess, sys

REFS = ["57720964", "57721397"]
print(f"Monitoring V37 ({REFS[0]}) and V38 ({REFS[1]})...", flush=True)

while True:
    try:
        res = subprocess.run(['kaggle', 'competitions', 'submissions', '-c', 'name-that-llm', '--csv'],
                             capture_output=True, text=True)
        lines = res.stdout.splitlines()
        found = {}
        for l in lines:
            for r in REFS:
                if r in l:
                    found[r] = l
        for r in REFS:
            if r in found:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {found[r]}", flush=True)
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ref {r} not in top submissions.", flush=True)
        
        # Check if all completed
        all_done = True
        for r in REFS:
            if r in found and 'PENDING' in found[r]:
                all_done = False
        if all_done and len(found) == len(REFS):
            print("All monitored submissions completed!", flush=True)
            sys.exit(0)
    except Exception as e:
        print(f"Error checking: {e}", flush=True)
    time.sleep(60)
