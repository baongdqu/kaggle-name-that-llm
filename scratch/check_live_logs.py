import subprocess
import os
import hashlib
from datetime import datetime

STATE_FILE = "scratch/last_log_state.txt"
SLUG = "baonguyenduyquoc/name-that-llm-champion-v17-apex-crown"

def get_kernel_status():
    try:
        res = subprocess.run(['kaggle', 'kernels', 'status', SLUG], capture_output=True, text=True, timeout=15)
        for line in res.stdout.splitlines():
            if SLUG in line:
                return line.strip()
        return res.stdout.strip()
    except Exception as e:
        return f"Status Error: {e}"

def get_live_log():
    try:
        p = subprocess.Popen(['kaggle', 'kernels', 'logs', '-f', SLUG],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, _ = p.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()
    except Exception as e:
        out = f"ERROR: {e}"
    return out.strip() if out else ""

def check_progress():
    status_line = get_kernel_status()
    current_log = get_live_log()
    current_hash = hashlib.md5(current_log.encode('utf-8')).hexdigest()
    
    last_hash = ""
    last_log = ""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                content = f.read().split("---DELIMITER---")
                if len(content) == 2:
                    last_hash = content[0].strip()
                    last_log = content[1].strip()
        except Exception:
            pass

    has_changed = (current_hash != last_hash)
    
    # Save current state
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        f.write(f"{current_hash}\n---DELIMITER---\n{current_log}")

    print(f"=== V17 PROGRESS CHECK AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Kernel Status: {status_line}")
    print(f"Has changed since last check: {has_changed}")
    print(f"Current log length: {len(current_log)} chars")
    
    if has_changed and last_log and current_log.startswith(last_log):
        diff = current_log[len(last_log):].strip()
        print("--- NEW LOG OUTPUT ADDED ---")
        print(diff)
    else:
        print("--- LATEST LOG SNIPPET (LAST 1500 CHARS) ---")
        print(current_log[-1500:] if current_log else "[No log output yet]")
        
    return has_changed, status_line, current_log

if __name__ == "__main__":
    check_progress()
