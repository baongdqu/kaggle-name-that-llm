import subprocess, time, datetime, sys

SUB_ID = '57749696'

def log(msg):
    ts = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open('scratch/monitor_v41.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')

log("=== KHỞI ĐỘNG GIÁM SÁT BÀI NỘP V41 (REF: 57749696) ===")

while True:
    try:
        res = subprocess.run(['kaggle', 'competitions', 'submissions', '-c', 'name-that-llm', '--csv'], capture_output=True, text=True)
        lines = res.stdout.strip().split('\n')
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) >= 6:
                ref, fn, dt, desc, status, pub = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
                if ref == SUB_ID:
                    log(f"Ref {ref} | {status} | Score: {pub} | {desc[:40]}...")
                    if 'PENDING' not in status:
                        log(f"🎉 V41 HOÀN TẤT CHẤM ĐIỂM! Score: {pub}")
                        try:
                            import winsound
                            for f in [800, 1000, 1200, 1500, 2000]:
                                winsound.Beep(f, 250)
                                time.sleep(0.05)
                        except Exception:
                            pass
                        sys.exit(0)
    except Exception as e:
        log(f"Lỗi: {e}")
    time.sleep(60)

