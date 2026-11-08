import subprocess, time, datetime

SUB_IDS = ['57736884', '57736951']

def log(msg):
    ts = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open('scratch/monitor_v39_v40.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')

log("=== KHỞI ĐỘNG GIÁM SÁT BÀI NỘP V39 VÀ V40 ===")

while True:
    try:
        res = subprocess.run(['kaggle', 'competitions', 'submissions', '-c', 'name-that-llm', '--csv'], capture_output=True, text=True)
        lines = res.stdout.strip().split('\n')
        header = lines[0].split(',')
        found_pending = False
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) >= 6:
                ref, fn, dt, desc, status, pub = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
                if ref in SUB_IDS:
                    log(f"Ref {ref} | {status} | Score: {pub} | {desc[:40]}...")
                    if 'PENDING' in status:
                        found_pending = True
        if not found_pending:
            log("Tất cả các bài nộp theo dõi đã hoàn tất!")
            break
    except Exception as e:
        log(f"Lỗi truy vấn: {e}")
    time.sleep(120)

