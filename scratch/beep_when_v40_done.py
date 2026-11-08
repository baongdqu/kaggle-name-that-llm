import subprocess, time, sys

SUB_ID = '57736951'

print("=== DAEMON THEO DÕI V40 & KÍCH HOẠT ÂM BÁO BÍP KHI HOÀN TẤT ===", flush=True)

while True:
    try:
        res = subprocess.run(['kaggle', 'competitions', 'submissions', '-c', 'name-that-llm', '--csv'], capture_output=True, text=True)
        lines = res.stdout.strip().split('\n')
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) >= 6:
                ref, fn, dt, desc, status, pub = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
                if ref == SUB_ID:
                    if 'PENDING' not in status:
                        print(f"!!! V40 HOÀN TẤT !!! Status: {status} | Score: {pub}", flush=True)
                        try:
                            import winsound
                            for f in [800, 1000, 1200, 1500, 2000]:
                                winsound.Beep(f, 250)
                                time.sleep(0.05)
                        except Exception as e_beep:
                            print(f"Beep error: {e_beep}", flush=True)
                        sys.exit(0)
                    else:
                        print(f"V40 vẫn đang chạy (PENDING)...", flush=True)
    except Exception as e:
        print(f"Lỗi: {e}", flush=True)
    time.sleep(20)

