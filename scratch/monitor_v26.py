import subprocess
import time
import datetime

SUB_REF = "57637923"
LOG_FILE = "scratch/monitor_v26.log"

def log(msg):
    ts = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

log(f"=== BẮT ĐẦU GIÁM SÁT BÀI NỘP CHAMPION V26 SOVEREIGN CALIBRATED (Ref: {SUB_REF}) ===")

start_time = time.time()
while True:
    try:
        res = subprocess.run(["kaggle", "competitions", "submissions", "-c", "name-that-llm"], capture_output=True, text=True)
        lines = res.stdout.strip().split("\n")
        matched = [l for l in lines if SUB_REF in l]
        elapsed_min = (time.time() - start_time) / 60.0
        
        if matched:
            row = matched[0]
            if "COMPLETE" in row:
                log(f"🎉 BÀI NỘP V26 ĐÃ HOÀN TẤT CHẤM ĐIỂM sau {elapsed_min:.1f} phút!")
                log(f"Kết quả chính thức: {row}")
                break
            elif "ERROR" in row or "FAILED" in row:
                log(f"❌ BÀI NỘP V26 BỊ LỖI sau {elapsed_min:.1f} phút!")
                log(f"Chi tiết: {row}")
                break
            else:
                log(f"[{elapsed_min:.1f}m] Status: PENDING | Full: {row[:150]}")
        else:
            log(f"[{elapsed_min:.1f}m] Không tìm thấy dòng chứa ref {SUB_REF}")
    except Exception as e:
        log(f"Lỗi khi kiểm tra: {e}")
        
    time.sleep(60)
