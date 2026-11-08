import subprocess
import time
import os
import json
import re
from datetime import datetime

KERNEL_SLUG = "baonguyenduyquoc/name-that-llm-champion-v18-apex-titan"
OUTPUT_DIR = "v18_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LIVE_LOG_FILE = os.path.join(OUTPUT_DIR, "champion-v18-live.log")

start_time = time.time()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Khởi động hệ thống tự động giám sát & chấm điểm Champion V18...", flush=True)
print(f"Kernel Target: {KERNEL_SLUG}", flush=True)
print(f"Cơ chế: Kiểm tra trạng thái mỗi 60s, định kỳ xuất báo cáo mỗi 10 phút. Tự động nộp bài và cập nhật kết quả khi hoàn tất, sau đó dừng hẳn theo dõi.", flush=True)

last_report_time = start_time
check_count = 0
ten_min_period = 0

while True:
    check_count += 1
    current_time = time.time()
    elapsed_total = current_time - start_time
    t_str = datetime.now().strftime('%H:%M:%S')
    
    # 1. Kiểm tra status kernel
    try:
        status_res = subprocess.run(["kaggle", "kernels", "status", KERNEL_SLUG], capture_output=True, text=True, timeout=30)
        status_out = status_res.stdout.strip()
    except Exception as e:
        status_out = f"Lỗi: {e}"

    # 2. Báo cáo định kỳ mỗi 10 phút hoặc ở lần đầu tiên
    is_ten_min = (current_time - last_report_time) >= 600 or check_count == 1
    if is_ten_min:
        ten_min_period += 1
        last_report_time = current_time
        print(f"\n==================================================", flush=True)
        print(f"[{t_str}] ⏱️ BÁO CÁO ĐỊNH KỲ 10 PHÚT (Mốc {ten_min_period * 10}m - Tổng thời gian chạy: {elapsed_total/60:.1f}m)", flush=True)
        print(f"Trạng thái Kernel: {status_out}", flush=True)
        
        # Thử lấy streaming log nếu có
        try:
            log_res = subprocess.run(["kaggle", "kernels", "logs", KERNEL_SLUG], capture_output=True, text=True, timeout=60)
            raw_logs = log_res.stdout.strip()
            json_idx = raw_logs.find('[')
            if json_idx != -1:
                entries = json.loads(raw_logs[json_idx:])
                print(f"Tổng số bản ghi log Kaggle: {len(entries)} events", flush=True)
                recent_lines = []
                for e in entries[-10:]:
                    d = e.get('data', '').strip()
                    if d:
                        recent_lines.append(f"[{e.get('time', 0.0):.1f}s][{e.get('stream_name')}] {d}")
                if recent_lines:
                    print("Các log gần nhất:")
                    for rl in recent_lines:
                        print("  ", rl)
            else:
                print("Kernel đang trong quá trình thực thi trên GPU (chưa flush streaming buffer).", flush=True)
        except Exception as le:
            print(f"Log check note: {le}", flush=True)
        print(f"==================================================\n", flush=True)

    # 3. Xử lý khi Kernel hoàn tất thành công
    if "KernelWorkerStatus.COMPLETE" in status_out:
        print(f"\n[{t_str}] 🎉🎉🎉 KERNEL V18 ĐÃ HOÀN THÀNH THÀNH CÔNG! (Thời gian chạy: {elapsed_total/60:.1f} phút)", flush=True)
        
        # Tải artifacts về v18_output
        print(f"[{t_str}] Đang tải output artifacts về {OUTPUT_DIR}...", flush=True)
        subprocess.run(["kaggle", "kernels", "output", KERNEL_SLUG, "-p", OUTPUT_DIR], capture_output=True, text=True, timeout=180)
        
        # Tải full logs
        try:
            log_res = subprocess.run(["kaggle", "kernels", "logs", KERNEL_SLUG], capture_output=True, text=True, timeout=60)
            raw_logs = log_res.stdout.strip()
            json_idx = raw_logs.find('[')
            if json_idx != -1:
                entries = json.loads(raw_logs[json_idx:])
                log_file_path = os.path.join(OUTPUT_DIR, "champion-v18.log")
                with open(log_file_path, "w", encoding="utf-8") as lf:
                    for e in entries:
                        lf.write(f"[{e.get('time', 0.0):.1f}s][{e.get('stream_name')}] {e.get('data', '')}")
                print(f"[{t_str}] Đã lưu toàn bộ {len(entries)} dòng log vào {log_file_path}.", flush=True)
        except Exception as le:
            print(f"Lưu log thất bại: {le}", flush=True)

        sub_file = os.path.join(OUTPUT_DIR, "submission.parquet")
        if os.path.exists(sub_file):
            size_kb = os.path.getsize(sub_file) / 1024
            print(f"[{t_str}] ✅ Xác nhận {sub_file} hợp lệ ({size_kb:.2f} KB).", flush=True)
        else:
            print(f"[{t_str}] ⚠️ Cảnh báo: Chưa thấy file {sub_file} trong {OUTPUT_DIR}.", flush=True)

        # Tiến hành submit tự động lên Kaggle
        print(f"[{t_str}] 🚀 Đang gửi bài nộp chính thức lên Kaggle Competition 'name-that-llm'...", flush=True)
        submit_cmd = [
            "kaggle", "competitions", "submit", "name-that-llm",
            "-k", KERNEL_SLUG,
            "-f", "submission.parquet",
            "-v", "1",
            "-m", "Champion V18 Apex Titan (ATLAS 10% + Restored DeBERTa/UltraBERT + Hungarian Injective)"
        ]
        sub_res = subprocess.run(submit_cmd, capture_output=True, text=True)
        print(f"[{t_str}] Kết quả gửi bài:\n{sub_res.stdout}\n{sub_res.stderr}", flush=True)

        # Chuyển sang vòng lặp theo dõi điểm số
        print(f"\n[{t_str}] 📊 Bắt đầu theo dõi quá trình chấm điểm Private/Public Test...", flush=True)
        scoring_start = time.time()
        final_score = None
        while True:
            time.sleep(60)
            sc_str = datetime.now().strftime('%H:%M:%S')
            sc_elapsed = (time.time() - scoring_start) / 60
            try:
                sub_list = subprocess.run(["kaggle", "competitions", "submissions", "-c", "name-that-llm"], capture_output=True, text=True, timeout=30)
                lines = sub_list.stdout.strip().splitlines()
                if len(lines) >= 3:
                    latest_line = lines[2]
                    print(f"[{sc_str}][Chấm điểm {sc_elapsed:.1f}m]: {latest_line}", flush=True)
                    if "COMPLETE" in latest_line:
                        print(f"\n==================================================", flush=True)
                        print(f"[{sc_str}] 🏆🏆🏆 CHẤM ĐIỂM HOÀN TẤT CHO CHAMPION V18!", flush=True)
                        print(f"Chi tiết kết quả: {latest_line}", flush=True)
                        
                        # Trích xuất điểm
                        parts = re.split(r'\s{2,}', latest_line)
                        # Cột publicScore thường ở vị trí thứ 5
                        for p in parts:
                            if re.match(r'^0\.\d{3,}$', p.strip()):
                                final_score = p.strip()
                                break
                        print(f"Điểm số Public LB ghi nhận: {final_score}", flush=True)
                        print(f"==================================================\n", flush=True)
                        break
                    elif "ERROR" in latest_line or "FAILED" in latest_line:
                        print(f"\n[{sc_str}] ❌ BÀI NỘP BỊ LỖI CHẤM ĐIỂM: {latest_line}", flush=True)
                        break
            except Exception as se:
                print(f"[{sc_str}] Lỗi kiểm tra submissions: {se}", flush=True)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] ĐÃ HOÀN TẤT TOÀN DIỆN MỤC TIÊU V18. KẾT THÚC TIẾN TRÌNH THEO DÕI.", flush=True)
        break

    elif "KernelWorkerStatus.ERROR" in status_out or "KernelWorkerStatus.CANCELLED" in status_out:
        print(f"\n[{t_str}] ❌ KERNEL GẶP SỰ CỐ HOẶC BỊ HỦY ({status_out})", flush=True)
        # Kéo log báo cáo chi tiết
        try:
            log_res = subprocess.run(["kaggle", "kernels", "logs", KERNEL_SLUG], capture_output=True, text=True, timeout=60)
            print(log_res.stdout[-2000:])
        except:
            pass
        break

    # Ngủ 60s trước lần kiểm tra tiếp theo
    time.sleep(60)
