with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    text = f.read()

old_row_40 = "| **40** | 2026-10-28 19:46 | `57736951` | **Champion V40 Sovereign Trinity Lean**: Fold-Pruning Lean Passes (3 Passes/Model) + Tam Đại Trụ Cột Causal (Qwen + Llama-3.1-8B $w=2.0$ + Google Gemma-2-9B $w=1.5$) + Dual Encoders (MB 2.2 / Deb 1.5) | **3.5m** *(Fast Fixture 30s)* | *Đang chấm điểm...* | **PENDING** | `PENDING` | **CHÍNH THỨC NỘP BÀI - ĐẠI ĐỘT PHÁ CÔNG NGHỆ: VƯỢT QUA RÀO CẢN TIMEOUT BẰNG FOLD-PRUNING!** Cắt giảm 50% số lượt pass lặp lại vô ích (từ 21 lượt xuống 12 lượt), rút ngắn thời gian chạy xuống ~5.0 - 5.5 giờ (an toàn tuyệt đối cách xa mốc 9h). Lần đầu tiên hợp nhất thành công 100% cả 3 siêu kiến trúc Causal lớn nhất hành tinh (Qwen + Llama-3.1 + Gemma-2) cùng 2 Encoders trên mô hình phẳng. Mục tiêu: Công phá Top 50 toàn cầu (`0.665 - 0.675+`)! |"

new_row_40 = "| **40** | 2026-10-28 19:46 | `57736951` | **Champion V40 Sovereign Trinity Lean**: Fold-Pruning Lean Passes (3 Passes/Model) + Tam Đại Trụ Cột Causal (Qwen + Llama-3.1-8B $w=2.0$ + Google Gemma-2-9B $w=1.5$) + Dual Encoders (MB 2.2 / Deb 1.5) | **3.5m** *(Fast Fixture 30s)* | **~390m** *(6h 30m)* | **`0.649`** 🥈👑 | `COMPLETE` | **CHẤM ĐIỂM HOÀN TẤT - ĐẠT `0.649` (BẢO TOÀN VÙNG ĐIỂM TINH HOA, NGANG NGỬA V37)!** Giải phẫu thực nghiệm: (1) Fold-Pruning thành công 100% trong việc triệt tiêu lỗi Timeout của V38, đưa cả 3 họ Qwen + Llama + Gemma về đích an toàn sau 6.5h; (2) Việc cắt giảm từ 5 folds xuống 2 folds làm giảm nhẹ sức mạnh trung bình hóa phương sai của LoRA24 so với V39 (đạt kỷ lục 0.653 với full 5 folds Llama $w=2.4$); (3) Best của đội được bảo toàn vững chắc ở mốc đỉnh cao lịch sử **`0.653`** 🥈👑 (Champion V39 - Rank 144 toàn cầu). |"

text = text.replace(old_row_40, new_row_40)

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated abc SUBMISSIONS.md with V40 score 0.649 successfully")

