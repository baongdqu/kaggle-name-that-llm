import os

with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.splitlines()

# Update header lines
new_lines = []
for i, line in enumerate(lines):
    if line.startswith('# TỔNG QUAN NỘP BÀI'):
        new_lines.append('# TỔNG QUAN NỘP BÀI, ĐỐI CHUẨN THỜI GIAN & LỊCH SỬ THỬ NGHIỆM TỪ V1 ĐẾN V61 (SUBMISSIONS.md)')
    elif '> **Đội thi:**' in line:
        new_lines.append('> **Đội thi:** Bao Ng 🔫🔫🔫 (Team ID: 17137218 | Tài khoản: `baonguyenduyquoc` | Team Name: `private` / `Mohamed Morsi`)  ')
    elif '> **Thành tích cao nhất hiện tại:**' in line:
        new_lines.append('> **Thành tích cao nhất hiện tại:** **Public Score: 0.662** 🥉👑 (KỶ LỤC MỚI TOÀN ĐỘI - RANK 312 / 4,150 ĐỘI - TOP 7.5% TOÀN CẦU, THIẾT LẬP BỞI V59 & V60)  ')
    elif '> **Thời điểm cập nhật:**' in line:
        new_lines.append('> **Thời điểm cập nhật:** 01/11/2026 (17:40 PM Local / 10:40 AM UTC)  ')
    elif '> **Mục tiêu tài liệu:**' in line:
        new_lines.append('> **Mục tiêu tài liệu:** Cẩm nang toàn thư duy nhất về toàn bộ chu trình nộp bài, định dạng kỹ thuật, đối chuẩn thống kê thời gian thực thi (Runtime Benchmark), quy tắc vàng quản trị GPU 45-90, bảng nhật ký 61 bản nộp và chiến lược lựa chọn bài nộp Private Leaderboard.')
    elif '- **Tổng số bài đã nộp:**' in line:
        new_lines.append('- **Tổng số bài đã nộp:** **61 bài** (55 bài hoàn tất `COMPLETE`, **1 bài đang chấm điểm `PENDING` - Champion V61 Judith Sovereign Apex**).')
    elif '- **Điểm kỷ lục cao nhất của Đội:**' in line:
        new_lines.append('- **Điểm kỷ lục cao nhất của Đội:** **`0.662` 🥉👑 (Thiết lập tại Champion V59, V60)**, giữ vững vị thế tại **Rank 312 / 4,150 đội - Top 7.5% toàn cầu - Vùng Huy Chương Đồng vững chắc**!')
    elif '- **Hạn mức hôm nay' in line:
        new_lines.append('- **Hạn mức hôm nay (01/11 UTC):** Hôm nay **đã sử dụng 4/5 lượt nộp bài (Champion V58, V59, V60, V61)** (còn **1 lượt nộp** duy nhất cho ngày hôm nay; hạn mức reset về 5 lượt mới vào 00:00 UTC / 07:00 sáng mai). Siêu phẩm V61 đang chạy chấm điểm ẩn trên tập kiểm tra của Kaggle.')
    else:
        new_lines.append(line)
        if line.startswith('| **60** |'):
            row_61 = '| **61** | 2026-11-01 17:40 | `57868985` | **Champion V61 Judith Sovereign Apex**: **0.662 V59 Anchor** + **Judith007 Marginal Sinkhorn 0.60 (`--sinkhorn 0.60`, `--sinkhorn_target_mix 0.35`)** + **Softened Evidence Clipping 1.2 (`clip(1.2 * evidence, -0.30, 0.30)`)** + **Boosted Qwen 2.5 14B (`w=2.0`)** + **V59 Prompt-Aware Relative Length Bias ($\Delta z_{\\text{len}} \\cdot \\beta_m$)** + **Anti-Hungarian Discipline (`--perpassage 1`)** | **0.6m** *(Fast Fixture 35s)* | **Đang chấm (Scoring)** | *Đang chờ* (Target: `0.664+` 🥈) | `PENDING` | **ĐỘT PHÁ CẤU HÌNH JUDITH007 0.664 TRÊN NỀN V59:** Triển khai trực tiếp công thức vàng giúp Judith007 vượt qua mốc 0.664 (Top 87-200, phá vỡ điểm nghẽn 430 đội): nâng lực nắn Marginal Sinkhorn từ 0.45 lên 0.60 để ép phân phối biên về sát chuẩn $N/18$, làm mềm ngưỡng Evidence Clipping từ 1.5 xuống 1.2 để tránh trừng phạt gắt các dự đoán đúng trong cụm đề thi, đồng thời nâng trọng số backbone Qwen 2.5 14B lên 2.0. Toàn bộ chuỗi suy luận chạy trên nền tảng an toàn của V59. Submission đã nộp thành công và đang được Kaggle chấm điểm trên tập test ẩn. |'
            new_lines.append(row_61)

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines) + '\n')

print("Successfully updated abc SUBMISSIONS.md with Row 61!")

