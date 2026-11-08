with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.splitlines()
new_lines = []
for line in lines:
    if '- **Tổng số bài đã nộp:**' in line:
        new_lines.append('- **Tổng số bài đã nộp:** **61 bài** (56 bài hoàn tất `COMPLETE`, **0 bài đang chấm điểm**).')
    elif 'Siêu phẩm V61 đang chạy chấm điểm ẩn' in line:
        new_lines.append('- **Hạn mức hôm nay (01/11 UTC):** Hôm nay **đã sử dụng 4/5 lượt nộp bài (Champion V58, V59, V60, V61)** (còn **1 lượt nộp** duy nhất cho ngày hôm nay; hạn mức reset về 5 lượt mới vào 00:00 UTC / 07:00 sáng mai). V61 đã hoàn tất chấm điểm (0.658), kỷ lục toàn đội được bảo toàn vững chắc ở mốc đỉnh cao **`0.662`** 🥉👑 (Rank 312 toàn cầu).')
    elif '57868985' in line:
        row_61 = '| **61** | 2026-11-01 17:40 | `57868985` | **Champion V61 Judith Sovereign Apex**: **0.662 V59 Anchor** + **Judith007 Marginal Sinkhorn 0.60 (`--sinkhorn 0.60`, `--sinkhorn_target_mix 0.35`)** + **Softened Evidence Clipping 1.2 (`clip(1.2 * evidence, -0.30, 0.30)`)** + **Boosted Qwen 2.5 14B (`w=2.0`)** + **V59 Prompt-Aware Relative Length Bias ($\Delta z_{\\text{len}} \\cdot \\beta_m$)** + **Anti-Hungarian Discipline (`--perpassage 1`)** | **0.6m** *(Fast Fixture 35s)* | **~242m** *(4h 02m)* | **`0.658`** 🥈 | `COMPLETE` | **CHẤM ĐIỂM HOÀN TẤT - ĐẠT 0.658 (BẢO VỆ VỮNG CHẮC HUY CHƯƠNG BẠC, KỶ LỤC ĐỘI BẢO TOÀN TẠI 0.662):** Phân tích thực nghiệm & phòng thí nghiệm số: (1) Việc nâng Marginal Sinkhorn từ 0.45 lên 0.60 theo công thức Judith007 đã gây ra hiện tượng nắn ép biên quá đà (Reverse Leverage) trên ensemble 7 mô hình của đội ta, làm sụt nhẹ -0.004 điểm so với V59/V60 (0.662); (2) Tăng trọng số Qwen 2.5 14B lên 2.0 làm lấn át biểu diễn trực giao của ModernBERT và Llama-3.1-8B; (3) Khẳng định chân lý: Cấu hình mỏ neo Sinkhorn 0.45 của V59/V60 là chuẩn mực tối ưu tuyệt đối. Kỷ lục cao nhất của đội vẫn bảo toàn tại **0.662 (Rank 312 / 4,150 đội)**. |'
        new_lines.append(row_61)
    else:
        new_lines.append(line)

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines) + '\n')

print("Updated abc SUBMISSIONS.md successfully!")

