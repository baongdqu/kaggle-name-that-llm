import sys

with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
updated = False
for line in lines:
    if '| **23** | 2026-10-25 11:26 |' in line:
        row23 = "| **23** | 2026-10-25 11:26 | `57616744` | **Champion V23 Perfected**: Curated Pool 86.5k + Lossless Trio + Twin `min_size=4` + Soft TypoGuard | **0.4m** *(24s fixture)* | **183.7m** *(3h 04m)* | `0.627` | `COMPLETE` | Bộ ba Lossless (A+B+C) chứng minh tiết kiệm 12.5m scoring. Rớt -0.012 điểm do: (1) Đòn bẩy ngược Sinkhorn khi TypoGuard liên họ dập Gemini (39% dòng) / Claude (21% dòng) khiến Sinkhorn bù hệ số cột x2.5-3.0 làm cướp nhãn sai ở các dòng khác; (2) Twin `min_size=4` triệt tiêu cụm size 2-3; (3) Nhiễu nhãn curated pool. Best đội vẫn bảo toàn 0.639. |\n"
        row24 = "| **24** | 2026-10-25 15:50 | `57623685` | **Champion V24 Clean Sovereign**: Golden Pool_v4 (75.8k) + Twin `min_size=2` + Proven Sibling Gates + Lossless Speedup Trio | **0.3m** *(20s fixture)* | *Đang chấm (~180m)* | *Đang chấm* | `SCORING` | **Đã nộp thành công!** Khắc phục toàn bộ 3 nguyên nhân của V23: Khôi phục nền dữ liệu chuẩn vàng `pool_v4.parquet` (75,783 mẫu) đồng pha LoRA Part Two; Triệt tiêu 100% TypoGuard liên họ (bảo vệ tuyệt đối đòn bẩy Sinkhorn); Khôi phục Twin Centering `min_size=2` cho cụm nhỏ; Giữ nguyên 100% Bộ ba tăng tốc Lossless A+B+C (tiết kiệm 12.5m) và Micro-Gates nội bộ họ an toàn. Daemon monitor đang theo dõi. |\n"
        new_lines.append(row23)
        new_lines.append(row24)
        updated = True
    else:
        new_lines.append(line)

if updated:
    with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("SUCCESSFULLY UPDATED abc SUBMISSIONS.md WITH ROWS 23 AND 24!")
else:
    print("FAILED TO FIND ROW 23!")
