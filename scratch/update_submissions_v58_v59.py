import re

with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Update total submissions count
content = content.replace("- **Tổng số bài đã nộp:** **57 bài**", "- **Tổng số bài đã nộp:** **59 bài**")
content = content.replace("Hôm nay (31/10 UTC):", "Hôm nay (01/11 UTC):")
content = content.replace(
    "Hôm nay **đã sử dụng trọn vẹn 5/5 lượt nộp bài (V53, V54, V55, V56, V57)**",
    "Hôm nay **đã sử dụng 2/5 lượt nộp bài (Champion V58, V59)** (còn 3 lượt nộp cho ngày hôm nay)"
)

new_rows = """| **58** | 2026-11-01 08:35 | *ref* | **Champion V58 Multi-Family Likelihood Titan**: **V50 Golden Anchor (0.659 High Watermark)** + **Expanded ATLAS 22% Sweet Spot (margin < 0.35)** + **Intra-Family Sibling LogSumExp Gate (100% Family Mass Conserved)** + **Anti-Hungarian Discipline (--perpassage 1)** + **Damped Sinkhorn 0.45** | **0.6m** *(Fast Fixture 35s)* | *Đang suy luận (PENDING)* | `PENDING` | *Đang chấm* | **ĐỘT PHÁ A: MỞ RỘNG LIKELIHOOD ĐA HỌ TRÊN LÕI V50:** Giữ nguyên 100% dàn Causal Titans đã được chứng minh hiệu quả của V50, mở rộng vùng quét ATLAS Likelihood lên các mẫu tranh chấp (margin < 0.35) và áp dụng cổng LogSumExp bảo toàn tuyệt đối 100% khối lượng xác suất liên họ, đánh sập 50% sai số anh em nội bộ. |
| **59** | 2026-11-01 08:36 | *ref* | **Champion V59 Prompt-Aware Relative Sovereign**: **V50 Golden Anchor (0.659 High Watermark)** + **Prompt-Aware Relative Length Calibration ($\Delta z_{\\text{len}} \\cdot \\beta_m$)** + **Soft Injective Collision Dispersion (Anti-Duplicate Dispersion on Prompt Clusters)** + **Anti-Hungarian Discipline (--perpassage 1)** + **Damped Sinkhorn 0.45** | **0.6m** *(Fast Fixture 35s)* | *Đang suy luận (PENDING)* | `PENDING` | *Đang chấm* | **ĐỘT PHÁ B: HIỆU CHUẨN TƯƠNG ĐỐI THEO CỤM ĐỀ THI:** Khai thác đặc trưng chiều dài tương đối của 18 models trong cùng cụm chủ đề đề thi test ($G_{te}$), kết hợp cơ chế phân tán va chạm mềm (damp 0.65) để ngăn ngừa gán nhãn trùng lặp trong cùng một đề thi mà không mắc bẫy sụp đổ của Hard Hungarian trên tập test thưa. |
"""

target = "| **57** | 2026-10-31 19:49 | 57838939 |"
# Find the line starting with target
lines = content.splitlines()
idx_57 = -1
for i, l in enumerate(lines):
    if l.startswith(target):
        idx_57 = i
        break

if idx_57 != -1:
    lines.insert(idx_57 + 1, new_rows.strip())
    content = '\n'.join(lines)
    with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Updated SUBMISSIONS.md with V58 and V59 successfully!')
else:
    print('Target V57 line not found!')

