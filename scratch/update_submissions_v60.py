with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Update total submissions count
content = content.replace("- **Tổng số bài đã nộp:** **59 bài**", "- **Tổng số bài đã nộp:** **60 bài**")
content = content.replace(
    "Hôm nay **đã sử dụng 2/5 lượt nộp bài (Champion V58, V59)** (còn 3 lượt nộp cho ngày hôm nay)",
    "Hôm nay **đã sử dụng 3/5 lượt nộp bài (Champion V58, V59, V60)** (còn 2 lượt nộp cho ngày hôm nay)"
)
content = content.replace("- **Số lần đang chờ kết quả (Pending):** **0** lần", "- **Số lần đang chờ kết quả (Pending):** **1** lần (V60)")
content = content.replace(
    "- **Trạng thái bài nộp cuối cùng:** `0.662` (V59 lập đỉnh kỷ lục mới `0.662` phá vỡ nút thắt 0.659; V58 bảo toàn vững chắc `0.659`)",
    "- **Trạng thái bài nộp cuối cùng:** `PENDING` (V60: 57860671 đang được Kaggle chấm điểm; đỉnh cao hiện tại: 0.662 tại V59)"
)

new_row_60 = """| **60** | 2026-11-01 11:56 | 57860671 | **Champion V60 Apex Synergy**: **0.662 V59 Breakthrough Anchor** + **Empirical Length Bias $\\Delta z_{\\text{len}} \\cdot \\beta_m$** + **Soft Typography Continuous Guards (em-dash suppresses Gemini by 90%, curly apostrophe suppresses Claude/GLM by 90%)** + **Intra-Family Sibling LogSumExp Sharpening (Family Mass Conserved)** + **Anti-Hungarian Discipline (--perpassage 1)** + **Damped Sinkhorn 0.45** | **0.6m** *(Fast Fixture 35s)* | *Đang suy luận (PENDING)* | `PENDING` | *Đang chấm* | **HIỆP ĐỒNG TỐI THƯỢNG CÔNG PHÁ MỐC HUY CHƯƠNG BẠC:** Hợp nhất toàn bộ tinh hoa của V59 (hiệu chuẩn độ dài theo cụm đề thi $G_{te}$ và phân tán va chạm mềm) với cổng làm sắc nét anh em nội bộ LogSumExp và bộ lọc dấu câu mềm liên tục. Quyết chiến vượt mốc 0.663 để nhảy vào Vùng Huy Chương Bạc Top 5%! |"""

target_59 = "| **59** | 2026-11-01 07:00 | 57853771 |"
lines = content.splitlines()
idx_59 = -1
for i, l in enumerate(lines):
    if l.startswith(target_59):
        idx_59 = i
        break

if idx_59 != -1:
    lines.insert(idx_59 + 1, new_row_60)
    content = '\n'.join(lines)
    with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Updated SUBMISSIONS.md with V60 (57860671) successfully!')
else:
    print('Target 59 line not found!')

