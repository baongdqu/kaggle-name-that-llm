with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Update top summary
content = content.replace(
    "- **Thành tích cao nhất hiện tại:** **Public Score: 0.653** 🥈👑 (KỶ LỤC MỚI TOÀN ĐỘI - CHÍNH THỨC VƯƠN LÊN RANK 128 / 3,420 ĐỘI - TOP 3.7% TOÀN CẦU)",
    "- **Thành tích cao nhất hiện tại:** **Public Score: 0.662** 🥉👑 (KỶ LỤC MỚI TOÀN ĐỘI - THIẾT LẬP TẠI CHAMPION V59 - CHÍNH THỨC PHÁ VỠ NÚT THẮT 0.659 VÀ TIẾN VÀO RANK 486 / 4,132 ĐỘI)"
)
content = content.replace(
    "- **Điểm kỷ lục cao nhất của Đội:** **`0.659`**",
    "- **Điểm kỷ lục cao nhất của Đội:** **`0.662` 🥉👑 (Thiết lập tại Champion V59)**"
)
content = content.replace("- **Số lần đang chờ kết quả (Pending):** **2** lần (V58, V59)", "- **Số lần đang chờ kết quả (Pending):** **0** lần")
content = content.replace(
    "- **Trạng thái bài nộp cuối cùng:** `PENDING` (V58: 57853752, V59: 57853771 đang được Kaggle chấm điểm)",
    "- **Trạng thái bài nộp cuối cùng:** `0.662` (V59 lập đỉnh kỷ lục mới `0.662` phá vỡ nút thắt 0.659; V58 bảo toàn vững chắc `0.659`)"
)
content = content.replace("- **Số lần nộp thành công (Scored):** **31** lần", "- **Số lần nộp thành công (Scored):** **33** lần")

# Replace V58 row
old_v58 = "| **58** | 2026-11-01 07:00 | 57853752 | **Champion V58 Multi-Family Likelihood Titan**: **V50 Golden Anchor (0.659 High Watermark)** + **Expanded ATLAS 22% Sweet Spot (margin < 0.35)** + **Intra-Family Sibling LogSumExp Gate (100% Family Mass Conserved)** + **Anti-Hungarian Discipline (--perpassage 1)** + **Damped Sinkhorn 0.45** | **0.6m** *(Fast Fixture 35s)* | *Đang suy luận (PENDING)* | `PENDING` | *Đang chấm* | **ĐỘT PHÁ A: MỞ RỘNG LIKELIHOOD ĐA HỌ TRÊN LÕI V50:** Giữ nguyên 100% dàn Causal Titans đã được chứng minh hiệu quả của V50, mở rộng vùng quét ATLAS Likelihood lên các mẫu tranh chấp (margin < 0.35) và áp dụng cổng LogSumExp bảo toàn tuyệt đối 100% khối lượng xác suất liên họ, đánh sập 50% sai số anh em nội bộ. |"
new_v58 = "| **58** | 2026-11-01 07:00 | 57853752 | **Champion V58 Multi-Family Likelihood Titan**: **V50 Golden Anchor (0.659 High Watermark)** + **Expanded ATLAS 22% Sweet Spot (margin < 0.35)** + **Intra-Family Sibling LogSumExp Gate (100% Family Mass Conserved)** + **Anti-Hungarian Discipline (--perpassage 1)** + **Damped Sinkhorn 0.45** | **0.6m** *(Fast Fixture 35s)* | *Thành công (Submission Completed)* | `0.659` | **Rank 522** | **ĐỘT PHÁ A: MỞ RỘNG LIKELIHOOD ĐA HỌ TRÊN LÕI V50:** Bảo lưu hoàn hảo mốc sàn 0.659, khẳng định cơ chế bảo toàn khối lượng họ LogSumExp vận hành an toàn 100% và không làm rò rỉ xác suất giữa các hãng. |"

# Replace V59 row
old_v59 = "| **59** | 2026-11-01 07:00 | 57853771 | **Champion V59 Prompt-Aware Relative Sovereign**: **V50 Golden Anchor (0.659 High Watermark)** + **Prompt-Aware Relative Length Calibration ($\Delta z_{\\text{len}} \\cdot \\beta_m$)** + **Soft Injective Collision Dispersion (Anti-Duplicate Dispersion on Prompt Clusters)** + **Anti-Hungarian Discipline (--perpassage 1)** + **Damped Sinkhorn 0.45** | **0.6m** *(Fast Fixture 35s)* | *Đang suy luận (PENDING)* | `PENDING` | *Đang chấm* | **ĐỘT PHÁ B: HIỆU CHUẨN TƯƠNG ĐỐI THEO CỤM ĐỀ THI:** Khai thác đặc trưng chiều dài tương đối của 18 models trong cùng cụm chủ đề đề thi test ($G_{te}$), kết hợp cơ chế phân tán va chạm mềm (damp 0.65) để ngăn ngừa gán nhãn trùng lặp trong cùng một đề thi mà không mắc bẫy sụp đổ của Hard Hungarian trên tập test thưa. |"
new_v59 = "| **59** | 2026-11-01 07:00 | 57853771 | **Champion V59 Prompt-Aware Relative Sovereign**: **V50 Golden Anchor (0.659 High Watermark)** + **Prompt-Aware Relative Length Calibration ($\Delta z_{\\text{len}} \\cdot \\beta_m$)** + **Soft Injective Collision Dispersion (Anti-Duplicate Dispersion on Prompt Clusters)** + **Anti-Hungarian Discipline (--perpassage 1)** + **Damped Sinkhorn 0.45** | **0.6m** *(Fast Fixture 35s)* | *Thành công (Submission Completed)* | `0.662` 🥉👑 | **Rank 486** *(Nhảy +36 hạng)* | **BƯỚC NGOẶT ĐỘT PHÁ LỊCH SỬ (KỶ LỤC MỚI):** Lần đầu tiên trong lịch sử đội phá vỡ bức tường kẹt xe 0.659, nhảy vọt lên 0.662 và tiến thẳng vào Rank 486! Chứng minh tính hiệu quả vượt bậc của việc khai thác cấu trúc đề thi bàn cờ (Prompt-Relative Length Bias $\\beta_m$) kết hợp phân tán va chạm mềm, chỉ còn cách Huy chương Bạc 0.663 đúng 0.001 điểm! |"

assert old_v58 in content, 'old_v58 not found'
assert old_v59 in content, 'old_v59 not found'

content = content.replace(old_v58, new_v58)
content = content.replace(old_v59, new_v59)

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write(content)

print('Updated SUBMISSIONS.md with V58 (0.659) and V59 (0.662) successfully!')

