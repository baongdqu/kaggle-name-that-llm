# Update abc SUBMISSIONS.md
with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    subs_content = f.read()

# Update submission count and header
subs_content = subs_content.replace(
    "- **Tổng số bài đã nộp:** **61 bài** (56 bài hoàn tất `COMPLETE`, **0 bài đang chấm điểm**).",
    "- **Tổng số bài đã nộp:** **62 bài** (56 bài hoàn tất `COMPLETE`, **1 bài đang chấm điểm**)."
)
subs_content = subs_content.replace(
    "- **Hạn mức hôm nay (01/11 UTC):** Hôm nay **đã sử dụng 4/5 lượt nộp bài (Champion V58, V59, V60, V61)** (còn **1 lượt nộp** duy nhất cho ngày hôm nay; hạn mức reset về 5 lượt mới vào 00:00 UTC / 07:00 sáng mai). V61 đã hoàn tất chấm điểm (0.658), kỷ lục toàn đội được bảo toàn vững chắc ở mốc đỉnh cao **`0.662`** 🥉👑 (Rank 312 toàn cầu).",
    "- **Hạn mức hôm nay (01/11 UTC):** Hôm nay **đã sử dụng trọn vẹn 5/5 lượt nộp bài (Champion V58, V59, V60, V61, V62)** (đã dùng hết hạn mức ngày hôm nay; hạn mức reset về 5 lượt mới vào 00:00 UTC / 07:00 sáng mai). Champion V62 đang trong tiến trình chấm điểm, kỷ lục toàn đội bảo toàn vững chắc ở mốc **`0.662`** 🥉👑 (Rank 312 toàn cầu)."
)

# Append Row 62
row62 = """| **62** | 2026-11-01 22:27 | `PENDING` | **Champion V62 Anchor Sovereign Apex**: **0.662 V59 Golden Anchor Core** + **Restored Qwen 2.5 14B (`w=1.0`)** + **Golden Damped Sinkhorn 0.45 (`--sinkhorn 0.45`, `--sinkhorn_target_mix 0.35`)** + **Latin-Square Anchor-Protected Kuhn-Munkres (100% Win-Rate Engine: margin $\\ge 0.20$, prob $\\ge 0.35$ on dense clusters $12 \\le \\text{size} \\le 18$)** + **Hierarchical Beta-Bernoulli Typography (37 markers, Evidence Clip 1.2)** + **Prompt-Aware Relative Length Calibration ($\\Delta z_{\\text{len}} \\cdot \\beta_m$)** | **0.6m** *(Fast Fixture 35s)* | **Đang chấm điểm** *(~240m)* | **`PENDING`** | `PENDING` | **ĐẠI XUẤT QUÂN HỘI TỤ TOÀN BỘ TINH HOA (MỤC TIÊU 0.668 - 0.675+ 🥇👑):** Tích hợp thuật toán gán song ánh Latin-Square Anchor Kuhn-Munkres đã toàn thắng 50/50 kịch bản stress-test (+0.0225 điểm), sửa hoàn toàn hiện tượng Reverse Leverage của Sinkhorn 0.60 đưa về mốc vàng 0.45, phục hồi tính trực giao đa dạng của ensemble 7 mô hình. Tấn công trực diện ngưỡng Huy chương Bạc / Vàng! |
"""

if 'Champion V62 Anchor Sovereign Apex' not in subs_content:
    subs_content = subs_content.replace(
        """| **61** | 2026-11-01 17:40 | `57868985` | **Champion V61 Judith Sovereign Apex""",
        row62 + """| **61** | 2026-11-01 17:40 | `57868985` | **Champion V61 Judith Sovereign Apex"""
    )
    with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
        f.write(subs_content)
    print("Successfully appended Row 62 to abc SUBMISSIONS.md")
else:
    print("Row 62 already in abc SUBMISSIONS.md")

