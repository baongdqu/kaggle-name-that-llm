import re

with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Update status
text = re.sub(
    r'- \*\*Tổng số bài đã nộp:\*\* \*\*63 bài\*\* \(57 bài hoàn tất `COMPLETE`, \*\*1 bài đang chấm điểm: Champion V63 - Ref `57887198`\*\*\)',
    '- **Tổng số bài đã nộp:** **63 bài** (58 bài hoàn tất `COMPLETE`, **0 bài đang chấm điểm**).',
    text
)

text = re.sub(
    r'- \*\*Hạn mức hôm nay \(02/11 UTC\):\*\* Đã nộp \*\*1/5 lượt \(Champion V63 lúc 07:00 AM VN / 00:00 UTC\)\*\*, \*\*còn lại 4 lượt nộp bài\*\* cho ngày hôm nay\. Champion V63 mang mã Ref `57887198`, đang trong tiến trình chấm điểm trên hệ thống Kaggle ngầm \(~4 tiếng\)\.',
    '- **Hạn mức hôm nay (02/11 UTC):** Đã nộp **1/5 lượt (Champion V63 - Điểm chính thức: `0.662` 🥉👑)**, **còn lại 4 lượt nộp bài** cho ngày hôm nay. Đội duy trì vững chắc kỷ lục tại **Rank 312 / 4,150 đội**.',
    text
)

# Update row 63
old_r63 = "| **63** | 2026-11-02 07:00 | `57887198` | **Champion V63 Adaptive Anchor Apex**: **0.662 V59/V62 Anchor Core** + **Public-Slice Adaptive Dense Cluster Threshold ($5 \\le \\text{size} \\le 18$)** + **Latin-Square Anchor Kuhn-Munkres (136/150 active clusters, 98% Win Rate in Lab)** + **Golden Damped Sinkhorn 0.45 (`--sinkhorn 0.45`, `--sinkhorn_target_mix 0.35`)** + **Restored Qwen 2.5 14B (`w=1.0`)** + **Llama 3.1 8B (`w=1.8`, dp=True)** + **Prompt-Aware Relative Length Calibration ($\\Delta z_{\\text{len}} \\cdot \\beta_m$)** | **0.6m** *(Fast Fixture 35s)* | **Đang chấm điểm ẩn** *(~240m)* | **`PENDING`** | `PENDING` |"

new_r63 = "| **63** | 2026-11-02 07:00 | `57887198` | **Champion V63 Adaptive Anchor Apex**: **0.662 V59/V62 Anchor Core** + **Public-Slice Adaptive Dense Cluster Threshold ($5 \\le \\text{size} \\le 18$)** + **Latin-Square Anchor Kuhn-Munkres (136/150 active clusters, 98% Win Rate in Lab)** + **Golden Damped Sinkhorn 0.45 (`--sinkhorn 0.45`, `--sinkhorn_target_mix 0.35`)** + **Restored Qwen 2.5 14B (`w=1.0`)** + **Llama 3.1 8B (`w=1.8`, dp=True)** + **Prompt-Aware Relative Length Calibration ($\\Delta z_{\\text{len}} \\cdot \\beta_m$)** | **0.6m** *(Fast Fixture 35s)* | **~240m** *(4h 00m)* | **`0.662`** 🥉👑 | `COMPLETE` |"

text = text.replace(old_r63, new_r63)

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated abc SUBMISSIONS.md for V63 COMPLETE 0.662!")

