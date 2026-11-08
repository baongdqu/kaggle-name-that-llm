import re

with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Update row 64
old_r64 = "| **64** | 2026-11-02 13:54 | `57898893` | **Champion V64 Sovereign ModernBERT Apex**: **Sovereign ModernBERT-Large 31k Pool (`w=2.2`, Val Acc 0.7185)** + **Guo et al. Calibrated Logits ($T=0.684$, Zero Logit Drift)** + **Public-Slice Adaptive Dense Cluster Threshold ($5 \\le \\text{size} \\le 18$)** + **Latin-Square Anchor Kuhn-Munkres (98% Win Rate in Lab)** + **Golden Damped Sinkhorn 0.45 (`--sinkhorn 0.45`, `--sinkhorn_target_mix 0.35`)** + **Restored Qwen 2.5 14B (`w=1.0`)** + **Llama 3.1 8B (`w=1.8`, dp=True)** + **Prompt-Aware Relative Length Calibration ($\\Delta z_{\\text{len}} \\cdot \\beta_m$)** | **0.6m** *(Fast Fixture 35s)* | **Đang chấm điểm ẩn** *(~240m)* | **`PENDING`** | `PENDING` |"

new_r64 = "| **64** | 2026-11-02 13:54 | `57898893` | **Champion V64 Sovereign ModernBERT Apex**: **Sovereign ModernBERT-Large 31k Pool (`w=2.2`, Val Acc 0.7185)** + **Guo et al. Calibrated Logits ($T=0.684$, Zero Logit Drift)** + **Public-Slice Adaptive Dense Cluster Threshold ($5 \\le \\text{size} \\le 18$)** + **Latin-Square Anchor Kuhn-Munkres (98% Win Rate in Lab)** + **Golden Damped Sinkhorn 0.45 (`--sinkhorn 0.45`, `--sinkhorn_target_mix 0.35`)** + **Restored Qwen 2.5 14B (`w=1.0`)** + **Llama 3.1 8B (`w=1.8`, dp=True)** + **Prompt-Aware Relative Length Calibration ($\\Delta z_{\\text{len}} \\cdot \\beta_m$)** | **0.6m** *(Fast Fixture 35s)* | **~240m** *(4h 00m)* | **`0.654`** | `COMPLETE` |"

text = text.replace(old_r64, new_r64)

# Update current status
text = re.sub(
    r'- \*\*Hạn mức hôm nay \(02/11 UTC\):\*\* Đã nộp \*\*2/5 lượt \(Champion V63 `0.662` 🥉👑, Champion V64 đang chấm điểm - Ref `57898893`\)\*\*, \*\*còn lại 3 lượt nộp bài\*\* cho ngày hôm nay\.',
    '- **Hạn mức hôm nay (02/11 UTC):** Đã nộp **2/5 lượt (Champion V63 `0.662` 🥉👑, Champion V64 `0.654`)**, **còn lại 3 lượt nộp bài** cho ngày hôm nay. Điểm chính thức vẫn bảo toàn tuyệt đối ở **`0.662` (Rank 312 / 4,150 đội)**.',
    text
)

text = re.sub(
    r'- \*\*Tổng số bài đã nộp:\*\* \*\*64 bài\*\* \(58 bài hoàn tất `COMPLETE`, \*\*1 bài đang chấm điểm: Champion V64 - Ref `57898893`\*\*\)\.',
    '- **Tổng số bài đã nộp:** **64 bài** (59 bài hoàn tất `COMPLETE`, **0 bài đang chấm điểm**).',
    text
)

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated abc SUBMISSIONS.md for V64 COMPLETE 0.654!")

