import re

with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace score
text = re.sub(r'Public Score: 0\.646', 'Public Score: 0.649', text, count=1)

# Replace table rows
old_v37_marker = r"\| \*\*37\*\* \| 2026-10-28 09:32 \| `57720964` \|.*?\n\| \*\*38\*\* \| 2026-10-28 09:51 \| `57721397` \|.*?\n"

new_rows = """| **37** | 2026-10-28 09:32 | `57720964` | **Champion V37 Llama Apex**: Meta Llama-3.1-8B 5-Fold (`ll8v5`, $w=1.0$) + Proven V32 Golden Ratio (LB 0.646) + Dual Encoders (MB 2.4 / Deb 1.6) + 4-View Stylometrics ($w=0.22$) + Topic-Bias De-noising ($\\alpha=0.62$) | **2.2m** *(Fast Fixture 30s)* | **~440m** *(7h 20m)* | **`0.649`** 🥈👑 | `COMPLETE` | **KỶ LỤC MỚI TOÀN ĐỘI (+0.003 SO VỚI TRẦN 0.646) - CHÍNH THỨC BỨT PHÁ LÊN 0.649!** Đập tan trần kính bão hòa 0.646 giam cầm suốt 7 phiên bản (V30-V36). Minh chứng thép khẳng định: Tín hiệu trực giao phi-Qwen từ Meta Llama-3.1-8B 5-Fold (tập từ vựng 128k, RoPE theta 500k, pretraining bias độc lập từ Meta) bắt chính xác các mẫu mà Qwen và Encoders bỏ sót. |
| **38** | 2026-10-28 09:51 | `57721397` | **Champion V38 Apex Trinity**: The Sovereign Trinity (Alibaba Qwen + Meta Llama-3.1-8B 5-Fold $w=1.0$ + Google DeepMind Gemma-2-9B $w=1.2$) + Dual Encoders (MB 2.4 / Deb 1.6) + 4-View Stylometrics ($w=0.22$) + Pure Flat 18-Class | **3.9m** *(Fast Fixture 30s)* | **>540m** *(>9h 00m)* | `None` | `ERROR (TIMEOUT)` | **SỰ CỐ NOTEBOOK TIMEOUT (QUÁ 9 TIẾNG SUY LUẬN TRÊN KAGGLE):** Việc nhồi nhét đồng thời 16 lượt forward pass Causal nặng (5-fold Qwen 14B + 5-fold Qwen 8B + 5-fold Llama-3.1-8B + Gemma-2-9B) trên 4,000 dòng test ẩn vượt quá giới hạn 9.0h của Kaggle GPU T4. Bài học sống còn: Cần tinh gọn (pruning) các fold trùng lặp hoặc chọn fold đại diện xuất sắc nhất để đảm bảo runtime an toàn trong ngưỡng 4-6 tiếng! |
"""

text = re.sub(old_v37_marker, new_rows, text, flags=re.DOTALL)

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated abc SUBMISSIONS.md successfully")

