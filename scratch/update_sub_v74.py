import re

file_path = 'abc SUBMISSIONS.md'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

v74_row = """| **74** | 2026-11-04 13:20 | `57973975` | **Champion V74 TokAttn Orthogonal Apex**: **0.662 V68 Golden Anchor Base** + **Task-Specific Learned Token-Attention Head** (`tokattn_debl23_final.pt` - DeBERTa-v3-Large Layer 23, $w=0.85$) + **Latin-Square Anchor-Protected Kuhn-Munkres** (size >= 5) + **Damped Sinkhorn 0.45**. Bước đột phá đưa nguồn biểu diễn mới có trọng số attention học theo token, đã qua kiểm chứng 100% GroupKFold không leakage (OOF bacc = 0.4108, đạt recall cực cao trên nhóm lỗi cứng đầu: Claude Opus 5 0.727, GPT-6-Astra 0.613, Gemini-3.1 0.647). | **3.4m** *(Commit)* | *Đang chấm (Queue)...* | `57973975` | ⏳ Đang chấm |
"""

target_marker = "| **73** |"
idx = content.find(target_marker)
if idx != -1:
    new_content = content[:idx] + v74_row + content[idx:]
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("V74 successfully added to abc SUBMISSIONS.md!")
else:
    print("Target marker for V73 not found!")

