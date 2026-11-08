with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Update V50 row
old_v50 = '| **50** | 2026-10-30 15:20 | `57799075` | **Champion V50 Pure Apex Sovereign**'
assert old_v50 in content, "old_v50 not found"

new_v50 = """| **50** | 2026-10-30 15:20 | `57799075` | **Champion V50 Pure Apex Sovereign**: 0.663 SOTA Foundation + Meta Llama-3.1-8B (Golden Ratio $w=1.8$, `dp=True`, No Gauss) + ModernBERT-Large ($w=2.0$, Context 8192) + Qwen3-14B ($w=3.0$, `gauss=0.70`) + Nishant 14B ($w=2.5$) + Qwen3-8B ($w=2.0$, `gauss=0.44`, `dp=True`) + 20% ATLAS + Pure Hierarchical Beta-Bernoulli Typography (37 markers) + Damped Sinkhorn 0.45 (100% Hungarian-Free) | **0.6m** *(Fast Fixture 30s)* | **~280m** *(4h 40m)* | **`0.659`** 🥈👑🎉 | `COMPLETE` | **KỶ LỤC MỚI MỌI THỜI ĐẠI TOÀN ĐỘI (+0.003 SO VỚI V48, +0.001 SO VỚI ĐỈNH V42 0.658)!** Đột phá ngoạn mục: (1) Phá vỡ trần kính 0.658 của Champion V42 để thiết lập đỉnh cao mới **`0.659`** (Top 4.3% thế giới); (2) Minh chứng thép khẳng định: Việc loại bỏ Hungarian gượng ép trên test ẩn thưa thớt và trả lại quyền cho Hierarchical Beta-Bernoulli Typography (37 markers) kết hợp Llama Golden Ratio $w=1.8$ và ModernBERT $w=2.0$ đã phát huy tối đa sức mạnh phân biệt; (3) Thu hẹp khoảng cách tới Top 1 thế giới (0.663) chỉ còn đúng **0.004 điểm**! |"""

idx50 = content.find(old_v50)
idx_end50 = content.find("\n", idx50)

content = content[:idx50] + new_v50 + content[idx_end50:]

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated abc SUBMISSIONS.md with V50 official score 0.659!")

