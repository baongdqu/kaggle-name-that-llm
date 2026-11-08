with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Add row 50
v50_row = """| **50** | 2026-10-30 15:20 | `57799075` | **Champion V50 Pure Apex Sovereign**: 0.663 SOTA Foundation + Meta Llama-3.1-8B (Golden Ratio $w=1.8$, `dp=True`, No Gauss) + ModernBERT-Large ($w=2.0$, Context 8192) + Qwen3-14B ($w=3.0$, `gauss=0.70`) + Nishant 14B ($w=2.5$) + Qwen3-8B ($w=2.0$, `gauss=0.44`, `dp=True`) + 20% ATLAS + Pure Hierarchical Beta-Bernoulli Typography (37 markers) + Damped Sinkhorn 0.45 (100% Hungarian-Free) | **0.6m** *(Fast Fixture 30s)* | *Đang xếp hàng suy luận (~3.8h)...* | **PENDING** | `PENDING` | **ĐÃ NỘP THÀNH CÔNG VÀO HÀNG ĐỢI KAGGLE (Ref: 57799075)!** Phiên bản cột mốc số 50: (1) Khắc phục triệt để bẫy Hungarian Matching trên test ẩn thưa thớt (2-4 câu/prompt); (2) Trả lại quyền quyết định tối thượng cho giải thuật Hierarchical Beta-Bernoulli Typography (37 markers) kết hợp Damped Sinkhorn 0.45; (3) Tinh chỉnh Meta Llama-3.1-8B về Tỷ lệ Vàng $w=1.8$ không dùng Gauss head để tránh over-smoothing với 20% ATLAS; (4) Tích hợp ModernBERT-Large $w=2.0$ context 8192 sâu sắc. |"""

idx49_end = content.find("| **49** |")
idx49_end = content.find("\n", idx49_end)

content = content[:idx49_end+1] + v50_row + "\n" + content[idx49_end+1:]

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("Appended V50 to abc SUBMISSIONS.md successfully!")

