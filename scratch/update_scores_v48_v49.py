with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Update V48 and V49 rows
old_v48 = '| **48** | 2026-10-30 10:28 | `57791134` | **Champion V48 Clean Sovereign**'
old_v49 = '| **49** | 2026-10-30 10:28 | `57791141` | **Champion V49 Apex Dual-Titan Gauss**'

new_v48 = """| **48** | 2026-10-30 10:28 | `57791134` | **Champion V48 Clean Sovereign**: 0.663 World SOTA Foundation + ModernBERT-Large ($w=2.6$) + Dual-Titan Llama-8B ($w=2.8$, `dp=True`) + Qwen3-14B ($w=3.0$) + Hepta-Gates + V42 Golden Anchor Hungarian ($\Delta \ge 0.35, P \ge 0.50$) + 100% Comp-Stack Bypassed | **0.6m** *(Fast Fixture 30s)* | **~245m** *(4h 05m)* | **`0.656`** 🥈 | `COMPLETE` | **CHẤM ĐIỂM HOÀN TẤT - ĐẠT `0.656` (+0.011 SO VỚI V45, +0.003 SO VỚI V46/V44)!** Thành công vang dội: (1) Sửa lỗi hoàn toàn crash của V47, chạy êm ru 100% không một vết gợn; (2) Điểm số bứt phá mạnh mẽ từ 0.645/0.653 lên **0.656**, trở thành điểm số cao thứ 2 trong toàn bộ lịch sử thi đấu của đội ta (vượt qua V43 `0.655`, V44 `0.653`, V39 `0.653`), chỉ kém kỷ lục V42 đúng 0.002! (3) Khẳng định chân lý: Việc kết hợp 0.663 SOTA + ModernBERT-Large context 8192 + Llama-8B $w=2.8$ trực giao + Hepta-Gates + Mỏ neo vàng V42 tạo nên một cỗ máy phân loại siêu ổn định và cực mạnh! |"""

new_v49 = """| **49** | 2026-10-30 10:28 | `57791141` | **Champion V49 Apex Dual-Titan Gauss**: Phục sinh Toàn diện Trọng tâm V42 (0.658 🥈) + Qwen14B (`gauss=0.70`, $w=3.0$) + Meta Llama-3.1-8B (`gauss=0.60`, $w=2.8$, `dp=True`) + ModernBERT-Large ($w=2.6$) + Nishant 14B ($w=2.5$) + Qwen3-8B (`gauss=0.44`, $w=2.0$, `dp=True`) + 20% ATLAS + Sinkhorn 0.45 + Hepta-Gates + V42 Golden Anchor Hungarian | **0.6m** *(Fast Fixture 30s)* | **~270m** *(4h 30m)* | **`0.654`** 🥈 | `COMPLETE` | **CHẤM ĐIỂM HOÀN TẤT - ĐẠT `0.654` (BẢO TOÀN VỮNG CHẮC VÙNG HUY CHƯƠNG BẠC TOÀN CẦU)!** Phân tích thực nghiệm so sánh đối đầu V48 vs V49: (1) Cả hai siêu phẩm đều hoàn thành xuất sắc trong giới hạn an toàn (V48 chạy 4h05m, V49 chạy 4h30m nhờ `dp=True`); (2) Việc thêm `gauss=0.60` vào Llama trong bối cảnh đã có 20% ATLAS Likelihood khiến các phân phối trung tâm prompt bị mượt hóa quá mức (over-smoothing) ở V49 (đạt 0.654), trong khi để Llama giữ nguyên phân phối token trực giao gốc ở V48 đem lại độ sắc bén cao hơn hẳn (đạt 0.656)! |"""

# Replace in content
idx48 = content.find(old_v48)
idx49 = content.find(old_v49)
idx_end49 = content.find("---", idx49)

content = content[:idx48] + new_v48 + "\n" + new_v49 + "\n" + content[idx_end49:]

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated abc SUBMISSIONS.md with official scores!")

