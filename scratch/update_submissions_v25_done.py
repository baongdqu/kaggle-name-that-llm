with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
updated = False
for line in lines:
    if '| **25** | 2026-10-25 20:18 | `57631244` |' in line:
        row25 = "| **25** | 2026-10-25 20:18 | `57631244` | **Champion V25 Apex Sovereign**: Comp Stack (25%) + Deep ATLAS (margin<0.38, 20%) + Qwen3-14B (0.60) | **0.8m** *(48s fixture)* | **232.3m** *(3h 52m)* | `0.630` | `COMPLETE` | Chấm điểm hoàn tất sau 232.3 phút! Đạt `0.630` (-0.009 so với 0.639). **Giải phẫu thực nghiệm chứng minh:** (1) `comp_stack` gốc là mô hình 0.563 của Lucas Boesen, khi nâng lên 25% đã kéo tụt độ chính xác của Two-Stage 14B; (2) ATLAS 8B Likelihood ở tỷ trọng 20% trên margin < 0.38 làm loãng tín hiệu của 75k Pool SVC; (3) Hạ `W_SVC` từ 1.0 về 0.85 làm mất độ sắc nét của 4,404 style heads. Điểm chính thức của đội vẫn bảo toàn nguyên vẹn ở mốc đỉnh `0.639` (Rank 152). |\n"
        new_lines.append(row25)
        updated = True
    else:
        new_lines.append(line)

if updated:
    with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("SUCCESSFULLY UPDATED abc SUBMISSIONS.md ROW 25 TO COMPLETE 0.630!")
else:
    print("FAILED TO FIND ROW 25!")
