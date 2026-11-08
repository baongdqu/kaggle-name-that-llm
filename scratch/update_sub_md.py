with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_row = '| **69** | 2026-11-03 14:25 | `57937986` | **Champion V69 Mixture of Specialists (MoS) Apex**: **0.662 V68 Golden Anchor Base** + **Stage-2 Selective Stylometric Specialists** (Function Words / Burrows\' Delta + Char N-Grams 3-5 + Sentence Stats) cho **Gemini Flash Trio** & **GLM 5.3 Duo** (ngưỡng tự tin $\\ge 0.60$) + **Hiệu chuẩn Độ dài Trực giao** ($w=0.05$) + **Lá chắn Latin-Square Kuhn-Munkres** ($\\text{size} \\ge 5$) + **Damped Sinkhorn 0.45** | **0.6m** *(Fast Fixture 35s)* | **~240m** *(4h 00m)* | *PENDING* | `PENDING` | **ĐỘT PHÁ TRIỆT TIÊU BAYES BOTTLENECK CỦA HỌ ANH EM RUỘT:** Giữ nguyên vẹn 100% mỏ neo 0.662 và lá chắn Hungarian bảo vệ 338 mẫu. Bổ sung Stage-2 MoS chạy suy luận độc lập trên hai cụm anh em gây ra 159 lỗi nhầm lẫn nghiêm trọng nhất (Gemini Flash 3.6/3.7/3.8 và GLM 5.3/5.3-flash). Khi specialist đạt độ tin cậy $\\ge 60%$, ghi đè nhãn anh em chính xác, không ảnh hưởng đến bất kỳ 13 nhãn nào khác. |\n'

for i, line in enumerate(lines):
    if line.strip().startswith('| **68**'):
        lines.insert(i, new_row)
        print(f'Inserted V69 before line {i}')
        break

# Update V68 score from PENDING to 0.662
for j in range(len(lines)):
    if lines[j].strip().startswith('| **68**'):
        lines[j] = lines[j].replace('*PENDING* | `PENDING`', '**0.662** | `0.662`')
        print(f'Updated V68 score to 0.662 at line {j}')
        break

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('abc SUBMISSIONS.md updated successfully!')

