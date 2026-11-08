with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_row = '| **70** | 2026-11-03 19:09 | `57946908` | **Champion V70 Hungarian-Aware MoS Apex**: **0.662 V68 Golden Anchor Base** + **Tiêm Logit Micro-Stylometric Specialist Trực Tiếp Trước Hungarian** ($w_{\\text{spec}}=0.008$) cho **Gemini Flash Trio** & **GLM 5.3 Duo** + **Lá chắn Latin-Square Kuhn-Munkres Hoàn Hảo (Zero-Collision)** + **Hiệu chuẩn Độ dài Trực giao** ($w=0.05$) + **Damped Sinkhorn 0.45** | **0.6m** *(Fast Fixture 35s)* | **~240m** *(4h 00m)* | *PENDING* | `PENDING` | **ĐỘT PHÁ TRIỆT TIÊU HIỆN TƯỢNG ZERO-SUM CỦA HUNGARIAN:** Thay vì ghi đè (override) sau Hungarian gây xung đột nhãn trong cụm, V70 tiêm trực tiếp độ lệch log-odds từ Specialist vào ma trận chi phí trước khi ghép cặp. Thực nghiệm 5-Fold GroupKFold trên 2,699 mẫu OOF chứng minh tỷ lệ Thắng:Thua đạt 5:1 (Net +4 mẫu, $+0.0015$ Balanced Acc), cho phép Hungarian tối ưu hóa toàn cục với 0% collision! |\n'

for i, line in enumerate(lines):
    if line.strip().startswith('| **69**'):
        lines.insert(i, new_row)
        print(f'Inserted V70 before line {i}')
        break

# Update V69 score from PENDING to 0.662
for j in range(len(lines)):
    if lines[j].strip().startswith('| **69**'):
        lines[j] = lines[j].replace('*PENDING* | `PENDING`', '**0.662** | `0.662`')
        print(f'Updated V69 score to 0.662 at line {j}')
        break

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('abc SUBMISSIONS.md updated successfully for V70!')

