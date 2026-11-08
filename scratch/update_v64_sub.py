import re

with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Update status section
text = re.sub(
    r'- \*\*Tổng số bài đã nộp:\*\* \*\*63 bài\*\* \(58 bài hoàn tất `COMPLETE`, \*\*0 bài đang chấm điểm\*\*\)\.',
    '- **Tổng số bài đã nộp:** **64 bài** (58 bài hoàn tất `COMPLETE`, **1 bài đang chấm điểm: Champion V64 - Ref `57898893`**).',
    text
)

text = re.sub(
    r'- \*\*Hạn mức hôm nay \(02/11 UTC\):\*\* Đã nộp \*\*1/5 lượt \(Champion V63 - Điểm chính thức: `0.662` 🥉👑\)\*\*, \*\*còn lại 4 lượt nộp bài\*\* cho ngày hôm nay\. Đội duy trì vững chắc kỷ lục tại \*\*Rank 312 / 4,150 đội\*\*\.',
    '- **Hạn mức hôm nay (02/11 UTC):** Đã nộp **2/5 lượt (Champion V63 `0.662` 🥉👑, Champion V64 đang chấm điểm - Ref `57898893`)**, **còn lại 3 lượt nộp bài** cho ngày hôm nay. Đội duy trì vững chắc kỷ lục tại **Rank 312 / 4,150 đội**.',
    text
)

# Update row 64
old_r64 = "| **64** | 2026-11-02 13:50 | `ĐANG COMMIT` |"
new_r64 = "| **64** | 2026-11-02 13:54 | `57898893` |"

text = text.replace(old_r64, new_r64)
text = text.replace("| **Chuẩn bị nộp** *(~240m)* | **`COMMITTING`** | `READY` |", "| **Đang chấm điểm ẩn** *(~240m)* | **`PENDING`** | `PENDING` |")

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated abc SUBMISSIONS.md for V64 PENDING Ref 57898893!")

