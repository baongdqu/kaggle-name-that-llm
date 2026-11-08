import re

with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Update status
text = re.sub(
    r'- \*\*Tổng số bài đã nộp:\*\* \*\*62 bài\*\*.*?\*Champion V63 đã đóng gói & commit xong trên Kaggle, sẵn sàng nộp lúc 07:00 sáng!\*',
    '- **Tổng số bài đã nộp:** **63 bài** (57 bài hoàn tất `COMPLETE`, **1 bài đang chấm điểm: Champion V63 - Ref `57887198`**)',
    text, flags=re.DOTALL
)

text = re.sub(
    r'- \*\*Hạn mức hôm nay \(01/11 UTC\):\*\* Hôm nay \*\*đã sử dụng trọn vẹn 5/5 lượt.*?\n',
    '- **Hạn mức hôm nay (02/11 UTC):** Đã nộp **1/5 lượt (Champion V63 lúc 07:00 AM VN / 00:00 UTC)**, **còn lại 4 lượt nộp bài** cho ngày hôm nay. Champion V63 mang mã Ref `57887198`, đang trong tiến trình chấm điểm trên hệ thống Kaggle ngầm (~4 tiếng).\n',
    text
)

# Update row 63
text = text.replace(
    '| **63** | 2026-11-02 07:00 | `SẴN SÀNG` |',
    '| **63** | 2026-11-02 07:00 | `57887198` |'
)
text = text.replace(
    '| **0.6m** *(Fast Fixture 35s)* | **Chờ nộp lúc 07:00 sáng** *(~240m)* | **`CHỜ 07:00`** | `READY` |',
    '| **0.6m** *(Fast Fixture 35s)* | **Đang chấm điểm ẩn** *(~240m)* | **`PENDING`** | `PENDING` |'
)

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated abc SUBMISSIONS.md successfully!")

