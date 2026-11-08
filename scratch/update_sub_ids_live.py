import re

with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace V48 and V49 rows with their official submission IDs
old_v48_row = '| **48** | 2026-10-30 10:09 | `Kernel Ready` | **Champion V48 Clean Sovereign**'
assert old_v48_row in content, "old_v48_row not found"

new_v48_row = '| **48** | 2026-10-30 10:28 | `57791134` | **Champion V48 Clean Sovereign**'

old_v49_row = '| **49** | 2026-10-30 10:13 | `Kernel Ready` | **Champion V49 Apex Dual-Titan Gauss**'
assert old_v49_row in content, "old_v49_row not found"

new_v49_row = '| **49** | 2026-10-30 10:28 | `57791141` | **Champion V49 Apex Dual-Titan Gauss**'

content = content.replace(old_v48_row, new_v48_row)
content = content.replace(old_v49_row, new_v49_row)

# Also update status column for V48 and V49 from READY to PENDING
content = content.replace('| **READY** | `COMPLETE` | **SỬA LỖI TRIỆT ĐỂ V47', '| **PENDING** | `PENDING` | **ĐÃ NỘP THÀNH CÔNG VÀO HÀNG ĐỢI KAGGLE (Ref: 57791134)!**')
content = content.replace('| **READY** | `COMPLETE` | **CÔNG THỨC VÔ ĐỊCH', '| **PENDING** | `PENDING` | **ĐÃ NỘP THÀNH CÔNG VÀO HÀNG ĐỢI KAGGLE (Ref: 57791141)!**')

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated abc SUBMISSIONS.md with live submission IDs!")

