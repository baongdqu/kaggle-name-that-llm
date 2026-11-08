with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('| **67** | 2026-11-02 20:26 | `PENDING` |', '| **67** | 2026-11-02 20:29 | `57910001` |')
text = text.replace('| **Tổng số Submissions đã nộp** | **66 lần**', '| **Tổng số Submissions đã nộp** | **67 lần**')

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated Row 67 Ref ID in abc SUBMISSIONS.md successfully!")

