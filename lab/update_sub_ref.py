with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    subs = f.read()

subs = subs.replace(
    "| **62** | 2026-11-01 22:27 | `PENDING` |",
    "| **62** | 2026-11-01 22:32 | `57876354` |"
)

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write(subs)

print("Updated abc SUBMISSIONS.md with ref 57876354")

