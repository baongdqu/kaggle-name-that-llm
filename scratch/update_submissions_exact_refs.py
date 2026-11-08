with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace placeholder *ref* with real submission refs
content = content.replace(
    "| **58** | 2026-11-01 08:35 | *ref* | **Champion V58",
    "| **58** | 2026-11-01 07:00 | 57853752 | **Champion V58"
)

content = content.replace(
    "| **59** | 2026-11-01 08:36 | *ref* | **Champion V59",
    "| **59** | 2026-11-01 07:00 | 57853771 | **Champion V59"
)

# Update pending count and remaining allowance
content = content.replace("- **Số lần đang chờ kết quả (Pending):** **0** lần", "- **Số lần đang chờ kết quả (Pending):** **2** lần (V58, V59)")
content = content.replace("- **Trạng thái bài nộp cuối cùng:** `0.655` (V57 đã hoàn tất chấm điểm)", "- **Trạng thái bài nộp cuối cùng:** `PENDING` (V58: 57853752, V59: 57853771 đang được Kaggle chấm điểm)")

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write(content)

print('Updated SUBMISSIONS.md with exact refs 57853752 and 57853771 successfully!')

