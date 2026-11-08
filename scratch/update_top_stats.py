with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Update pending count and completed count
content = content.replace("- **Trạng thái bài nộp cuối cùng:** `PENDING` (V57 đang được Kaggle chấm điểm)", "- **Trạng thái bài nộp cuối cùng:** `0.655` (V57 đã hoàn tất chấm điểm)")
content = content.replace("- **Số lần nộp thành công (Scored):** **30** lần", "- **Số lần nộp thành công (Scored):** **31** lần")
content = content.replace("- **Số lần đang chờ kết quả (Pending):** **1** lần", "- **Số lần đang chờ kết quả (Pending):** **0** lần")

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write(content)
print('Updated top stats in SUBMISSIONS.md!')

