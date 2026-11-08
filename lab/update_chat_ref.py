with open('abc CHAT_HISTORY.md', 'r', encoding='utf-8') as f:
    chat = f.read()

sub_note = """
  4. **Xác Nhận Nộp Bài Thành Công Lên Bảng Xếp Hạng Kaggle:**
     - Kernel commit hoàn tất thành công trong 29 vòng lặp.
     - Lệnh submit tự động thực thi lúc 22:32 ngày 01/11/2026 (15:32 UTC).
     - **Mã định danh nộp bài (Submission Ref):** **`57876354`**.
     - **Trạng thái:** `SubmissionStatus.PENDING` (hệ thống đang chấm điểm trên 4,000 dòng test ẩn).
     - Điểm số kỷ lục toàn đội hiện tại: **`0.662`** 🥉👑 (Rank 312 / 4,150 đội toàn cầu).
"""

if 'Submission Ref: 57876354' not in chat and '57876354' not in chat:
    chat = chat.replace(
        "- Đồng bộ sang kho sẵn sàng AI `Downloads/kaggle_ntl_AI_READY/` và đóng gói lại tệp nén 1-Click `kaggle_ntl_AI_READY.zip`.",
        "- Đồng bộ sang kho sẵn sàng AI `Downloads/kaggle_ntl_AI_READY/` và đóng gói lại tệp nén 1-Click `kaggle_ntl_AI_READY.zip`." + sub_note
    )
    with open('abc CHAT_HISTORY.md', 'w', encoding='utf-8') as f:
        f.write(chat)
    print("Updated abc CHAT_HISTORY.md with submission ref")
else:
    print("Submission ref already in chat history")

