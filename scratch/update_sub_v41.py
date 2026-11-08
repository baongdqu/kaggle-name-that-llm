with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Update total submissions count
text = text.replace("- **Tổng số bài đã nộp:** **40 bài**", "- **Tổng số bài đã nộp:** **41 bài**")
text = text.replace("Sau V39 & V40, hôm nay **còn lại 1 lượt nộp bài**", "Sau V41, hôm nay **đã sử dụng trọn vẹn 5/5 lượt nộp bài** (0 lượt còn lại, hạn mức sẽ reset về 5 lượt vào 07:00 sáng nay / 00:00 UTC)")

# Append row 41 to table
row_40_marker = "Champion V40 Sovereign Trinity Lean"
idx = text.find(row_40_marker)
end_of_row_40 = text.find("\n", idx)
end_of_row_40 = text.find("\n", end_of_row_40 + 1)

v41_row = """
| **41** | 2026-10-29 05:00 | `57749696` | **Champion V41 Apex Titan Sovereign**: Meta Llama-3.1-8B ($w=2.6$, Full 5 Folds) + **Llama Shared-Prompt Gaussian Head (`gauss=0.60`)** + ModernBERT Amplified ($w=2.6$) + Qwen 14B/8B ($w=2.6/1.6$) | **2.2m** *(Fast Fixture 30s)* | *Đang chấm điểm...* | **PENDING** | `PENDING` | **CHÍNH THỨC NỘP BÀI - LẦN ĐẦU TIÊN KÍCH HOẠT SHARED-PROMPT GAUSSIAN HEAD CHO LLAMA (`gauss=0.60`)!** Khắc phục điểm mù chí mạng của V39: cấp phát bộ phân loại sinh Gaussian LDA trên không gian 128k token của Llama để bắt tương quan cụm prompt. Đưa Llama-3.1-8B lên $w=2.6$ ngang hàng Qwen-14B, khuếch đại ModernBERT lên $w=2.6$. Tận dụng chuẩn xác lượt nộp cuối cùng của ngày 28/10 UTC trước giờ reset 07:00. Mục tiêu: Phá vỡ kỷ lục 0.653 tiến sát Top 50 (`0.658 - 0.665+`)! |"""

text = text[:end_of_row_40] + v41_row + text[end_of_row_40:]

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated abc SUBMISSIONS.md with V41 successfully")

