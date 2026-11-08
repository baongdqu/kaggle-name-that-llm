# Name That LLM - Competition Overview

## 1. Giới thiệu tổng quan (Description)
- **Tên cuộc thi:** Name That LLM (Identify the model behind the text)
- **Host:** Kaggle
- **Mục tiêu:** Xác định mô hình ngôn ngữ lớn (LLM) nào trong số **18 mô hình frontier** từ 6 phòng thí nghiệm AI (OpenAI, Google, Anthropic, Qwen, ZAI, Moonshot AI) đã sinh ra một đoạn văn bản ngắn (passage) chưa được gán nhãn.
- **Ý nghĩa:** Phục vụ cho việc xác thực nguồn gốc AI (AI provenance), kiểm toán dữ liệu tổng hợp (synthetic data auditing) và phát hiện ô nhiễm benchmark (benchmark contamination).
- **Đặc điểm dữ liệu:**
  - Khoảng **~4,000 đoạn văn bản** ở tập test.
  - Độ dài thay đổi ngẫu nhiên từ **40 đến 250 từ** (word count).
  - Không dựa vào các shortcut đơn giản (như số lượng ký tự hay từ khóa chủ đề) vì prompt bao quát nhiều chủ đề: triết học, vật lý, toán học, sinh học, khoa học máy tính và văn xuôi sáng tạo.
  - Cần khai thác các dấu vết phong cách (stylistic fingerprints): từ vựng, cách diễn đạt, cấu trúc câu, dấu câu, v.v.

---

## 2. Danh sách 18 Mô hình hợp lệ (Valid Labels)

| Nhà phát triển (Provider) | Tên mô hình (Model Name) |
| :--- | :--- |
| **OpenAI** | `gpt-5.5` |
| **OpenAI** | `gpt-5.6-luna` |
| **OpenAI** | `gpt-5.6-sol` |
| **OpenAI** | `gpt-5.6-terra` |
| **OpenAI** | `gpt-6-astra` |
| **Google** | `gemini-3.1-pro-preview` |
| **Google** | `gemini-3.6-flash` |
| **Google** | `gemini-3.7-flash` |
| **Google** | `gemini-3.8-flash` |
| **Anthropic** | `claude-fable-5` |
| **Anthropic** | `claude-opus-4-6` |
| **Anthropic** | `claude-opus-4-8` |
| **Anthropic** | `claude-opus-5` |
| **Anthropic** | `claude-sonnet-5` |
| **Qwen** | `qwen3.8-max` |
| **ZAI** | `glm-5.3` |
| **ZAI** | `glm-5.3-flash` |
| **Moonshot AI** | `kimi-k3` |

> [!WARNING]
> Mỗi dự đoán trong file nộp phải **chính xác 100%** là một trong 18 tên mô hình trên. Nếu sai chính tả hoặc thừa nhãn ngoài danh sách, bài nộp sẽ bị từ chối.

---

## 3. Thang đo đánh giá (Evaluation Metric)

- **Metric:** **Balanced Accuracy Score** (Macro Recall)
  $$\text{Balanced Accuracy} = \frac{1}{K} \sum_{i=1}^K \text{Recall}_i$$
  Trong đó:
  - $K = 18$ (số lượng model dự đoán).
  - Tỉ lệ phân bổ số đoạn văn bản giữa các mô hình **không đồng đều** và phân bổ giữa Public / Private Leaderboard có sự khác nhau.
  - Balanced Accuracy gán trọng số ngang nhau cho mọi class (mỗi model có vai trò như nhau), tránh bias vào class chiếm đa số.

---

## 4. Quy định & Định dạng nộp bài (Submission & Code Requirements)

- **Hình thức cuộc thi:** Code Competition (chạy qua Kaggle Notebooks).
- **File nộp:** Bắt buộc xuất ra file `submission.parquet` trong thư mục working directory (`/kaggle/working/submission.parquet`).
- **Input test:** Đọc từ `/kaggle/input/competitions/name-that-llm/test.csv`.
- **Cấu trúc file nộp:**
  - Gồm 2 cột: `id`, `model`.
  - Mọi `id` xuất hiện đúng một lần, không có giá trị rỗng/null.
- **Ràng buộc tài nguyên Notebook:**
  - Thời gian chạy tối đa: **$\le$ 9 giờ** (cả CPU Notebook lẫn GPU Notebook).
  - **Internet access:** Bị vô hiệu hóa (Disabled). Các pretrained model / checkpoint / thư viện bên ngoài phải được thêm dưới dạng Kaggle Dataset.
  - **External data:** Cho phép sử dụng dữ liệu và pretrained model công khai, miễn phí.

### Code submission mẫu:
```python
import pandas as pd

test = pd.read_csv('/kaggle/input/competitions/name-that-llm/test.csv')

# predictions là mảng / Series chứa nhãn dự đoán cho từng hàng
submission = pd.DataFrame({
    'id': test['id'],
    'model': predictions
})

submission.to_parquet('submission.parquet', index=False)
```

---

## 5. Mốc thời gian (Timeline)

- **Ngày bắt đầu:** 14/10/2026
- **Hạn chót nhập cuộc (Entry Deadline):** 06/01/2027
- **Hạn chót ghép đội (Team Merger Deadline):** 06/01/2027
- **Hạn chót nộp bài cuối cùng (Final Submission Deadline):** 13/01/2027 (11:59 PM UTC)

---

## 6. Giải thưởng (Prizes)
- **Tổng giải thưởng:** $50,000 USD
- **Phân bổ:** Top 1 đến Top 10 đều nhận **$5,000 USD/giải**.
