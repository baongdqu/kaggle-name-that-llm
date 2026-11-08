leaderboard_content = """# BÁO CÁO PHÂN TÍCH BẢNG XẾP HẠNG TOÀN CẦU (abc LEADERBOARD.md)

> **Cuộc thi:** Kaggle Competition: Name That LLM - Identify the model behind the text  
> **Thời điểm trích xuất dữ liệu:** 01/11/2026 (00:50 AM Local / 17:50 PM UTC 31/10/2026)  
> **Tổng số đội tham gia toàn cầu:** **4,089 đội**  
> **Vị thế đội chúng ta:** **private** / `baonguyenduyquoc` (Team ID: `17137218`)  
> **Điểm kỷ lục cao nhất:** **`0.659`** (Champion V50 / V51) | **Rank: 522 / 4,089** (Top 12.7% toàn cầu)  
> **Khoảng cách tới Huy chương Đồng (Top 10% - Rank 408):** **`0.003 điểm`** (`0.662` vs `0.659`)  
> **Khoảng cách tới Huy chương Bạc (Top 5% - Rank 204):** **`0.004 điểm`** (`0.663` vs `0.659`)  
> **Khoảng cách tới Nhóm Dẫn Đầu (Top 1 - 10):** **`0.045 - 0.077 điểm`** (`0.704 - 0.736` vs `0.659`)

---

## 1. TỔNG QUAN PHÂN BỐ ĐIỂM SỐ & HIỆN TƯỢNG "KẸT XE PUBLIC" (THE 0.660 BOTTLENECK)

Cuộc thi hiện có **4,089 đội** tham gia với cấu trúc phân tầng điểm số vô cùng đặc thù:

```mermaid
graph TD
    subgraph "Tầng 1: Đỉnh Cao Độc Quyền (Top 10)"
        T1["Top 1 - 10: 0.704 - 0.736 (Lian Matsuo 0.736, Hod Carriers 0.732)"]
    end
    subgraph "Tầng 2: Vùng Huy Chương Bạc/Vàng (Rank 11 - 204)"
        T2["Rank 11 - 204: 0.663 - 0.703"]
    end
    subgraph "Tầng 3: Nút Thắt Kẹt Xe Public (Rank 205 - 600+)"
        T3["Rank 205 - 600+: 0.659 - 0.662 (>400 đội chen chúc trong biên độ 0.003)"]
        OUR["private (Đội chúng ta) - Rank 522: 0.659"]
    end
    subgraph "Tầng 4: Vùng Baseline Cơ Bản (Rank 601 - 4089)"
        T4["Rank 601+: < 0.658"]
    end
    T1 -.->|Cách biệt +0.075| T3
    T2 -.->|Cách biệt +0.004| T3
    T3 --- OUR
```

### Điểm Cắt Huy Chương (Medal Cutoffs - 4,089 Đội):
1. **Huy chương Vàng (Gold Medal - Top 0.2% + 10 = Top 18 đội):** Điểm cắt hiện tại khoảng **`0.685 - 0.700`**.
2. **Huy chương Bạc (Silver Medal - Top 5% = Rank 204):** Điểm cắt hiện tại là **`0.663`**.
3. **Huy chương Đồng (Bronze Medal - Top 10% = Rank 408):** Điểm cắt hiện tại là **`0.662`**.

> [!IMPORTANT]
> **Nhận định sống còn:** Hơn 350 - 400 đội đang bị "kẹt xe" ở mức `0.659 - 0.662` vì tất cả đều forking và blending từ các public baseline chung (Omer Faruk Merey 0.662, Alex Chilton 0.663, Joseph Adamski ATLAS). Đội chúng ta chỉ cần thêm **+0.003 đến +0.004 điểm** là lập tức nhảy vọt từ Rank 522 lên thẳng Rank 200 (Top 5% Silver Zone).

---

## 2. BẢNG XẾP HẠNG TOP 10 TOÀN CẦU (SNAPSHOT MỚI NHẤT)

| Hạng (#) | Đội thi (Team) | Thành viên (Members) | Điểm số (Score) | Số lần nộp (Entries) | Cập nhật gần nhất |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **1** 🥇 | **Lian Matsuo** | Lian Matsuo | **`0.736`** | 22 | 15h trước |
| **2** 🥈 | **Hod Carriers** | Hod Carriers | **`0.732`** | 47 | 1d trước |
| **3** 🥉 | **Bootstrapping** | Bootstrapping | **`0.714`** | 30 | 8h trước |
| **4** | **kulshekhar** | kulshekhar | **`0.714`** | 38 | 2d trước |
| **5** | **Le Viet** | Le Viet | **`0.711`** | 53 | 7h trước |
| **6** | **Harveen Singh Chadha** | Harveen Singh Chadha | **`0.706`** | 33 | 20h trước |
| **7** | **t.shiga** | t.shiga | **`0.705`** | 23 | 16h trước |
| **8** | **Team AllIs** | Team AllIs (5 Grandmasters) | **`0.704`** | 70 | 11h trước |
| **9** | **Kimiの名は** | Kimiの名は | **`0.704`** | 76 | 16h trước |
| **10** | **Adversarial** | Adversarial | **`0.702`** | 41 | 18h trước |

---

## 3. BẢNG SNAPSHOT KHU VỰC CỦA ĐỘI CHÚNG TA (RANK 517 - 527)

| Hạng (#) | Đội thi (Team) | Điểm số (Score) | Số lần nộp (Entries) | Khoảng cách với đội ta |
| :---: | :--- | :---: | :---: | :---: |
| **517** | david rozanes | `0.660` | 18 | +0.001 |
| **518** | TanChugh | `0.660` | 11 | +0.001 |
| **519** | Sergey Shcherbakov | `0.660` | 32 | +0.001 |
| **520** | Mustafa Toğluk | `0.660` | 5 | +0.001 |
| **521** | Nguyễn Đức Oanh | `0.660` | 26 | +0.001 |
| **522** 👑 | **private (Đội chúng ta)** | **`0.659`** | **57** | **MỎ NEO CHỦ LỰC** |
| **523** | F.A.Nina | `0.659` | 43 | 0.000 |
| **524** | Lingzhenxu | `0.659` | 11 | 0.000 |
| **525** | yupro | `0.659` | 8 | 0.000 |
| **526** | Khaled Yousef | `0.659` | 2 | 0.000 |
| **527** | dasfds | `0.659` | 13 | 0.000 |

---

## 4. PHÂN TÍCH KHOẢNG CÁCH KỸ THUẬT: TẠI SAO TOP 10 ĐẠT `0.704 - 0.736`?

Khoảng cách `+0.075` giữa Top 10 và nhóm `0.659` không thể giải quyết bằng việc cộng trừ trọng số thủ công lẻ ($w=2.2$ hay $2.4$). Nhóm dẫn đầu đang áp dụng 3 trụ cột kỹ thuật mang tính chuyển đổi hệ quy chiếu:

1. **Trụ cột 1: Tối ưu hóa Gán nhãn Bàn cờ (Prompt-Grid Latin Square Matching):**
   - Đề bài có đúng 18 mô hình trả lời cho mỗi prompt (mỗi model xuất hiện 1 lần duy nhất).
   - Bằng cách nhóm các văn bản test cùng prompt qua Dense Sentence Embeddings và chạy thuật toán phân bổ Hungarian Injective Assignment, họ triệt tiêu hoàn toàn lỗi dự đoán trùng lặp model trong cùng một prompt, mang lại **+0.030 đến +0.050 điểm tức thì**.
2. **Trụ cột 2: Đo lường Perplexity Token Trực tiếp từ Causal LLM (Zero-Shot Token Log-Likelihood):**
   - Sử dụng các mô hình gốc (Qwen 2.5 14B/32B, GLM-4 9B, Llama 3.1 8B) để tính toán trực tiếp Token Cross-Entropy Loss của đoạn văn bản khi điều kiện hóa trên system prompt. Model nào sinh ra văn bản đó sẽ có perplexity thấp nhất.
3. **Trụ cột 3: Meta-Learner Stacking (LightGBM/Ridge trên OOF Predictions + Stylometrics):**
   - Thay thế việc điều chỉnh trọng số trung bình cộng thủ công bằng mô hình Gradient Boosting có điều kiện theo độ dài văn bản, tỷ lệ dấu câu và ma trận xác suất OOF của các backbone.

---

## 5. KẾ HOẠCH HÀNH ĐỘNG ĐỂ BỨT PHÁ KHỎI VÙNG 500+ (01/11/2026 UTC)

1. **Khôi phục mỏ neo V50/V51 (`0.659`):** Giữ nguyên ensemble causal titan đã chứng minh tính ổn định cao nhất.
2. **Triển khai Prompt-Grid Hungarian Injective Post-Processing:** Áp dụng trên V50 để phá vỡ ngay nút thắt 0.660, tiến vào vùng `0.670 - 0.685`.
3. **Hiệu chỉnh Logit Drift của Sovereign ModernBERT:** Sử dụng Platt Scaling / Temperature Scaling trước khi hòa trộn, tránh làm loãng tín hiệu của Causal LLMs.
"""

with open('abc LEADERBOARD.md', 'w', encoding='utf-8') as f:
    f.write(leaderboard_content)

print('Updated abc LEADERBOARD.md successfully!')

