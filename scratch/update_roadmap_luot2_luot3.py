import re

with open('EXPERIMENTS_REVIEW_AND_ROADMAP_TOP1.md', 'r', encoding='utf-8') as f:
    text = f.read()

part2_anchor = '## PHẦN II: CHIẾN LƯỢC BỨT PHÁ ĐIỂM SỐ (LỘ TRÌNH 3 BƯỚC ĐOẠT TOP 1)'
assert part2_anchor in text, 'Anchor not found'

part2_new = """## PHẦN II: CHIẾN LƯỢC 3 LƯỢT SUBMIT BỨT PHÁ TOP 1 KAGGLE (V4 -> V5 -> V6)

Bảng tổng hợp chiến lược 3 lượt nộp bài hôm nay:

| Lượt Submit | Phiên bản Đề xuất | Nội dung Cài đặt Cụ thể | Mục tiêu Kiểm chứng | Kỳ vọng Điểm số |
| :---: | :--- | :--- | :--- | :---: |
| **Lượt 1** | **V4: Calibrated QuadGate + 18-Class Stylometrics** | Kết hợp 4 Gate chuyên biệt (Claude, OpenAI, Gemini, GLM) + Chuẩn hóa Z-score/CalibratedClassifierCV cho đầu ra 4 Gate trước khi cộng logit + Hòa trộn 18-Class Stylometrics. | Xác nhận sức mạnh của External Data trên cả 4 họ và 18 lớp toàn cục, giải quyết triệt để độ lệch thang đo giữa các Gate. | **0.558 – 0.562+** (Vào Top 4) |
| **Lượt 2** | **V5: V4 + Micro-Formatting + Intra-Cluster Gate** | Thêm ~25 đặc trưng định dạng Markdown & khoảng trắng (`\\n\\n`, `—`, `**`, `*`, indents, quotes) + Chuẩn hóa điểm Gate theo từng cụm Prompt (Prompt Cluster Mean-Centering). | Tấn công trực diện vào lỗi nhầm lẫn nội bộ gay gắt nhất (`luna`/`sol`/`terra` & `opus 4.6`/`4.8`/`5`), triệt tiêu bias prompt. | **0.568 – 0.574+** (Vào Top 3 Podium) |
| **Lượt 3** | **V6: V5 + Margin-Locked Sinkhorn + Pseudo-Label Refit** | Khóa cứng các mẫu có margin tự tin cao ($P_{\\text{top1}} - P_{\\text{top2}} > 0.35$), điều chỉnh Quota Sinkhorn cho phần còn lại + Transductive Pseudo-labeling refit cho nhánh Style N-grams LinearSVC. | Tối ưu hóa điểm số cuối cùng cho thước đo Macro-Recall, san đều nhãn ở vùng ranh giới và thích ứng trực tiếp với test set ẩn. | **0.582 – 0.586+** (Chiếm lĩnh TOP 1) |

---

### 🚀 CHI TIẾT KỸ THUẬT LƯỢT 1: V4 (CALIBRATED QUADGATE + 18-CLASS STYLOMETRICS)
1. **Calibrated QuadGate System:**
   - **4 bộ Gate chuyên biệt:**
     - `Claude Gate`: 5 model Anthropic (huấn luyện trên 5,000 API passages ngoài).
     - `OpenAI Gate`: 5 model GPT (huấn luyện trên 9,808 mẫu external corpus).
     - `Gemini Gate`: 4 model Google Gemini (`gemini-3.1-pro-preview`, `3.6-flash`, `3.7-flash`, `3.8-flash`).
     - `GLM Gate`: 2 model ZAI GLM (`glm-5.3`, `glm-5.3-flash`).
   - **Chuẩn hóa Z-Score cho điểm quyết định (Decision Function Standardization):**
     - Vì GLM là phân loại nhị phân (binary), thang đo của `decision_function` có độ biến thiên khác biệt so với multiclass của OpenAI (5 lớp) và Gemini (4 lớp).
     - Áp dụng chuẩn hóa Z-score:
       $$\\widetilde{\\text{DC}}_{fam} = \\frac{\\text{DC}_{fam} - \\mu_{fam}}{\\sigma_{fam} + 10^{-6}}$$
     - Đảm bảo điểm boost từ 4 Gate có cùng thang độ phân tán, không làm lấn át các họ mô hình khác.
2. **Universal 18-Class Stylometrics Blending:**
   - Huấn luyện LinearSVC 18 lớp trên toàn bộ 27,000 mẫu ngoài + tập train với n-grams hình thái và hư từ (`char 3-5`, `shape 2-6`, `function 1-3`).
   - Hòa trộn xác suất trước khi đưa vào Sinkhorn mềm ($\\\\alpha = 0.5$).

---

### 🚀 CHI TIẾT KỸ THUẬT LƯỢT 2: V5 (MICRO-FORMATTING + INTRA-CLUSTER GATE CENTERING)
1. **Bộ ~25 Đặc trưng Vi định dạng (Micro-Formatting & Typographic Features):**
   - Các mô hình cùng họ thường có thói quen sinh token cú pháp rất riêng:
     - **Khoảng trắng & Xuống dòng:**
       - Tần suất ngắt dòng kép `\\n\\n` vs ngắt dòng đơn `\\n` (GPT-5.6 có nhịp ngắt đoạn thưa hơn Claude).
       - Tỷ lệ khoảng trắng so với toàn văn bản (`whitespace_ratio`).
       - Độ dài trung bình các đoạn văn bản (`mean_paragraph_length`, `std_paragraph_length`).
     - **Cú pháp Markdown:**
       - Mật độ in đậm (`**...**` hoặc `__...__`).
       - Mật độ in nghiêng (`*...*` hoặc `_..._`).
       - Mật độ danh sách gạch đầu dòng (`- `, `* `, `1. `).
       - Khối mã nguồn (````code````) và code nội dòng (``` `inline` ```).
       - Dấu trích dẫn (`> `).
     - **Dấu câu đặc thù:**
       - Dấu gạch ngang dài em-dash `—` (đặc trưng nổi tiếng của Claude Opus) vs en-dash `–` vs gạch nối `-`.
       - Dấu chấm lửng unicode `…` vs 3 dấu chấm thường `...`.
       - Ngoặc kép cong `“ ”` vs thẳng `\" \"`.
       - Mật độ dấu hai chấm `:`, chấm phẩy `;`, chấm cảm `!` và chấm hỏi `?`.
     - **Từ nối mở đầu (Sentence Openers):**
       - Tần suất câu bắt đầu bằng: `However`, `Furthermore`, `Moreover`, `Therefore`, `Specifically`, `Indeed`.
       - Tần suất câu bắt đầu bằng đại từ chỉ định: `This`, `These`, `It`, `They`.
   - Toàn bộ 25 đặc trưng được đưa vào mô hình LightGBM Stylometrics và chuẩn hóa StandardScaler.
2. **Intra-Cluster Gate Mean-Centering (Triệt tiêu Bias Prompt trong từng Gate):**
   - Vấn đề: Mỗi prompt hỏi một chủ đề khác nhau, khiến điểm Gate bị trôi (prompt shift).
   - Giải pháp: Sử dụng ma trận cụm prompt từ phân cụm phân cấp (TF-IDF content-words + GLM layer 24 cosine distance cut tại 0.60).
   - Với mỗi cụm gồm $M$ mẫu ($2 \\le M \\le 18$):
     $$\\text{DC}_{gate}^{\\text{centered}} = \\text{DC}_{gate} - \\text{Mean}_{cluster}(\\text{DC}_{gate})$$
   - Giúp Gate so sánh độ chênh lệch văn phong thực sự giữa các mô hình đang trả lời trên CÙNG một đề bài!
   - Tấn công trực diện vào cặp nhầm lẫn khó nhất:
     - `gpt-5.6-luna` vs `gpt-5.6-sol` vs `gpt-5.6-terra`
     - `claude-opus-4-6` vs `claude-opus-4-8` vs `claude-opus-5`

---

### 🚀 CHI TIẾT KỸ THUẬT LƯỢT 3: V6 (MARGIN-LOCKED SINKHORN + TRANSDUCTIVE PSEUDO-LABELING)
1. **Margin-Locked Soft Sinkhorn Balancing:**
   - Khắc phục hoàn toàn sai lầm của V3:
     - Tính khoảng cách tự tin giữa nhãn số 1 và nhãn số 2 cho từng mẫu test:
       $$\\text{Margin}_i = P_{i, \\text{top1}} - P_{i, \\text{top2}}$$
     - **Điều kiện Khóa Mẫu (Anchor Lock):** Nếu $\\text{Margin}_i > 0.35$ (hoặc $P_{i, \\text{top1}} > 0.60$):
       - Mẫu $i$ được xem là "Chắc chắn đúng" (High-Confidence Anchor).
       - Khóa cứng nhãn dự đoán cho mẫu này: $\\text{pred}_i = \\text{top1}_i$.
     - **Cân bằng Quota còn lại (Residual Sinkhorn Balancing):**
       - Trừ số lượng mẫu đã khóa khỏi quota mục tiêu của 18 lớp:
         $$\\text{QuotaRemaining}_c = \\max\\left(1, \\frac{N}{18} - \\text{LockedCount}_c\\right)$$
       - Chỉ áp dụng Sinkhorn balancing trên các mẫu chưa khóa ($\\\\text{Margin}_i \\le 0.35$).
       - Đảm bảo $100\\%$ mẫu tự tin cao không bị xê dịch, trong khi toàn bộ các mẫu mập mờ được điều phối san phẳng để tối đa hóa Macro-Recall!
2. **Transductive Pseudo-Labeling Refit:**
   - Lấy top các mẫu test ẩn có độ tự tin cao nhất ($\\\\text{Margin} > 0.40$) làm pseudo-labels.
   - Thêm các mẫu này vào tập train và huấn luyện lại nhánh Style N-grams LinearSVC (nhánh có trọng số lớn nhất `1.516`).
   - Mô hình học trực tiếp đặc tính văn phong từ chính tập test ẩn hiện tại, bứt phá ngưỡng **0.582 – 0.586+ để đoạt TOP 1 LEADERBOARD**!"""

prefix = text.split(part2_anchor)[0]
full_updated = prefix + part2_new + '\n\n## PHẦN III: NGUYÊN TẮC THI ĐẤU & QUẢN TRỊ RỦI RO' + text.split('## PHẦN III: NGUYÊN TẮC THI ĐẤU & QUẢN TRỊ RỦI RO')[1]

with open('EXPERIMENTS_REVIEW_AND_ROADMAP_TOP1.md', 'w', encoding='utf-8') as f:
    f.write(full_updated)

print('Updated EXPERIMENTS_REVIEW_AND_ROADMAP_TOP1.md successfully!')
