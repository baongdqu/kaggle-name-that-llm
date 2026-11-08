# Name That LLM - Dataset Overview (DATA.md)

## 1. Mô tả tập dữ liệu (Dataset Description)
- Dữ liệu bao gồm các đoạn văn hoàn chỉnh (self-contained paragraphs) được sinh ra bởi **18 mô hình ngôn ngữ lớn (LLM)** thuộc các chủ đề đa dạng.
- Mỗi đoạn văn là câu trả lời xác thực từ LLM (authentic LLM completion).
- Dữ liệu chỉ qua xử lý làm sạch định dạng nhẹ (ví dụ: loại bỏ thẻ `**in đậm**` bao quanh cả đoạn).
- **Tổng dung lượng:** ~2.14 MB.
- **License:** CC BY-SA 4.0.

---

## 2. Chi tiết các tệp dữ liệu (Files)

### 2.1. `train.csv` (Tập huấn luyện)
- **Số lượng mẫu:** **2,699 đoạn văn** (passages) được sinh ra từ **150 prompt khác nhau**.
- **Cấu trúc các cột:**
  - `id` *(string)*: Định danh ẩn danh duy nhất cho từng đoạn văn.
  - `prompt_id` *(string)*: Mã định danh ẩn danh của prompt mà đoạn văn đó trả lời.
  - `text` *(string)*: Đoạn văn bản do mô hình sinh ra.
  - `model` *(string)*: Tên mô hình ground-truth sinh ra văn bản (một trong 18 mô hình hợp lệ).

> [!NOTE]
> Trung bình mỗi prompt trong tập train có khoảng 18 mô hình cùng trả lời ($150 \times 18 \approx 2,700$). Cột `prompt_id` cho phép nhóm các câu trả lời cùng một câu hỏi lại với nhau.

---

### 2.2. `test.csv` (Tập kiểm tra)
- **Interactive session (lúc code/debug):** Chỉ chứa **3 dòng ví dụ** để test pipeline chạy end-to-end.
- **Hidden rerun (lúc submit chấm điểm):** Chứa khoảng **~4,000 đoạn văn bản**.
- **Quan trọng:** Các passage trong `test.csv` được sinh ra từ **các prompt hoàn toàn mới, không xuất hiện trong `train.csv`** (unseen prompts).
- **Cấu trúc các cột:**
  - `id` *(string)*: Định danh ẩn danh duy nhất cho đoạn văn test.
  - `text` *(string)*: Đoạn văn bản chưa được gắn nhãn cần dự đoán.

> [!IMPORTANT]
> Vì test set dùng **unseen prompts**, mô hình không được phụ thuộc vào chủ đề/ngữ nghĩa của prompt (leak semantic content), mà phải tập trung học các đặc trưng văn phong, ngữ pháp, độ dài, cách dùng từ (stylistic fingerprints).

---

### 2.3. `sample_submission.parquet`
- File mẫu định dạng nộp bài (Parquet format).
- **Cấu trúc các cột:**
  - `id` *(string)*: Khớp với từng `id` trong `test.csv`.
  - `model` *(string)*: Tên mô hình dự đoán (ví dụ: `gpt-5.5`, `claude-opus-5`, `gemini-3.8-flash`, v.v.).

---

## 3. Tổng hợp cấu trúc dữ liệu

| Tên File | Định dạng | Số dòng | Cột dữ liệu | Ghi chú |
| :--- | :--- | :--- | :--- | :--- |
| `train.csv` | CSV | 2,699 | `id`, `prompt_id`, `text`, `model` | 150 prompt gốc |
| `test.csv` | CSV | ~4,000 (Hidden) / 3 (Local) | `id`, `text` | Prompt hoàn toàn mới, không có `prompt_id` |
| `sample_submission.parquet` | Parquet | Tương ứng `test.csv` | `id`, `model` | Định dạng bắt buộc nộp bài |

---

## 4. Những lưu ý quan trọng khi xây dựng mô hình (Modeling Insights)
1. **Group / Split Validation Strategy:**
   - Khi chia tập Train / Validation (K-Fold), **bắt buộc phải chia theo `prompt_id`** (ví dụ dùng `GroupKFold` hoặc `StratifiedGroupKFold`) để đảm bảo không rò rỉ prompt giữa train và validation, mô phỏng đúng kịch bản của `test.csv`.
2. **Kích thước mẫu tương đối nhỏ (~2,700 mẫu train):**
   - Số lượng mẫu của mỗi mô hình vào khoảng $2699 / 18 \approx 150$ mẫu. Cần chú ý tránh overfitting.
   - Có thể tận dụng dữ liệu ngoại vi (external data) hoặc sinh thêm synthetic data từ các API nếu cần, miễn là tuân thủ quy định cuộc thi.
3. **Độ dài văn bản:**
   - Dao động từ 40 đến 250 từ. Các mô hình transformer phân loại văn bản (như DeBERTa-v3, RoBERTa, hoặc LLM fine-tuning/prompting) hoàn toàn có thể xử lý nguyên văn đoạn văn mà không lo bị cắt xén (max token < 512).

---

## 5. 🔬 Phát hiện nâng cao từ thực chiến (Advanced Insights from Competition) — *Cập nhật 2026-11-05*

### 5.1. Cấu trúc nội tại của Test Set: **Prompt-Row Clustering (Prompt Clusters)** ⭐ **KEY DISCOVERY**
> **Khám phá từ phân tích pháp y (Forensic Analysis) các submission V76-V80 (0.662 sàn) & kernel đối thủ `ntl-0668-stack-pool-prompt-rows`**

- **Bản chất thực sự**: `test.csv` (~4,000 passages) **KHÔNG phải 4,000 prompt độc lập**. Thay vào đó, các passage được gom thành **các cụm (clusters) theo `prompt_id` ẩn** — mỗi prompt được đưa cho **18 mô hình cùng trả lời** → tạo thành 1 cụm 18 dòng test có chung prompt.
- **Hệ quả**: Số lượng prompt unique trong test ≈ $4000 / 18 \approx 222$ prompt.
- **Ràng buộc loại trừ lẫn nhau (Mutual Exclusivity Constraint)**: Trong mỗi cụm prompt, **mỗi mô hình chỉ xuất hiện đúng 1 lần** (bijection: 18 models ↔ 18 responses). Đây là **ràng buộc cứng (hard constraint)** của cuộc thi.

### 5.2. Chiến lược khai thác: **Test-Internal Hungarian Matching / Global Assignment**
- Vì ràng buộc trên, bài toán dự đoán có thể chuyển từ **phân loại độc lập từng dòng** → **bài toán gán nhãn tối ưu toàn cục (Global Optimal Assignment)** cho từng cụm prompt.
- **Thuật toán**: Hungarian Algorithm (Kuhn-Munkres) trên ma trận cost = $-log(prob)$ hoặc $1 - prob$ từ classifier, áp dụng **riêng cho từng cụm prompt**.
- **Tác động điểm số**: Đối thủ Top 10 (0.710–0.747) đạt được nhờ khai thác triệt để ràng buộc này. Chúng ta hiện tại **CHƯA khai thác** → sàn 0.662.

### 5.3. Danh sách 18 mô hình mục tiêu (Target Models) — *Xác nhận từ sample_submission & leaderboard*
| Model Family | Các biến thể (Variants) |
| :--- | :--- |
| **GPT** | `gpt-5.5`, `gpt-5.5-pro`, `gpt-5.5-codex`, `gpt-5.5-codex-max`, `gpt-5.5-codex-mini`, `gpt-5.6-luna`, `gpt-5.6-luna-pro`, `gpt-5.6-sol`, `gpt-5.6-sol-pro`, `gpt-5.6-terra`, `gpt-5.6-terra-pro`, `gpt-6-astra`, `gpt-6-astra-pro`, `gpt-6-luna`, `gpt-6-luna-pro`, `gpt-6-sol`, `gpt-6-sol-pro`, `gpt-6.1-sol`, `gpt-6.1-sol-pro`, `gpt-6.1-terra`, `gpt-6.1-terra-pro` |
| **Claude** | `claude-opus-5`, `claude-opus-5.5`, `claude-sonnet-5`, `claude-sonnet-5.5`, `claude-haiku-5.5`, `claude-fable-5`, `claude-fable-5.1` |
| **Gemini** | `gemini-3.8-flash`, `gemini-3.8-flash:batch`, `gemini-4-argon`, `gemini-4-argon:batch`, `gemini-3.7-flash`, `gemini-3.7-flash:batch`, `gemini-3.6-flash`, `gemini-3.6-flash:batch`, `gemini-3.5-flash`, `gemini-3.5-flash:batch`, `gemini-3.1-pro-preview`, `gemini-3.1-pro-preview:batch` |
| **Llama** | `llama-4-maverick`, `llama-4-scout`, `llama-3.3-70b-instruct` |
| **Qwen** | `qwen3-235b-a22b-thinking-2507`, `qwen3-coder-30b-a3b-instruct`, `qwen3-next-80b-a3b-instruct`, `qwen3.5-plus-20260420`, `qwen3.8-max-prime` |
| **Others** | `deepseek-v3.2`, `deepseek-v4-pro`, `kimi-k3`, `kimi-k3.1-pro`, `glm-5.3-prime`, `nemotron-3-ultra-550b-a55b`, `grok-4.7`, `hy4` |

> **Lưu ý**: Danh sách trên là tập hợp từ sample_submission + kernel metadata + leaderboard analysis. Tập 18 model *thực tế* trong test set là subset cố định của danh sách này. Cần xác định chính xác 18 model nào xuất hiện trong test (có thể khác nhau giữa public/private LB).

### 5.4. Cập nhật Modeling Insights với kiến thức mới
4. **Prompt-Aware Validation**: Ngoài `GroupKFold(prompt_id)`, có thể mô phỏng test-time clustering bằng cách tạo **pseudo-clusters** trong validation (gộp 18 mẫu khác model cùng prompt) và đánh giá Hungarian accuracy thay vì per-sample accuracy.
5. **Stylometric Features are King**: Vì test dùng unseen prompts, **semantic content leakage = 0**. Chỉ **stylistic fingerprints** (độ dài, punctuation, n-grams, formatting, reasoning patterns, token usage) mới generalizable.
6. **Sibling Confusion is Real**: Các model cùng family (Gemini Flash variants, GPT-5.x variants, Claude Sonnet/Opus/Haiku) cực kỳ khó tách biệt → cần **contrastive learning / sibling-specific heads** (như Sibling LoRA L24) nhưng **chỉ áp dụng có chọn lọc (Expert Routing)** để tránh nhiễu.
7. **External Pool Matching**: Clean Open Pool (198k passages) + Exact/N-gram/Embedding matching → **Hard Override** cho các test case match được → giảm tải cho classifier trên phần còn lại.
8. **⭐ Test Set Structure — Prompt Clusters với Sparse Distribution (05/11/2026)**: 
   - Test set ~4,000 passages = **~222 prompt clusters** (không phải 4,000 prompt độc lập)
   - **Mỗi cluster KHÔNG LUÔN đủ 18 models** — phân phối model không đều, khác nhau giữa Public/Private LB (Topic #755733)
   - **Mutual Exclusivity Constraint**: Trong mỗi cluster, mỗi model xuất hiện **tối đa 1 lần** (bijection/injection)
   - **Hard Hungarian (cưỡng ép 18) sẽ FAIL** — phải dùng **Sparse/Injective Hungarian (g ≤ 6)**
9. **⭐ GLM vs Claude Confusion Root Cause (05/11/2026)**:
   - GLM-5.3 bị classify thành Claude 68/150 vs GLM chỉ 34/150 (Topic #755314)
   - Nguyên nhân: **GLM distills Claude** hoặc **GLM routes requests to Claude** (Anthropic claim)
   - **Giải pháp duy nhất**: Backbone **GLM-4 9B** (cùng họ Zhipu AI, shared token vocabulary & representation space)

---

## 6. 📌 Checklist kỹ thuật cho phiên làm việc tiếp theo (Next Session Action Items)
- [ ] **Implement Test-Internal Prompt Clustering**: Phát hiện cụm prompt trong `test.csv` hidden (khi submit) hoặc mô phỏng bằng embedding clustering / response similarity.
- [ ] **Implement Global Hungarian Matching**: Áp dụng Hungarian Algorithm trên từng cụm prompt với cost matrix từ ensemble probabilities — **Sparse/Injective (g ≤ 6)**.
- [ ] **Verify 18 Target Models**: Xác định chính xác 18 model xuất hiện trong test set (public vs private có thể khác).
- [ ] **Retire Sibling LoRA L24**: Khóa `w=0` vĩnh viễn (V79 đã chứng minh net-negative ở w=0.45 → 0.660).
- [ ] **Preserve Golden Anchor 0.662**: Giữ submission fallback an toàn, chỉ thêm Hungarian layer trên đỉnh.
- [ ] **Add GLM-4 9B Backbone**: Tích hợp `lucasboesen/name-that-llm-glm4-9b-trunc25` để giải quyết GLM vs Claude confusion.
- [ ] **Upgrade Qwen Weight**: Nâng trọng số Qwen 2.5 14B/32B trong ensemble (single model 0.670+).
- [ ] **Build Meta-Learner**: LightGBM/CatBoost trên OOF features cho dynamic routing.
