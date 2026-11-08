# Name That LLM - Model Backbones & Benchmark Analysis (MODEL.md)

## 1. Tổng quan tình hình sử dụng Model trên Kaggle

Trang Model Benchmarks của cuộc thi phản ánh rõ ràng các kiến trúc backbone đang được cộng đồng tin dùng nhất, số lượng người sử dụng, và điểm số cao nhất (Best Public Score) đạt được với từng model.

> [!IMPORTANT]
> **Dòng Qwen đang thống trị tuyệt đối bảng xếp hạng:**
> Các model thuộc họ **Qwen 3 (8b-base, 4b-base)** và **Qwen 2.5 (7b)** đang nắm giữ điểm số cao nhất hiện tại là **`0.547`**.
> 
> **⭐ CẬP NHẬT 05/11/2026 — PHÁT HIỆN TỪ TOP 10, FORENSIC ANALYSIS & REAL-TIME KAGGLE DISCUSSIONS:**
> - **GLM-4 9B** (`lucasboesen/name-that-llm-glm4-9b-trunc25`) là **BẮT BUỘC** để tách `glm-5.3` khỏi Claude (dấu câu giống hệt nhau) — **Topic #755314: GLM-5.3 classified as Claude 68 vs 34/150**
> - **Qwen 2.5 14B/32B** cần trọng số cao hơn (public notebooks underweight) — **Topic #754860: Single model Qwen3-14B đạt 0.670+**
> - **Gemma-2 9B/27B** bắt trọn phong cách Google Gemini routing
> - **Sibling LoRA L24** — **RETIRED (w=0 vĩnh viễn)** — V79 net-negative confirmed (0.660 < 0.662)
> - **Test set Sparse Distribution** — Không phải lúc nào cũng 18 models/prompt, mix khác Public/Private — **Topic #755733**

---

## 2. Bảng xếp hạng các Pretrained Models được sử dụng

| Tên Model | Nhà phát triển | Framework / Kiểu Model | Số người dùng (Users) | Best Public Score | Nhận xét & Đánh giá |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 3 (8b-base)** | QwenLM | Transformers (LLM Base) | **25 users** | **0.547** | Backbone phổ biến nhất và đạt SOTA hiện tại. Rất mạnh về văn phong và biểu diễn từ vựng. |
| **Qwen 3 (4b-base)** | QwenLM | Transformers (LLM Base) | **15 users** | **0.547** | Hiệu năng tương đương bản 8b nhưng nhẹ hơn, train và inference nhanh hơn đáng kể trong giới hạn GPU 9h. |
| **Qwen2.5 (7b)** | QwenLM | Transformers (LLM Base) | **12 users** | **0.547** | Bản tiền nhiệm rất ổn định, đạt cùng mức SOTA 0.547 khi kết hợp với LoRA. |
| **Qwen 3 (14b-base)** | QwenLM | Transformers (LLM Base) | 10 users | **0.531** | Model kích thước lớn hơn, nhưng có thể tốn tài nguyên và dễ overfit với tập train nhỏ (2.7k sample). |
| **Gemma 4 (gemma-4-12b)** | Google | Transformers (LLM) | 7 users | **0.531** | Đại diện mạnh từ Google, cung cấp góc nhìn biểu diễn khác biệt, rất tiềm năng cho Ensemble với Qwen. |
| **Qwen 3 Embedding (0.6b)** | QwenLM | Transformers (Embedding Model) | 6 users | **0.531** | Cực kỳ nhẹ và nhanh. Rất hiệu quả khi dùng trích xuất embedding để đưa vào classifier hoặc SVM/GBDT. |
| **Qwen 3 (1.7b-base)** | QwenLM | Transformers (LLM Base) | 6 users | **0.459** | Bản siêu gọn, điểm ở mức trung bình khá. |
| **Qwen 3 Embedding (4b)** | QwenLM | Transformers (Embedding Model) | 1 user | **0.486** | Model embedding lớn hơn của Qwen. |
| **EmbeddingGemma 2** | Google DeepMind | Multimodal Embedding (768-d) | 1 user | **0.376** | Vector không gian 768 chiều, hiệu quả chưa cao bằng Qwen Embedding. |
| **UltraBert_380m** | Community | Transformers (Encoder) | 1 user | **0.362** | Mô hình Encoder cổ điển. |
| **Qwen3.5 (qwen3.5-9b)** | ShelterW | Transformers | 1 user | **0.309** | Bản tùy biến cộng đồng. |
| **Qwen 3 (0.6b)** | QwenLM | Transformers (LLM Base) | 1 user | **0.309** | Kích thước quá nhỏ dẫn tới khả năng phân biệt văn phong bị hạn chế. |
| **ModernBERT (large)** | Answer.AI | Transformers (Encoder) | 1 user | **0.238** | Encoder hiện đại nhưng hiệu quả chưa cao đối với bài toán nhận diện LLM style. |

---

## 3. Phân tích chuyên sâu các kiến trúc

### 3.1. Nhóm Decoder LLM (Qwen 3 / Qwen 2.5 / Gemma 4)
- **Điểm số:** **0.531 - 0.547** (Public Benchmark) | **Single Model SOTA: 0.670+** (Topic #754860, 05/11)
- **Ưu điểm:**
  - Nhận biết ngữ pháp, cấu trúc cú pháp và thói quen sinh từ của LLM tốt hơn hẳn các mô hình encoder truyền thống.
  - Hỗ trợ tốt kỹ thuật **LoRA** (Low-Rank Adaptation) giúp huấn luyện cực kỳ ổn định trên ~2,700 mẫu dữ liệu.
- **Lựa chọn tối ưu:** **`Qwen 3 (4b-base)`** hoặc **`Qwen 3 (8b-base)`** là ứng viên hàng đầu cho giải pháp đơn lẻ (single model) hoặc backbone chính.
- **⭐ BREAKTHROUGH (05/11)**: **Qwen3-14B single model đạt 0.670** (Horikita Saku, no post-process, single fold) — **Backbone mạnh nhất open-source hiện nay**. Cần nâng trọng số ensemble từ 1.0 → **2.0+**.

### 3.2. Nhóm Dedicated Embedding Models (Qwen 3 Embedding 0.6b / 4b)
- **Điểm số:** **0.486 - 0.531**
- **Ưu điểm:**
  - `Qwen 3 Embedding 0.6b` đạt tới **0.531** dù chỉ có 600M tham số.
  - Tốc độ trích xuất vector cực nhanh, rất phù hợp làm feature extractor kết hợp với các thuật toán:
    - Logistic Regression / Ridge / SVM / GBDT (LightGBM, CatBoost).
    - MLP Classifier với dropout cao.
    - Kỹ thuật **Adaptive Centering** trên vector embedding.

### 3.3. ⭐ Nhóm GLM-4 9B — **CRITICAL BACKBONE (05/11/2026)**
- **Model**: `lucasboesen/name-that-llm-glm4-9b-trunc25` (GLM-4 9B truncated 25 layers)
- **Điểm số**: Chưa có public benchmark đơn lẻ, nhưng **bắt buộc cho ensemble**
- **Lý do quan trọng (Topic #755314)**:
  - GLM-5.3 bị classify thành **Claude 68/150 vs GLM chỉ 34/150**
  - Nguyên nhân: **GLM distills Claude** hoặc **GLM routes requests to Claude** (Anthropic claim)
  - **Chỉ model cùng họ Zhipu AI (GLM-4) mới có shared token vocabulary & representation space** để tách biệt GLM vs Claude
- **Vai trò trong ensemble**: Orthogonal backbone chuyên giải quyết **GLM vs Claude confusion** — không thể thay thế bằng Qwen/Llama/Gemma

### 3.3. Nhóm Encoder-only (ModernBERT, UltraBERT)
- **Điểm số:** **0.238 - 0.362** (Kém xa nhóm Decoder & Embedding)
- **Hạn chế:** Các encoder chỉ tập trung hiểu ngữ nghĩa (semantic understanding) chứ không nhạy với xác suất sinh từ (token generation probabilities) hay phong cách hành văn của các LLM thế hệ mới.

---

## 4. Chiến lược lựa chọn Model cho dự án

1. **Model Chính (Primary Backbone):**
   - Fine-tune **Qwen 3 (4b-base)** hoặc **Qwen 3 (8b-base)** bằng **LoRA**.
   - Thiết lập `max_seq_length = 384` hoặc `512` (đủ bao quát toàn bộ 40-250 từ của dataset).
2. **Model Phụ & Đa dạng hóa (Diversity for Ensembling):**
   - **Gemma 4 (12b)**: Kết hợp dự đoán giữa Qwen và Gemma sẽ giảm thiểu bias từ cùng một họ mô hình.
   - **Qwen 3 Embedding (0.6b)**: Trích xuất embedding + Stylometric features $\rightarrow$ Classifier nhẹ để ensemble xác suất.
3. **Chiến lược Quản lý Bộ nhớ & Giới hạn 9h:**
   - 4b/8b model kết hợp 4-bit / 8-bit quantization (QLoRA hoặc bitsandbytes) để inference mượt mà trong giới hạn thời gian chạy trên GPU Kaggle (T4/P100).

---

## 5. Kiến trúc Multi-Backbone Thực chiến của Đội — **CẬP NHẬT 05/11/2026 (Champion V76-V80 → V81+)**

### 5.1. Kiến trúc Golden Anchor (V77/V78/V80 — Score 0.662 🥉)

| Nhánh Backbone | Kiến trúc & Kích thước | Trọng số ($w$) | Trạng thái | Đóng góp Kỹ thuật |
| :--- | :--- | :---: | :--- | :--- |
| **Golden Anchor Ensemble** | Clean Open Pool 198k + Anchored OT + SVC | **Core** | ✅ **ACTIVE** | Baseline 0.662 vững chắc, không đổi |
| **Qwen 2.5 14B / 32B** | Decoder LLM (Orthogonal Backbone) | **TĂNG** | ⚡ **UPGRADE NEEDED** | Backbone mạnh nhất open-source, underweight trong public |
| **GLM-4 9B** | `lucasboesen/name-that-llm-glm4-9b-trunc25` | **THÊM MỚI** | 🔥 **CRITICAL** | Giải quyết GLM vs Claude confusion (dấu câu giống hệt) |
| **Gemma-2 9B / 27B** | Google Decoder LLM | **THÊM MỚI** | ⚡ **NEEDED** | Bắt trọn phong cách Gemini routing |
| **ATLAS Qwen3-8B Adapter** | LoRA Adapter (Joseph Adamski) | `10-20% blend` | ✅ **ACTIVE** | +0.002 LB ổn định |
| **DeBERTa-v3-large** | Token-attention head (L23 pooling) | `0.185` | ✅ **ACTIVE** | Từ vựng phân tầng, dấu câu tinh vi |
| **ModernBERT / UltraBERT** | Encoder Ensemble | `0.095` | ✅ **ACTIVE** | Ổn định hóa phương sai |
| **4-View LinearSVC** | N-grams Char/Word/Shape/Function | `1.516` | ✅ **ACTIVE** | Topic-independent stylometrics |
| **Sibling LoRA L24** | Contrastive LoRA (Layer 24) | **`0.0` (RETIRED)** | 🚫 **RETIRED** | V79 net-negative: 0.660 < 0.662 |

### 5.2. Kiến trúc Mục tiêu Champion V81+ (Target 0.668+ → 0.704+)

| Nhánh Backbone | Kiến trúc & Kích thước | Trọng số ($w$) | Ưu tiên | Ghi chú |
| :--- | :--- | :---: | :---: | :--- |
| **Golden Anchor (Preserved)** | Pool Matching + Anchored OT + SVC | Core | 🔥 **LOCKED** | Không thay đổi, bảo toàn 0.662 |
| **Test-Internal Hungarian Layer** | Prompt Clustering + Global Assignment | **NEW LAYER** | 🔥 **#1 PRIORITY** | +0.006 → +0.040 (Key breakthrough) |
| **GLM-4 9B** | Truncated 25 layers | High | 🔥 **#2 PRIORITY** | Orthogonal to Qwen/Llama |
| **Qwen 2.5 14B / 32B** | Full / LoRA | High | ⚡ **#3 PRIORITY** | Nâng trọng số từ 1.0 → 2.0+ |
| **Gemma-2 9B / 27B** | Full / LoRA | Medium | ⚡ **#4 PRIORITY** | Gemini family specialist |
| **Meta-Learner (LightGBM/CatBoost)** | Dynamic Routing on OOF features | New | ⚡ **#5 PRIORITY** | Non-linear stacking |
| **Zero-Shot Perplexity** | Cross-Entropy on Open Models | New | 💡 **RESEARCH** | Top 10 technique |

### 5.3. Quy tắc Ensemble Cập nhật (Post-V80)

1. **Golden Anchor = Immutable Core** — Không bao giờ thay đổi, luôn giữ submission fallback 0.662
2. **Hungarian Layer = Additive** — Chỉ áp dụng trên đỉnh Golden Anchor, không replace
3. **Sibling LoRA = Banned** — w=0 vĩnh viễn, không bao giờ tái kích hoạt
4. **Orthogonal Backbones Only** — Mỗi backbone mới phải giảm correlation lỗi với ensemble hiện tại
5. **Meta-Learner > Linear Blend** — Phi tuyến thay thế cộng trọng số cố định

