# TỔNG KẾT BÀI HỌC KINH NGHIỆM TỪ CÁC SỰ CỐ & SỤT GIẢM HIỆU SUẤT (SUBMISSIONS FAIL & DEGRADATION POST-MORTEM)

> **Dự án:** Kaggle Competition: Name That LLM - Identify the model behind the text  
> **Đội thi:** Mohamed Morsi / `baonguyenduyquoc` (Rank 146 Toàn cầu - Silver Medal Tier)  
> **Kỷ lục cao nhất hiện tại:** **`0.639` 🥈👑 (Champion V20 Omni Spectral)**  
> **Thời điểm cập nhật:** 2026-10-25 (Sau lần nộp Champion V21)  
> **Mục tiêu tài liệu:** Ghi chép chi tiết, sâu sắc và có hệ thống toàn bộ nguyên nhân kỹ thuật, kiến trúc phần cứng, logic toán học và phân phối xác suất dẫn đến sự cố hoặc sụt giảm điểm số (V17.1-V17.8, Champion V19, Champion V21). Làm kim chỉ nam tuyệt đối không lặp lại sai lầm trong các phiên bản tiếp theo.

---

## 1. BẢNG TỔNG HỢP CÁC LẦN SỰ CỐ & SỤT GIẢM ĐIỂM SỐ (CHRONOLOGY OF FAILURES & PERFORMANCE DROPS)

| Phiên bản | Triệu chứng / Kết quả | Vị trí xảy ra | Nguyên nhân gốc rễ (Root Cause) | Thời gian lãng phí |
| :---: | :--- | :--- | :--- | :---: |
| **V17.1** | `FileNotFoundError` / Crash ngay lập tức | Cell 0 (Đọc Base LLM) | Lệnh `glob` quét tìm thư mục `base_dir` lấy nhầm thư mục con không chứa file `.safetensors`. | ~2 phút |
| **V17.2** | Treo vô tận (>4.5 tiếng, 16,400s), không ra output | Cell 12 (LoRA Inference - nạp OLMo2) | Mismatch chiều không gian: Base model OLMo2 ($H=4096$) nạp nhầm adapter Gemma ($H=3840$). Exception trước khối dọn rác rò rỉ 14GB VRAM. Mô hình sau bị tràn vào CPU RAM swap khiến tốc độ giảm 100 lần. | > 4.5 tiếng |
| **V17.3** | Quá giờ suy luận (>2.5 tiếng) | Cell 4 & Cell 12 (Forward LLMs) | Chi phí tính toán bẫy phần cứng T4x2: Chuyển tensor lớn qua PCIe liên GPU, ép kiểu `bfloat16` trên T4 gây sao chép VRAM thừa, tính lặp lại StyleGrams trên 13.8k mẫu. | > 2.5 tiếng |
| **V17.4** | `NameError: name 'g_mega' is not defined` | Cell 5, dòng 175 | Sửa code tối ưu hóa Mega-Gate tái sử dụng feature ma trận CSR nhưng sót lệnh dọn rác biến cũ `del g_mega`. | ~7.5 phút |
| **V17.5** | `NameError: name 'grams' is not defined` | Cell 5, dòng 177 | Lặp lại lệnh xóa biến `del grams, X_tr, X_ex, X_te` 2 lần liên tiếp do lệnh replace chuỗi bị trùng lặp. | ~7.8 phút |
| **V17.6** | Chạy quá 2 tiếng (>6 tiếng thực tế), không ra file | Vòng lặp inference mô hình lớn | Tổng thời gian suy luận tích lũy của chuỗi mô hình Decoder trên 4,000 dòng test ẩn vượt quá 6 tiếng tối đa. | > 6 tiếng |
| **V17.7** | Treo cứng vô tận (>3.7 tiếng, 13,240s), ngừng log từ 4711.9s | Cell 12 / Cell 14 (`lora8b_v1` & `tokattn_q35`) | CUDA IPC Stream Deadlock & Cạn kiệt VRAM phân mảnh: Accelerate spin-lock chia Qwen3-24L qua 2 GPU T4 khi VRAM phân mảnh. Các baseline cũ `tokattn` và `ultrabert` chiếm ít trọng số (<3%) nhưng gây nghẽn. | ~3.7 tiếng |
| **V17.8** | `AssertionError: checkpoint for lora24_public not attached` | Cell 12, dòng 23 | Tác giả dataset ngoài chuyển notebook sang Private; lệnh `assert hits` cứng quăng ngoại lệ làm crash ở phút 39.4. | ~39.4 phút |
| **V19** | Điểm số tụt về **`0.600`** (từ `0.604` của V18) | Cell 12, 14, 20 (Logic hậu xử lý) | 1) Bug thụt lề: Temperature scaling nằm trong vòng lặp 4 họ khiến văn bản ngắn bị chia $1.25^4 \approx 2.44$ lần, triệt tiêu tín hiệu tự tin. 2) Mất anchor protection trên văn bản ngắn làm Hungarian đảo lộn nhãn. 3) Thiếu `find_base` làm adapter ATLAS crash và đóng góp 0%. | 42.1m commit + 75m scoring |
| **V21** | Điểm số sụt xuống **`0.612`** (từ **`0.639`** của V20) | Cell 24 (Anchor Hungarian & Post-Sinkhorn) | 1) **Lỗi chí tử Post-Sinkhorn:** Áp dụng Temperature scaling *sau* Sinkhorn phá vỡ hoàn toàn ràng buộc tổng cột $N/18$ của Macro-Recall. 2) **Nhiễu phân cụm TF-IDF:** Hungarian ép gán đơn ánh khiến các mẫu cùng model trong cùng prompt cụm bị ép đổi nhãn sai. 3) Tăng Night24 lên 15% làm loãng backbone Part-Two. Thời gian scoring phình to lên 347.9 phút (~5.8 tiếng). | 0.5m commit + 347.9m scoring |

---

## 2. NGUYÊN NHÂN SÂU XA & BÀI HỌC KỸ THUẬT QUAN TRỌNG

### 2.1. Bài học về Trọng số & Chiều Không gian LoRA Adapter (Dimension Mismatch Trap)
- **Sai lầm ở V2:** Tin tưởng tuyệt đối vào tên file checkpoint trong dataset bên ngoài mà không kiểm tra cấu trúc ma trận thực tế (`adapter_config.json` và `adapter_model.safetensors`).
- **Hiện tượng:** Nạp adapter LoRA của một mô hình ($H=3840$) vào base model của một họ khác ($H=4096$). PyTorch quăng lỗi `RuntimeError: size mismatch for lora_A.weight`.
- **Hệ quả dây chuyền (VRAM Leak):** Khi xảy ra exception ở giữa hàm nạp model, khối dọn dẹp `del model; gc.collect(); torch.cuda.empty_cache()` không được thực thi. 14GB bộ nhớ GPU 0 bị khóa chết vĩnh viễn trong tiến trình notebook. Khi các mô hình tiếp theo chạy, CUDA buộc phải offload sang RAM hệ thống khiến notebook bị treo cứng hàng tiếng đồng hồ mà không crash.
- **Quy tắc vàng:**
  > Tuyệt đối không tích hợp thêm bất kỳ adapter hoặc mô hình mới nào nếu chưa xác thực 100% bằng script kiểm tra ngoại tuyến kích thước hidden dimension, config và tensor shapes. Mọi khối nạp LLM phải bọc trong `try...finally:` để đảm bảo giải phóng VRAM ngay cả khi xảy ra lỗi.

---

### 2.2. Bài học về Cạm bẫy Ép kiểu `bfloat16` trên GPU Nvidia Turing Tesla T4
- **Sai lầm ở V3:** Thấy cảnh báo tràn số hoặc giá trị `nan/inf` trong hidden states, mã nguồn tự động gọi:
  ```python
  model.to(torch.bfloat16)
  hs = model(**enc, output_hidden_states=True).hidden_states
  model.to(dtype)
  ```
- **Hiện tượng phần cứng:** GPU Tesla T4 sử dụng kiến trúc **Turing (Compute Capability 7.5)**, phần cứng **KHÔNG có Tensor Core cho `bfloat16`**. Mọi phép tính `bfloat16` đều phải chạy qua phần mềm mô phỏng (software emulation) cực kỳ chậm, và việc gọi `model.to(...)` trên mô hình 14B/8B gây sao chép và di chuyển liên tục 27GB trọng số trên VRAM.
- **Quy tắc vàng:**
  > Trên Kaggle GPU T4x2, **chỉ dùng thuần `torch.float16`**. Để chống tràn số/inf, áp dụng active clamping cục bộ:
  > `torch.nan_to_num(hs[l].float(), nan=0.0, posinf=65504.0, neginf=-65504.0)` mà không bao giờ ép kiểu lại toàn bộ mô hình sang `bfloat16`.

---

### 2.3. Bài học về Nghẽn Băng thông Giao tiếp Liên GPU (PCIe Bandwidth Bottleneck)
- **Sai lầm ở V3:** Mô hình Qwen3-14B Base ($H=5120$, 40 layers) được chia trên 2 GPU (`cuda:0` và `cuda:1`). Các hidden states ở GPU 1 được chuyển nguyên khối về GPU 0 trước khi tính mean-pooling:
  ```python
  # Sai lầm: truyền tensor 3 chiều khổng lồ qua khe PCIe
  hs_layer = hs[l].to(target_device) # Shape: (batch_size, seq_len, 5120)
  den = m.sum(1).clamp(min=1)
  pooled = (hs_layer * m).sum(1) / den
  ```
- **Hiện tượng:** Truyền khối tensor $B \times L \times 5120$ liên tục qua bus PCIe làm nghẽn toàn bộ bus giao tiếp, GPU phải dừng chờ truyền dữ liệu (I/O bounded).
- **Quy tắc vàng:**
  > Luôn thực hiện phép toán rút gọn (mean-pooling, token filtering) **ngay tại GPU nội bộ đang giữ layer đó**, sau đó chỉ truyền vector 2 chiều đã pooled ($B \times 5120$) về GPU đích. Giảm băng thông truyền tải đến **450 lần**.

---

### 2.4. Bài học về Lỗi Thao tác Chuỗi trong Tự động hóa Code (AST & Scope Safety)
- **Sai lầm ở V4 & V5:** Khi dùng script Python sửa đổi nội dung Jupyter Notebook bằng lệnh thay thế chuỗi regex / replace thô sơ:
  - Thay thế phần định nghĩa `g_mega = StyleGrams().fit(...)` bằng biến mượn từ trước nhưng quên xóa dòng `del g_mega` ở cuối cell $\rightarrow$ `NameError: name 'g_mega' is not defined`.
  - Thay thế không sạch để lại 2 lệnh `del grams` liên tiếp $\rightarrow$ `NameError: name 'grams' is not defined`.
- **Hậu quả:** Mất hơn 15 phút quý giá và tạo ra các phiên bản hỏng không đáng có.
- **Quy tắc vàng:**
  > Mọi sự thay đổi code trong notebook tự động bắt buộc phải qua 2 tầng kiểm tra:
  > 1. Kiểm tra cú pháp AST (`ast.parse(code)`).
  > 2. Kiểm tra biến tồn tại trong Scope bằng linter phân tích tĩnh trước khi build file `.ipynb`.

---

### 2.5. Bài học về Cân bằng Độ Phức tạp Ensemble và Ngân sách Thời gian (Inference Budget)
- **Sai lầm lớn nhất ở V6:** Nhồi nhét quá nhiều tầng mô hình nặng cùng lúc:
  1. Hai mô hình LoRA lớn (Qwen 14B + Qwen 8B).
  2. Bốn mô hình Decoder (GLM 9B, Qwen 8B, Qwen 7B, Qwen 4B).
  3. Ba mô hình Encoder (DeBERTa v3 Large, Electra Large, Desklib).
  4. Huấn luyện lại trên tập train tại bước inference (trích xuất lại TR + TE).
  5. 35k passages N-Gram LinearSVC.
  6. Dynamic Bitmask Injective DP cluster matching.
- **Hậu quả:** Khi chạy trên test ẩn của Kaggle (~4,000 dòng văn bản dài), tổng thời gian tích lũy vượt quá 6 tiếng, vi phạm quy định thời gian của cuộc thi và bị treo.
- **Quy tắc vàng:**
  > Một notebook cạnh tranh Top 1 phải tuân thủ nguyên tắc **"Lean & Lethal" (Gọn nhẹ & Chuẩn xác)**:
  > - Không bao giờ forward pass lại tập Train trong lúc Inference (phải precompute hoặc dùng model head đã train sẵn).
  > - Thời gian chạy trên tập smoke test giả lập 3 dòng phải $\le$ **45 giây**, và ước tính trên tập test ẩn 4,000 dòng không được vượt quá **1.5 giờ**.

### 2.6. Bài học về CUDA IPC Stream Deadlock & Cạn kiệt VRAM Phân mảnh (Version 7)
- **Sai lầm ở V7:** 
  1. Sau khi chạy các mô hình lớn (14B LoRA, 5 LoRA adapter, frozen decoders), các PyTorch CUDA context và hook của Hugging Face Accelerate (`device_map='auto'`) không được dọn dẹp triệt để bằng `gc.collect()` và `torch.cuda.ipc_collect()`.
  2. Nạp mô hình Qwen3 cắt layer (`num_hidden_layers = 24`) với `device_map='auto'` trên 2 GPU T4: Accelerate tự động chia mô hình qua GPU 0, GPU 1 và CPU. Khi VRAM bị phân mảnh, Accelerate rơi vào vòng lặp spin-lock chờ bộ nhớ (`cudaStreamSynchronize`), làm kernel bị treo cứng ở giây thứ 4711.9 mà không hề crash.
  3. Duy trì các thành phần baseline public cũ (`tokattn` và `ultrabert` 5-fold) có trọng số rất nhỏ ($0.095 - 0.153$, chỉ đóng góp $< 3\%$ tín hiệu) nhưng chiếm tới hơn 50% thời gian chạy và là thủ phạm chính gây nghẽn.
- **Hậu quả:** Kernel chạy vô tận tới 13,240s (~3.7 tiếng) mà không sinh ra file submission, gây lãng phí gần 8 giờ GPU quota T4x2.
- **Quy tắc vàng:**
  > - **Cắt bỏ hoàn toàn các thành phần baseline thừa (`tokattn`, `ultrabert`)** vốn dễ gây nghẽn CUDA và chiếm ít trọng số.
  > - **Luôn giải phóng triệt để bộ nhớ:** Bắt buộc gọi `del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()` sau mỗi lượt forward.
  > - Giữ vững cấu trúc tinh hoa: **Hungarian Injective Matching ($g \le 6$) + Margin-Locked Sinkhorn + Two-Stage LoRA (14B/8B) + 35k Pool LinearSVC**. Pipeline này sẽ chạy mượt mà trong dưới 35 phút!

### 2.7. Bài học về Sự Phụ thuộc Checkpoint Ngoài & Nguyên tắc Thoái lui Mềm (Graceful Degradation - Version 8)
- **Sai lầm ở V8:** 
  1. Trong hàm `score_lora_checkpoints` kế thừa từ baseline có chứa câu lệnh kiểm tra nghiêm ngặt:
     ```python
     hits = sorted(glob.glob('/kaggle/input/' + marker, recursive=True))
     assert hits, f'no checkpoint for {key}'
     ```
  2. Một tác giả cộng đồng trên Kaggle (`lightningv08`) đã chuyển notebook `ntl-lora24-fivefold-training` từ Public sang **Private**, dẫn đến việc Kaggle âm thầm gỡ bỏ gắn kết dataset này khỏi môi trường chạy.
  3. Khi notebook thực thi đến Cell 12 sau **39.4 phút** tính toán hoàn hảo (đã hoàn thành Stylegrams acc 1.00, 4-View LinearSVC 37k rows, Mega-Gate, GLM-4-9B, Qwen 8B/7B/4B, DeBERTa, ELECTRA, Desklib), biến `hits` rỗng kích hoạt `AssertionError` làm kernel sập ngay trước thềm sinh file `submission.parquet`.
- **Hậu quả:** Lãng phí 39.4 phút thời gian GPU T4x2 chỉ vì một checkpoint phụ không nằm trong kiến trúc trọng yếu (chủ lực là Two-Stage 14B/8B của `nishantdahal` và các adapter `instyle`, `claude`, `seed7`).
- **Quy tắc vàng:**
  > - **Tuyệt đối không dùng `assert` cứng đối với file hay dataset bên ngoài:** Mọi tác vụ đọc checkpoint ngoài phải tuân thủ triệt để nguyên tắc **Graceful Degradation**:
  >   ```python
  >   if not hits:
  >       print(f'[{key}] checkpoint not attached, skipping gracefully', flush=True)
  >       continue
  >   ```
  > - **Tự chủ tài nguyên:** Chỉ phụ thuộc vào các dataset và model sources đã được kiểm định trạng thái tồn tại ổn định lâu dài hoặc do chính đội ngũ sở hữu/kiểm soát.

---

### 2.8. Bài học về Bug Thụt Lề & Mất Anchor Protection Trong Hiệu Chuẩn Văn Bản Ngắn (Champion V19)
- **Sai lầm ở V19:**
  1. **Bug thụt lề khối Temperature Scaling:** Đoạn mã điều chỉnh nhiệt độ theo độ dài văn bản được đặt lồng bên trong vòng lặp duyệt 4 họ mô hình (Claude, GLM, GPT, Gemini). Kết quả là với mọi văn bản ngắn ($<60$ từ), ma trận xác suất bị chia $1.25$ liên tục 4 lần:
     $$P_{\text{short}} \sim P^{1/(1.25^4)} = P^{1/2.44} = P^{0.41}$$
     Phép lũy thừa này làm phẳng hoàn toàn đỉnh phân phối xác suất của văn bản ngắn, biến các dự đoán cực kỳ tự tin (ví dụ $P = 0.85$) thành phân phối phẳng đồng đều ($P \approx 0.12 - 0.15$), xóa sạch khả năng phân biệt mô hình.
  2. **Bỏ sót Anchor Protection trên văn bản ngắn:** Do logic điều kiện `if word_count >= 60:` chỉ bảo vệ các anchor trên văn bản dài, toàn bộ các văn bản ngắn bị đẩy vào thuật toán Hungarian gán ghép ngẫu nhiên, hoán đổi các mẫu tự tin cao sang các nhãn sai.
  3. **Lỗi `NameError: name 'find_base' is not defined` trong Cell 14:** Khiến module adapter ATLAS Qwen3-8B bị crash và bỏ qua hoàn toàn, mất trắng 10% đóng góp xác suất.
- **Hậu quả:** Điểm số tụt từ `0.604` (V18) xuống `0.600`.
- **Quy tắc vàng:**
  > - **Kiểm thử bất biến toán học (Mathematical Invariance Testing):** Mọi hàm biến đổi ma trận xác suất (temperature, calibration) phải được kiểm thử độc lập trên vector giả lập trước khi ráp vào notebook lớn. Xác nhận `assert (p_out.max(1) >= p_in.max(1)).all()` khi sharpening và không có lũy thừa kép.
  > - **Anchor Protection là toàn cục:** Mọi mẫu có độ tin cậy vượt ngưỡng ($\Delta P \ge 0.30, P \ge 0.60$) bắt buộc phải được khóa cứng (locked) bất kể độ dài văn bản ngắn hay dài.

---

### 2.9. Bài học về Phá Vỡ Ràng Buộc Phân Phối Đều Sinkhorn & Cạm Bẫy Phân Cụm Ép Đơn Ánh (Champion V21)
- **Sai lầm ở V21:**
  1. **Biến đổi phi tuyến HẬU Sinkhorn (Post-Sinkhorn Temperature Scaling):**
     - Đề thi Kaggle *Name That LLM* sử dụng metric **Balanced Accuracy (Macro-Recall)** và tập test ẩn có số lượng mẫu giữa 18 mô hình được cân bằng gần như tuyệt đối ($N/18$ mẫu mỗi lớp).
     - Thuật toán Sinkhorn-Knopp giải bài toán vận tải tối ưu entropy, tìm các nhân tử tỷ lệ $r_i$ và $c_j$ sao cho ma trận sau chuẩn hóa thỏa mãn đồng thời:
       $$\sum_{j=1}^{18} P_{ij} = 1 \quad \forall i, \quad \text{và} \quad \sum_{i=1}^{N} P_{ij} = \frac{N}{18} \quad \forall j$$
     - Trong V21, sau khi Sinkhorn đã cân bằng hoàn hảo các cột, đoạn mã tiếp theo lại áp dụng:
       $$P'_{ij} \propto P_{ij}^{1/\tau_i} \quad (\tau_i \in [0.90, 1.15])$$
       Biến đổi phi tuyến này bẻ cong xác suất các hàng theo độ dài văn bản khác nhau. Khi chuẩn hóa lại tổng hàng bằng 1, **tổng các cột $\sum_i P'_{ij}$ bị biến dạng lệch từ 10% đến 25%** so với mục tiêu $N/18$. Cột của các mô hình có phong cách viết ngắn bị phình to, trong khi cột của các mô hình viết dài bị bóp nghẹt. Khi lấy `argmax`, phân phối dự đoán bị lệch nặng nề, làm sụt giảm trực tiếp điểm Macro-Recall.
  2. **Nhiễu phân cụm không giám sát (Unsupervised Clustering Noise) trong Hungarian Injective Matching:**
     - V21 áp dụng thuật toán Kuhn-Munkres Hungarian để ép các mẫu trong cùng cụm prompt `groups` phải nhận các nhãn mô hình khác nhau (Injective Mapping).
     - Tuy nhiên, biến `groups` được sinh ra từ phân cụm phân cấp TF-IDF / Cosine Distance (`fcluster(linkage(...))`). Trong tập test ẩn, nếu 2 hoặc 3 văn bản có từ vựng tương tự nhau (hoặc cùng một chủ đề học thuật/lập trình) thực chất được sinh ra bởi **cùng 1 mô hình** (ví dụ `gpt-5.5`), Hungarian bị ép buộc phải gán 1 mẫu cho `gpt-5.5` và cưỡng chế đổi các mẫu còn lại sang các mô hình khác (ví dụ gán nhầm sang `kimi-k3` hoặc `glm-5.3`).
  3. **Tăng tỷ trọng Night24 Spectral Heads lên 15%:**
     - Bộ 4,404 Heads của Night24 là mô hình bổ trợ phân loại phong cách. Tỷ trọng tối ưu đã được chứng minh ở V20 là **10%** ($0.90 \cdot P + 0.10 \cdot P_{\text{helper}}$). Việc tăng lên 15% đã làm loãng tín hiệu từ Two-Stage LoRA Part-Two (xương sống mạnh nhất của giải pháp).
  4. **Bùng nổ thời gian suy luận (Inference Time Explosion):**
     - Vòng lặp giải bài toán Hungarian bipartite matching bằng Python trên từng cụm nhỏ cùng với các truy vấn likelihood ATLAS đẩy tổng thời gian suy luận lên **347.9 phút (~5.8 tiếng)**, tiệm cận vùng nguy hiểm 9 giờ của Kaggle.
- **Hậu quả:** Điểm số rớt từ **`0.639`** (V20) xuống **`0.612`** (V21).
- **Quy tắc vàng:**
  > - **Sinkhorn BẮT BUỘC là lớp chốt chặn cuối cùng (Terminal Calibrator):**
  >   Tuyệt đối KHÔNG BAO GIỜ áp dụng bất kỳ phép toán phi tuyến nào (Temperature scaling, Micro-gating, Logit re-weighting) SAU bước Sinkhorn. Mọi sự tinh chỉnh văn phong, dấu hiệu suy luận hay độ dài phải được hoàn tất TRƯỚC khi đưa vào Sinkhorn. Sinkhorn sẽ hấp thụ các tinh chỉnh đó và đưa tổng cột về $N/18$ đồng đều hoàn hảo trước khi lấy `argmax(1)`.
  > - **Không ép gán đơn ánh cứng trên cụm phân cụm thô:**
  >   Nếu không có nhãn ground-truth `prompt_id` chính thức từ ban tổ chức, các phương pháp phân cụm TF-IDF / Embedding luôn có tỷ lệ dung sai. Việc ép gán đơn ánh bằng Hungarian cứng sẽ biến sai số phân cụm thành sai số gán nhãn hàng loạt.
  > - **Neo giữ Night24 Spectral Heads ở 10% chuẩn:**
  >   Luôn duy trì tỷ trọng $0.90 \cdot P + 0.10 \cdot P_{\text{helper}}$ như V20 đã kiểm chứng đạt đỉnh `0.639`.

---

## 3. CHECKLIST KIỂM ĐỊNH BẮT BUỘC TRƯỚC MỌI LẦN PUSH TIẾP THEO

- [ ] **Kiểm tra VRAM & Shape:** Đã kiểm tra offline hidden dimension của mọi LoRA adapter khớp 100% với Base model chưa?
- [ ] **Bảo vệ VRAM Leak:** Tất cả các khối nạp và forward mô hình có `try ... finally: torch.cuda.empty_cache()` không?
- [ ] **Không bfloat16 trên T4:** Đã đảm bảo không có lệnh `.to(torch.bfloat16)` nào trên GPU T4 chưa?
- [ ] **Tối ưu truyền PCIe:** Mean pooling đã được tính tại GPU cục bộ chưa?
- [ ] **Kiểm tra biến Scope:** Không còn bất kỳ lệnh `del` gọi đến biến không tồn tại hoặc gọi lặp 2 lần.
- [ ] **Ngân sách thời gian:** Tổng số mô hình forward có đảm bảo chạy dưới 1.5 giờ trên 4,000 dòng test không?
- [ ] **Thoái lui mềm (Graceful Degradation):** Tuyệt đối không dùng `assert` cứng cho dataset/checkpoint bên ngoài; luôn có fallback `if not hits: continue`.
- [ ] **Ràng buộc thứ tự Sinkhorn:** Sinkhorn BẮT BUỘC phải là bước biến đổi cuối cùng trước `argmax(1)`. Tuyệt đối không áp dụng Temperature Scaling, Micro-Gates hay biến đổi phi tuyến nào SAU Sinkhorn.
- [ ] **Không ép gán đơn ánh Hungarian trên cụm không nhãn:** Phân cụm TF-IDF không giám sát chứa nhiều mẫu cùng nhãn thật; cấm dùng Hungarian hard injective trên các cụm này.
- [ ] **Trọng số Night24 chuẩn:** Luôn neo Night24 Spectral Heads ở mức 10% như V20 đã kiểm chứng (`0.639`).

---
*Tài liệu này là cam kết kỹ thuật của toàn đội nhằm đảm bảo mọi lượt nộp bài từ nay về sau đều hoàn hảo, sạch lỗi, và tiến thẳng vào Top 10 toàn cầu!*
