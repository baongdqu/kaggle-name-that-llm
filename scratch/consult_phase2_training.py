import sys, os
sys.path.append('C:/Users/s3cr3t/.gemini/antigravity/scratch/kaggle_claude_qa')
from claude_session import chat_with_claude

prompt = """Chào Claude Sonnet!
Champion V74 (TokAttn Orthogonal Apex) đã được submit lên Kaggle và đang chạy chấm điểm trong nền.

Theo đúng lộ trình "Bước Nhảy Vọt" (Leap Forward), bây giờ chúng ta kích hoạt Pha 2: Thiết kế kiến trúc huấn luyện Task-Specific Adapter (LoRA / Hard-Negative Contrastive Head) để tái hiện cú nhảy lịch sử kiểu V20 (+0.035).

Tôi (Gemini) đã kiểm toán mã nguồn huấn luyện LoRA thành công nhất trong lịch sử cuộc thi này:
1. RECIPE GỐC CỦA LORA24 (lucasboesen / vnton37nguyn):
- Backbone: Qwen2.5-7B hoặc Qwen3-8B-Base (truncate về 20-24 layers).
- LoRA: rank=16, alpha=32, dropout=0.05 trên 7 module chiếu: q, k, v, o, gate, up, down.
- Multi-layer Pooling: Layers [2, 4, 6, 8, 10, 12, 16, 20] -> mỗi layer qua một LayerNorm riêng biệt -> concatenate thành vector biểu diễn $Z$.
- Dual Readout Heads: 
  + `plain(Z)`: Head tuyến tính dự đoán nhãn trực tiếp.
  + `centered(Z - Z_mean)`: Head tuyến tính dự đoán nhãn sau khi đã trừ đi vector trung bình của các câu trả lời cùng prompt trong microbatch (3 same-prompt answers).
- Hàm Loss:
  $\mathcal{L} = \text{CrossEntropy}(\text{plain}(Z), y) + \text{CrossEntropy}(\text{centered}(Z - \bar{Z}), y)$

2. ĐIỂM NGHẼN CẦN NÂNG CẤP ĐỂ TẠO ĐỘT PHÁ LÊN 0.680 - 0.700+:
- Recipe cũ chọn ngẫu nhiên 3 mẫu cùng prompt bất kỳ.
- Trong khi đó, chẩn đoán lỗi của chúng ta đã xác định 1,538 mẫu lỗi cốt lõi (Core Stubborn Errors), mà 49.3% lỗi nằm ở các cặp anh em sinh đôi (Gemini Flash trio, GLM 5.3 duo, GPT-5.6 trio, Claude Opus).

Hãy tư vấn kiến trúc huấn luyện cụ thể cho Phase 2:
1. Nâng cấp cơ chế Batching & Loss: Chúng ta nên thiết kế `Hard-Negative Contrastive / Sibling-Aware Triplet Loss` như thế nào để ép mô hình phải kéo xa các cặp anh em ruột (ví dụ cùng prompt nhưng có cả gemini-3.6-flash vs gemini-3.7-flash, hoặc glm-5.3 vs glm-5.3-flash)?
2. Lựa chọn Backbone tối ưu nhất: Giữa Qwen3-8B-Base, Qwen2.5-14B-Base, Llama-3.1-8B, hoặc ModernBERT-Large, đâu là backbone đem lại tỷ lệ thành công cao nhất trong giới hạn 9 giờ GPU T4x2 / P100 trên Kaggle?
3. Thiết kế chiến lược huấn luyện 2 giai đoạn (Stage 1 trên tập pool ngoài, Stage 2 trên 2,699 mẫu thi đấu + Hard Negative Mining)?
"""

reply = chat_with_claude(prompt, thinking_budget=2048)
print("=== CLAUDE'S PHASE 2 TRAINING BLUEPRINT ===")
print(reply)

