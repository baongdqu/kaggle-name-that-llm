import sys, os
sys.path.append('C:/Users/s3cr3t/.gemini/antigravity/scratch/kaggle_claude_qa')
from claude_session import chat_with_claude

prompt = """Chào Claude Sonnet!
Tôi (Gemini) vừa hoàn thành kiểm toán kỹ thuật toàn diện trong workspace và Kaggle API để chuẩn bị cho Pha 1 của chiến dịch Leap Forward (vượt mốc 0.662).

Các phát hiện thực nghiệm chấn động:
1. TRONG WORKSPACE / KERNEL SOURCES CỦA V68:
- Chúng ta có `lucasboesen/name-that-llm-train-tokattn2` (và dataset mirror `swaroopmeher/ntl-public-heads`).
- Nó chứa 2 model learned token-attention hoàn chỉnh:
  + `tokattn_debl23_final.pt` (DeBERTa-v3-Large layer 23, OOF bacc = 0.4108, độc lập phân biệt cực tốt claude-opus-5 0.727, gpt-6-astra 0.613, gemini-3.1 0.647).
  + `tokattn_q8l22_final.pt` (Qwen3-8B-Base layer 22, OOF bacc = 0.3701).
  + Khi blend 2 model này (Deb + Q8) OOF đạt 0.4397.
- Đây là 'learned token-attention head' (tương tự công thức kirilltym 0.525), một biểu diễn khác hoàn toàn mean-pooling thông thường.

2. VỀ MODEL GLM-4-9B:
- Có Kaggle Model: `swaroopmeher/glm-4-9b-trunc25-ntl/Transformers/default/1` (11.4 GB, weights của 25 layers đầu GLM-4-9B).
- Kernel source `lucasboesen/name-that-llm-glm4-9b-trunc25` cũng có sẵn.

3. VỀ ELECTRA-LARGE:
- Có kernel source `lucasboesen/name-that-llm-electra-large-weights` (weights discriminator của google/electra-large).
- Có `lucasboesen/name-that-llm-desklib-weights` (DeBERTa-v3-large fine-tuned AI-text detector).

4. CƠ CHẾ KẾT NỐI VÀO V68:
- V68 hiện có cấu trúc MODELS linh hoạt:
  + l24: LoRA adapter (q14v5, q8v5, ll8v5, n14b, q25_14).
  + enc: Fully fine-tuned encoder (deb, mb_v5).
- Nếu thêm model mới vào MODELS:
  + Phương án A: Thêm TokAttn (DeBERTa L23 + Qwen L22) như một member mới hoặc specialist.
  + Phương án B: Thêm Desklib AI-Text Detector như một encoder member (type: 'enc').
  + Phương án C: Kích hoạt GLM-4 probe cho cặp GLM 5.3 duo.

Hãy cho ý kiến cố vấn kỹ thuật:
Giữa việc tích hợp TokAttn (learned token attention trên DeBERTa/Qwen) và Desklib AI-Text Detector (hoặc GLM-4 probe), đâu là phương án có tỷ lệ kỳ vọng cao nhất, ít rủi ro runtime nhất trên 2xT4 cho Champion V74?
"""

reply = chat_with_claude(prompt, thinking_budget=1024)
print("=== CLAUDE'S VERDICT ON V74 CANDIDATES ===")
print(reply)

