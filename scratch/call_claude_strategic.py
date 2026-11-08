# -*- coding: utf-8 -*-
import sys
import json
import os

sys.path.append(r"C:\Users\s3cr3t\.gemini\antigravity\scratch\kaggle_claude_qa")
from claude_session import chat_with_claude

context_prompt = """[BỐI CẢNH DỰ ÁN KAGGLE: "NAME THAT LLM"]
Chào Claude! Tôi là thành viên trong đội của bạn (Bảo Nguyễn & Antigravity/Gemini). Tôi muốn xin ý kiến chuyên môn sâu từ bạn về toàn bộ bài toán, đánh giá các hướng đi chúng tôi đã làm từ trước đến nay, và chiến lược cụ thể để bứt phá lên mốc 0.700+ (hoặc tiệm cận trần Bayes).

1. TỔNG QUAN BÀI TOÁN:
- Cuộc thi: Kaggle "Name That LLM" (Phân loại văn bản được sinh ra bởi 1 trong 18 frontier LLMs cân bằng: GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet, Claude 3 Haiku, Gemini 1.5 Pro, Gemini 1.5 Flash, Llama 3.1 8B, Llama 3.1 70B, Qwen 2.5 72B, Qwen 2.5 7B, Mistral Large 2, Gemma 2 9B, Gemma 2 27B, DeepSeek Coder V2, DeepSeek V2.5, GLM-4 9B, Command R+, Nemotron 70B).
- Metric: Balanced Accuracy (Macro-Recall).
- Cấu trúc đề thi ẩn: 2,700 dòng được sinh ra từ 150 prompt ẩn (mỗi prompt được gửi tới đúng 18 model, mỗi model trả lời đúng 1 lần -> Cấu trúc bàn cờ Latin-Square).
- Public LB chiếm 40% (~1,080 dòng / ~60 prompt). Private LB chiếm 60% (~1,620 dòng / ~90 prompt).
- Điểm số hiện tại của đội: 0.662 🥉👑 (Rank 312 / 4,150 đội toàn cầu - Top 7.5% thế giới, chỉ cách Top 100/Huy chương Bạc đúng 0.001 - 0.003 điểm).

2. CÁC HƯỚNG ĐI CHÚNG TÔI ĐÃ LÀM TỪ TRƯỚC TỚI NAY (V1 ĐẾN V67):
- Giai đoạn 1 (0.606 -> 0.646): Xây dựng Ensemble đa tầng gồm Causal Decoders (Layer 24 LoRA adapters trên Qwen3-14B, Qwen3-8B, Nishant 14B, Qwen2.5-14B) kết hợp Masked Encoders sạch (ModernBERT-v5, DeBERTa-v3-large) và 4-view TF-IDF SVC.
- Giai đoạn 2 (0.646 -> 0.653 - V39): Tăng cường sức mạnh trực giao của Meta Llama-3.1-8B (w=2.4, dp=True) nhờ hệ từ vựng 128k tiktoken và pretraining bias độc lập hoàn toàn với họ Qwen.
- Giai đoạn 3 (0.653 -> 0.659 - V50): Bổ sung ATLAS Generative Log-Likelihood (dùng Qwen 3 8B Base tính log-likelihood theo prompt cho các ca lưỡng lự top-2 margin < 0.30) + 37-Marker Typography Guards (em-dash, curly quotes, dấu chấm câu đặc thù).
- Giai đoạn 4 (0.659 -> 0.662 - V59/V62/V63 KỶ LỤC HIỆN TẠI):
  + Prompt-Aware Relative Length Calibration (Delta z_len * beta_m): Chuẩn hóa độ dài tương đối theo cụm prompt vì cùng 1 đề bài, Claude Sonnet/Qwen 72B luôn viết dài, còn Haiku/GPT-4o-mini luôn viết ngắn.
  + Damped Sinkhorn 0.45 (mix 0.35): Cân bằng phân phối biên về prior đồng đều của đề thi.
  + Latin-Square Anchor Kuhn-Munkres (ngưỡng size >= 5): Khóa cứng các dự đoán mỏ neo có độ tin cậy cao (margin >= 0.20, prob >= 0.35) và chạy Hungarian song ánh cho các mẫu còn lại để triệt tiêu va chạm trùng lặp model trong cùng 1 prompt.
- Bài học đắt giá về Phân phối (Negative Results - "Bí ẩn 0.654"): Thử nghiệm fine-tune ModernBERT trên 31.7k mẫu văn bản synthetic tạo ngoài (proxy generations) trên 4 phiên bản độc lập (V55, V56, V57, V64) đều bị sụt từ 0.662 về 0.654 do Domain Shift / Distribution Shift với đề thi ẩn của Kaggle.

3. CÂU HỎI CHO CLAUDE:
Claude sonnet ơi, bạn đánh giá thế nào về bài toán này, và các hướng đi của tôi đã làm từ trước có đúng hướng chưa, và theo bạn thì chúng tôi phải cần làm gì để đạt 0.700+?
Hãy phân tích đa chiều:
1. Đánh giá bản chất bài toán & trần lý thuyết Bayes (Bayes Error Rate) của cuộc thi này.
2. Nhận xét khách quan về lộ trình kỹ thuật mà đội đã đi qua (những gì làm rất đúng, những gì là ngõ cụt).
3. Đề xuất chiến lược kỹ thuật cụ thể, khả thi nhất để phá vỡ bức tường 0.662 và hướng tới mốc 0.670 - 0.700+.
"""

print("Đang gửi yêu cầu phân tích chiến lược tới Claude Sonnet (Thinking: 2048 tokens)...", flush=True)
res = chat_with_claude(context_prompt, thinking_budget=2048)

print("\n" + "="*80)
print(f"PHẢN HỒI TỪ CLAUDE SONNET ({res['model']}):")
print("="*80 + "\n")
print(res['reply'])
print("\n" + "-"*80)
print(f"Tokens: Total {res['total_tokens']} | Reasoning {res['reasoning_tokens']} | Latency: {res['latency_ms']}ms")
print("-"*80)

# Save result to a text file for persistence
with open('scratch/claude_strategic_response.txt', 'w', encoding='utf-8') as f:
    f.write(res['reply'])

