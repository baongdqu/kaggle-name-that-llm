import sys, os
sys.path.append('C:/Users/s3cr3t/.gemini/antigravity/scratch/kaggle_claude_qa')
from claude_session import chat_with_claude

prompt = """Chào Claude Sonnet!
User vừa đưa ra một yêu cầu chiến lược rất sâu sắc:
"okie, các bạn hãy tìm hiểu lại đề bài, xem lại các lời giải ta đã làm và bảng điểm các lời giải cũng như leaderboard hiện có, bạn có thể có nhận xét gì ko?"

Bối cảnh hiện tại:
1. Đề bài Kaggle "Name That LLM":
   - 18 class LLM (Claude, Gemini, GPT, GLM, Qwen, Kimi, v.v.).
   - Metric: Macro-Averaged Recall (Balanced Accuracy).
   - Bản chất dữ liệu: Mỗi prompt sinh ra một nhóm văn bản, mỗi model trả lời mỗi prompt đúng 1 lần (Latin-Square bipartite matching).
   - Test set ẩn: ~4,000 mẫu (Public test ~18%-40%).
2. Kết quả của chúng ta:
   - Đạt 0.662 từ V59, và tiếp tục lặp lại 0.662 ở V60, V62, V63, V68, V69.
   - Hiện xếp hạng ~Top 7.5% (Rank ~312/4150).
3. Leaderboard hiện tại của top đầu thế giới:
   - Rank 1: Hod Carriers (0.747)
   - Rank 2: Lian Matsuo (0.742)
   - Rank 3: Le Viet (0.731)
   - Top 10 dao động từ 0.707 đến 0.724.
   - Top 20 đạt 0.693 - 0.705.
   - Khoảng cách giữa chúng ta (0.662) và Top 1 (0.747) là ~0.085 điểm.

Hãy đóng vai trò Senior Research Scientist & Principal Strategic Advisor, phân tích đa chiều:
1. Nhìn lại toàn bộ hành trình lời giải của chúng ta từ V1 đến V70: Chúng ta đã làm rất tốt điều gì? Tại sao chúng ta vươn lên và neo vững ở 0.662 mà không bị rớt?
2. Nhận xét về Leaderboard và khoảng cách 0.662 vs 0.747: Các đội đạt 0.72 - 0.74 đang làm gì mà chúng ta chưa làm? (Về mô hình nền tảng, token likelihood, out-of-domain data, pre-training, contrastive learning, hay prompt clustering?)
3. Bản chất mâu thuẫn giữa "tối ưu cục bộ hậu xử lý (post-processing)" vs "sức mạnh biểu diễn của mô hình nền tảng (representation learning)". Liệu chúng ta đã chạm trần lý thuyết của ensemble hiện tại chưa?
4. Đề xuất chiến lược mang tính cách mạng (paradigm shift) cho các bước đi tiếp theo.
"""

reply = chat_with_claude(prompt, thinking_budget=2048)
print("=== STRATEGIC ADVICE FROM CLAUDE ===")
print(reply)

