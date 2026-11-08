import sys, os
sys.path.append('C:/Users/s3cr3t/.gemini/antigravity/scratch/kaggle_claude_qa')
from claude_session import chat_with_claude

prompt = """Chào Claude Sonnet!
TÌNH HUỐNG KHẨN CẤP (SOS):
Lượt nộp bài Champion V73 Relative Fingerprint Apex (Ref: 57965536) vừa được chấm xong:
Public Score: 0.657 (Sụt -0.005 so với mốc 0.662 của V68).
Trong khi đó, ở thực nghiệm local OOF (5-Fold GroupKFold trên 2,699 mẫu), V73 tăng từ 0.3608 lên 0.3708 (+0.0100 net gain, +27 mẫu sửa đúng thuần, OOF Win/Loss 132:105).

Cơ chế của V73:
1. Trích xuất 28-D LLM Stylistic Fingerprints (Markdown, TTR, nhịp điệu câu, độ dài từ, markers lập luận).
2. Tính Z-score tương đối theo từng cụm prompt (Intra-Prompt Relative Scaling):
   - Trên train: dùng prompt_id thật của train.
   - Trên test: dùng Gte (nhãn cụm dự đoán từ content-word TF-IDF average-linkage clustering).
3. Tiêm log-odds vào ma trận chi phí trước Kuhn-Munkres với w_fp = 0.120 cho 4 nhóm anh em: Gemini Flash, GLM 5.3, GPT 5.6, Claude Opus.

Tại sao trên Local OOF tăng mạnh (+0.0100) mà trên Public Test lại bị SỤT -0.005?
Hãy giải phẫu nguyên nhân kỹ thuật và toán học:
1. Có phải do trên Test, cụm prompt Gte chỉ là 'cụm dự đoán' (estimated cluster) với độ nhiễu ~5-10%, khiến cho việc tính Z-score tương đối nội bộ cụm bị sai lệch hoàn toàn so với khi dùng ground-truth prompt_id trên train? (Cluster Leakage / Mismatch).
2. Trọng số w_fp = 0.120 quá lớn khi đưa vào thực tế test set?
3. Việc can thiệp vào cả 4 nhóm anh em (đặc biệt là Claude Opus) có gây ra Collateral Damage dây chuyền khi các nhãn bị hoán vị sai trong Latin-Square?
4. Đánh giá và kế hoạch ứng cứu ngay lập tức.
"""

reply = chat_with_claude(prompt, thinking_budget=1536)
print("=== CLAUDE AUTOPSY ===")
print(reply)

