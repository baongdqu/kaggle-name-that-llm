import sys, os
sys.path.append('C:/Users/s3cr3t/.gemini/antigravity/scratch/kaggle_claude_qa')
from claude_session import chat_with_claude

prompt = """Tiếp tục phần 2 (Lựa chọn Backbone tối ưu nhất) và phần 3 (Chiến lược huấn luyện 2 giai đoạn) của Phase 2 Training Blueprint."""

reply = chat_with_claude(prompt, thinking_budget=1024)
print("=== CLAUDE'S PHASE 2 PART 2 & 3 ===")
print(reply)

