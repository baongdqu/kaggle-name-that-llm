import sys
sys.path.append('C:/Users/s3cr3t/.gemini/antigravity/scratch/kaggle_claude_qa')
from claude_session import chat_with_claude

prompt = "tiếp tục từ: 'hai chỉ số k...' và hoàn thiện trọn vẹn phần 3 (Giao thức nhổ neo an toàn) và các bước hành động cụ thể."
res = chat_with_claude(prompt, thinking_budget=1024)
print(res['reply'])

