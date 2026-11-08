import urllib.request
import json
import os

ENV_PATH = r"C:\Users\s3cr3t\.gemini\antigravity\scratch\kaggle_claude_qa\.env"
env = {}
with open(ENV_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()

prompt = """
Tiếp tục bài phản biện từ câu bị ngắt:
"Cách top teams (Le Viet, Meni Gottesman at 0.72 - 0.75) reconstruct prompt groups reliably..."

Hãy hoàn thành trọn vẹn:
1. Tiếp tục Section III: Giải pháp phục hồi prompt group chuẩn xác nhất (Deduplication, Exact prefix/suffix, TF-IDF prompt content matching vs embedding, graph components, fallback logic).
2. Section IV: The Sibling Family Bottleneck (Tại sao Deep Learning thất bại trên Opus 4.6/4.8/5, Gemini 3.7/3.8, GPT Sol/Terra và Classical ML can thiệp thế nào).
3. Section V: The Decisive Strategic Blueprint for Champion V94 (Kế hoạch hành động cụ thể từng bước, không mơ hồ, để phá vỡ ngay lập tức 0.664 và nhảy vọt lên 0.70 - 0.73+).
"""

data = json.dumps({
    "model": "anthropic/claude-sonnet-5@default",
    "messages": [
        {"role": "user", "content": "Tiếp tục bài phản biện từ phần: Cách top teams reconstruct prompt groups và các phần IV, V."},
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 8192
}).encode("utf-8")

req = urllib.request.Request(
    "https://mp-staging.kaggle.net/models/openapi/chat/completions",
    headers={"Authorization": "Bearer " + env["MODEL_PROXY_API_KEY"], "Content-Type": "application/json"},
    data=data,
    method="POST"
)

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        msg = res["choices"][0].get("message")
        if msg and msg.get("content"):
            content = msg["content"]
            with open(r"C:\Users\s3cr3t\.gemini\antigravity\scratch\claude_full_critique_p2.txt", "w", encoding="utf-8") as f:
                f.write(content)
            print("SUCCESS! Wrote", len(content), "characters to scratch/claude_full_critique_p2.txt")
except Exception as e:
    import traceback
    traceback.print_exc()

