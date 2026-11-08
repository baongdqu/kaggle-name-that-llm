import sys
import json
import urllib.request

sys.path.append(r"C:\Users\s3cr3t\.gemini\antigravity\scratch\kaggle_claude_qa")
from claude_session import load_env, SYSTEM_PROMPT

with open('scratch/claude_strategic_answer.txt', 'r', encoding='utf-8') as f:
    part1 = f.read()

prompt_followup = f"""[TIẾP TỤC PHÂN TÍCH]
Claude ơi, bạn vừa phân tích rất xuất sắc ở phần 1 và 2, cũng như P0 (Pairwise Specialist Cascade). Tuy nhiên ở mục P1 (Mở rộng ATLAS log-likelihood thành vector đầy đủ) đoạn cuối bị ngắt do chạm giới hạn token:

"{part1[-300:]}"

Bạn hãy viết tiếp từ mục P1 trở đi:
- Hoàn thiện ý tưởng P1 (Vector Perplexity / Likelihood đa mô hình mở).
- Các mục tiếp theo (P2, P3...) trong lộ trình bứt phá 0.700+.
- Thứ tự ưu tiên hành động cụ thể cho đội (Actionable Roadmap) trong các ngày tới.
Hãy viết súc tích, đi thẳng vào các bước kỹ thuật cụ thể!
"""

env = load_env()
api_key = env.get("MODEL_PROXY_API_KEY")
endpoint = "https://mp-staging.kaggle.net/models/openapi/chat/completions"

headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": prompt_followup}
]

payload = {
    "model": "anthropic/claude-sonnet-5@default",
    "messages": messages,
    "max_tokens": 4096,
    "thinking": {
        "type": "enabled",
        "budget_tokens": 1024
    }
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        msg = res["choices"][0]["message"]
        part2 = msg.get("content")
        with open('scratch/claude_strategic_answer_part2.txt', 'w', encoding='utf-8') as fo:
            fo.write(part2)
        print("Part 2 received and saved successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()

