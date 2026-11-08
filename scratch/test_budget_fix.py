import sys
import json
import urllib.request

sys.path.append(r"C:\Users\s3cr3t\.gemini\antigravity\scratch\kaggle_claude_qa")
from claude_session import load_env, SYSTEM_PROMPT

with open('scratch/call_claude_strategic.py', 'r', encoding='utf-8') as f:
    text = f.read()

start = text.find('context_prompt = """') + len('context_prompt = """')
end = text.find('"""\n\nprint("Đang gửi')
context_prompt = text[start:end]

env = load_env()
api_key = env.get("MODEL_PROXY_API_KEY")
endpoint = "https://mp-staging.kaggle.net/models/openapi/chat/completions"

headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": context_prompt}
]

payload = {
    "model": "anthropic/claude-sonnet-5@default",
    "messages": messages,
    "max_tokens": 8192,
    "thinking": {
        "type": "enabled",
        "budget_tokens": 1024
    }
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

print("Sending with budget_tokens=1024 and max_tokens=8192...", flush=True)
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        msg = res["choices"][0]["message"]
        content = msg.get("content")
        print("Success! Content length:", len(content) if content else "None")
        print("Finish reason:", res["choices"][0].get("finish_reason"))
        print("Usage:", res.get("usage"))
        with open('scratch/claude_strategic_answer.txt', 'w', encoding='utf-8') as fo:
            fo.write(content)
        print("Saved to scratch/claude_strategic_answer.txt!")
except Exception as e:
    import traceback
    traceback.print_exc()

