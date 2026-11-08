import sys
import json
import urllib.request

sys.path.append(r"C:\Users\s3cr3t\.gemini\antigravity\scratch\kaggle_claude_qa")
from claude_session import load_env, SYSTEM_PROMPT

with open('scratch/ask_claude_v68.py', 'r', encoding='utf-8') as f:
    text = f.read()

start = text.find('v68_prompt = """') + len('v68_prompt = """')
end = text.find('"""\n\nenv = load_env()')
v68_prompt = text[start:end]

env = load_env()
api_key = env.get("MODEL_PROXY_API_KEY")
endpoint = "https://mp-staging.kaggle.net/models/openapi/chat/completions"

headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": v68_prompt}
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

print("Đang gửi yêu cầu với max_tokens=8192...", flush=True)
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        msg = res["choices"][0]["message"]
        reply = msg.get("content")
        print("Success! Reply length:", len(reply))
        with open('scratch/claude_v68_feedback.txt', 'w', encoding='utf-8') as fo:
            fo.write(reply)
        print("Wrote scratch/claude_v68_feedback.txt!")
except Exception as e:
    import traceback
    traceback.print_exc()

