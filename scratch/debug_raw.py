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
    "max_tokens": 4096,
    "thinking": {"type": "disabled"}
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        with open('scratch/raw_res.json', 'w', encoding='utf-8') as fo:
            json.dump(res, fo, ensure_ascii=False, indent=2)
        print("Wrote raw_res.json! Choices length:", len(res.get("choices", [])))
        if res.get("choices"):
            print("Choice 0:", res["choices"][0])
except Exception as e:
    import traceback
    traceback.print_exc()

