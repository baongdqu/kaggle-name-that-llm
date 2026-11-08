import urllib.request, json, os

with open(r"C:\Users\s3cr3t\.gemini\antigravity\scratch\kaggle_claude_qa\.env") as f:
    env = dict(l.strip().split("=", 1) for l in f if "=" in l and not l.startswith("#"))

headers = {
    "Authorization": f"Bearer {env['MODEL_PROXY_API_KEY']}",
    "Content-Type": "application/json"
}

payload = {
    "model": "anthropic/claude-sonnet-5@default",
    "messages": [
        {"role": "user", "content": "Explain briefly in 2 sentences how Hungarian algorithm works."}
    ],
    "max_tokens": 1000
}

req = urllib.request.Request("https://mp-staging.kaggle.net/models/openapi/chat/completions",
                             data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode("utf-8"))
    choice = res.get("choices", [{}])[0]
    msg = choice.get("message") or {}
    print("Choice keys:", choice.keys())
    print("Finish reason:", choice.get("finish_reason"))
    print("Message keys:", msg.keys())
    print("Raw message:", repr(msg)[:500])

