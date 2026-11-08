import json

log_path = r'C:\Users\s3cr3t\.gemini\antigravity-ide\brain\f5ca0093-20f3-4308-a03b-7ccba8e6503a\.system_generated\logs\transcript.jsonl'
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data.get('type') == 'USER_INPUT':
            print(f"Step {data.get('step_index')}: {data.get('content')}")
