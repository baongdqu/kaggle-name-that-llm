import json

log_path = r'C:\Users\s3cr3t\.gemini\antigravity-ide\brain\f5ca0093-20f3-4308-a03b-7ccba8e6503a\.system_generated\logs\transcript.jsonl'
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        idx = data.get('step_index')
        if idx in [8234, 8242]:
            print(f"=== Step {idx} ({data.get('type')}) ===")
            print(str(data.get('content'))[:1000])
