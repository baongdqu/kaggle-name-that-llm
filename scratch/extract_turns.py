import json

transcript_path = r'C:\Users\s3cr3t\.gemini\antigravity-ide\brain\f5ca0093-20f3-4308-a03b-7ccba8e6503a\.system_generated\logs\transcript.jsonl'
turns = []

with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data.get('type') == 'USER_INPUT':
            turns.append({
                'step_index': data.get('step_index'),
                'created_at': data.get('created_at'),
                'content': data.get('content')
            })

print(f"Total user turns: {len(turns)}")
for i, t in enumerate(turns):
    content_snippet = t['content'][:100].replace('\n', ' ') if t['content'] else ''
    print(f"Turn {i+1} (step {t['step_index']} at {t['created_at']}): {content_snippet}")
