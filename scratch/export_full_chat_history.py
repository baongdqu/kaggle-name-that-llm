import json, os, datetime

transcript_path = r'C:\Users\s3cr3t\.gemini\antigravity\brain\f5ca0093-20f3-4308-a03b-7ccba8e6503a\.system_generated\logs\transcript.jsonl'
output_path = r'c:\Users\s3cr3t\My Drive (baongdqu@gmail.com)\zzz kỹ năng tech - ai (separator)\( project ) kaggle competition name that label\abc CHAT_HISTORY.md'

messages = []
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        
        step_type = item.get('type')
        source = item.get('source')
        created_at = item.get('created_at', '')
        
        # User prompt
        if step_type == 'USER_INPUT' or source == 'USER_EXPLICIT':
            content = item.get('content', '')
            if content:
                messages.append({
                    'role': 'USER',
                    'time': created_at,
                    'text': content
                })
        # Model response
        elif step_type == 'PLANNER_RESPONSE' or source == 'MODEL':
            content = item.get('content', '')
            if content and not content.startswith('Thought:') and not content.startswith('{'):
                # Extract clean text if any
                messages.append({
                    'role': 'ASSISTANT',
                    'time': created_at,
                    'text': content
                })

print(f"Extracted {len(messages)} messages.")

# Format to markdown
md_lines = [
    "# NHẬT KÝ CHI TIẾT LỊCH SỬ HỘI THOẠI & PHÁT TRIỂN (CHAT_HISTORY.md)",
    "",
    f"> **Dự án:** Kaggle Competition: Name That LLM",
    f"> **Thời gian trích xuất:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"> **Tổng số lượt tương tác được trích xuất:** {len(messages)} lượt",
    "",
    "---",
    ""
]

for i, m in enumerate(messages, 1):
    role_icon = "👤 **USER**" if m['role'] == 'USER' else "🤖 **ANTIGRAVITY & CLAUDE SONNET**"
    time_str = f" `[{m['time']}]`" if m['time'] else ""
    md_lines.append(f"### Lượt #{i} — {role_icon}{time_str}")
    md_lines.append("")
    md_lines.append(m['text'].strip())
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

with open(output_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(md_lines))

print(f"Successfully updated {output_path} with {len(messages)} dialog turns.")

