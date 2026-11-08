import json

transcript_path = r'C:\Users\s3cr3t\.gemini\antigravity-ide\brain\f5ca0093-20f3-4308-a03b-7ccba8e6503a\.system_generated\logs\transcript.jsonl'

with open(transcript_path, 'r', encoding='utf-8') as f:
    lines = [json.loads(line) for line in f]

user_turn_indices = []
for idx, item in enumerate(lines):
    if item.get('type') == 'USER_INPUT':
        user_turn_indices.append(idx)

print(f"Total user turns detected: {len(user_turn_indices)}")

# Let's inspect turns 29 to 40 (0-indexed 28 to 39)
for turn_num in range(29, 41):
    turn_idx = turn_num - 1
    start_step = user_turn_indices[turn_idx]
    end_step = user_turn_indices[turn_idx + 1] if turn_idx + 1 < len(user_turn_indices) else len(lines)
    
    user_item = lines[start_step]
    user_content = user_item.get('content', '')
    created_at = user_item.get('created_at', '')
    
    # Collect tools called in this range
    tools_called = []
    assistant_responses = []
    for step in range(start_step + 1, end_step):
        st = lines[step]
        if st.get('type') == 'PLANNER_RESPONSE':
            for tc in st.get('tool_calls', []):
                tools_called.append(tc.get('name'))
            if st.get('content'):
                assistant_responses.append(st.get('content'))
                
    print(f"\n==================== TURN {turn_num} ====================")
    print(f"Time: {created_at}")
    print(f"User Request:\n{user_content[:200]}...")
    print(f"Tools Used: {list(set(tools_called))}")
    if assistant_responses:
        print(f"Summary of Response:\n{assistant_responses[-1][:250]}...")
