import json

transcript_path = r'C:\Users\s3cr3t\.gemini\antigravity-ide\brain\f5ca0093-20f3-4308-a03b-7ccba8e6503a\.system_generated\logs\transcript.jsonl'
output_path = r'c:\Users\s3cr3t\My Drive (baongdqu@gmail.com)\zzz kỹ năng tech - ai (separator)\( project ) kaggle competition name that label\CHAT_HISTORY.md'

with open(transcript_path, 'r', encoding='utf-8') as f:
    lines = [json.loads(line) for line in f]

user_turn_indices = []
for idx, item in enumerate(lines):
    if item.get('type') == 'USER_INPUT':
        user_turn_indices.append(idx)

# Let's inspect turns 29 to 40
turn_data = []
for turn_num in range(1, len(user_turn_indices) + 1):
    turn_idx = turn_num - 1
    start_step = user_turn_indices[turn_idx]
    end_step = user_turn_indices[turn_idx + 1] if turn_idx + 1 < len(user_turn_indices) else len(lines)
    
    user_item = lines[start_step]
    raw_user_content = user_item.get('content', '')
    
    # Extract only inside <USER_REQUEST> if present
    if '<USER_REQUEST>' in raw_user_content and '</USER_REQUEST>' in raw_user_content:
        u_req = raw_user_content.split('<USER_REQUEST>')[1].split('</USER_REQUEST>')[0].strip()
    else:
        u_req = raw_user_content.strip()
        
    created_at = user_item.get('created_at', '')
    
    tools_called = []
    assistant_text = ""
    for step in range(start_step + 1, end_step):
        st = lines[step]
        if st.get('type') == 'PLANNER_RESPONSE':
            for tc in st.get('tool_calls', []):
                tname = tc.get('name')
                if tname and tname not in tools_called:
                    tools_called.append(tname)
            if st.get('content'):
                assistant_text = st.get('content').strip()
                
    turn_data.append({
        'turn': turn_num,
        'created_at': created_at,
        'user_request': u_req,
        'tools': tools_called,
        'assistant_text': assistant_text
    })

print(f"Loaded {len(turn_data)} turns.")
