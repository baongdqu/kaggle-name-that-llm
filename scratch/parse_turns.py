import re

with open('abc CHAT_HISTORY.md', 'r', encoding='utf-8') as f:
    text = f.read()

turns = re.split(r'### Lượt \d+', text)
print(f"Total turns parsed: {len(turns)}")

# Search each turn for submission info, times, etc.
for i in range(1, len(turns)):
    turn_text = turns[i]
    # Check if there is submission or scoring info
    if any(k in turn_text.lower() for k in ['submission', 'submit', 'ref', 'kỷ lục', 'đạt', 'chấm điểm', 'complete', 'thời gian']):
        # extract title
        first_line = turn_text.strip().split('\n')[0]
        # extract time
        time_m = re.findall(r'\*\*Thời gian:\*\*\s*`?([^`\n]+)`?', turn_text)
        t_str = time_m[0] if time_m else ""
        print(f"--- Turn {i}: {first_line[:70]} | Time: {t_str}")
        for line in turn_text.split('\n'):
            if any(term in line.lower() for term in ['5728', '5729', '5733', '5735', '5737', '5738', '5739', '5758', 'chạy xong', 'suy luận', '1966s', '2104s', '2327s', '2357s', '2649s', '2591s', 'phút', 'tiếng']):
                print(f"    {line.strip()[:100]}")
