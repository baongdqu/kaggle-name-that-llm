import subprocess, csv, io

# We need to fetch page by page or fetch all using kaggle competitions leaderboard
# Let's inspect pages
next_token = ""
all_teams = []
page = 1
while True:
    cmd = ['kaggle', 'competitions', 'leaderboard', '-c', 'name-that-llm', '--show', '--format', 'csv', '--page-size', '100']
    if next_token:
        cmd.extend(['--page-token', next_token])
    res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    lines = [l for l in res.stdout.splitlines() if l.strip()]
    next_token = ""
    for l in lines:
        if 'Next Page Token' in l:
            next_token = l.split('=')[1].strip()
    csv_start = 0
    for i, l in enumerate(lines):
        if l.startswith('teamId'):
            csv_start = i
            break
    csv_text = '\n'.join(lines[csv_start:])
    reader = csv.DictReader(io.StringIO(csv_text))
    page_rows = list(reader)
    if not page_rows:
        break
    all_teams.extend(page_rows)
    print(f"Page {page}: fetched {len(page_rows)} teams. Total: {len(all_teams)}")
    # Check if team is here
    for idx, r in enumerate(all_teams):
        if r.get('teamId') == '17137218' or 'morsi' in r.get('teamName', '').lower():
            print(f"!!! FOUND TEAM: Rank {idx+1} | {r.get('teamName')} | TeamId {r.get('teamId')} | Score {r.get('score')}")
            break
    if any(r.get('teamId') == '17137218' for r in all_teams):
        break
    if not next_token or page >= 10:
        break
    page += 1
