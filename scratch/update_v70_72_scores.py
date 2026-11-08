with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.split('\n')
for i, line in enumerate(lines):
    if '57946908' in line:
        lines[i] = line.replace('*PENDING* | `PENDING`', '**0.662** | `0.662`')
    elif '57951221' in line:
        lines[i] = line.replace('*PENDING* | `PENDING`', '**0.661** | `0.661`')
    elif '57951307' in line:
        lines[i] = line.replace('*PENDING* | `PENDING`', '**0.662** | `0.662`')

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('abc SUBMISSIONS.md updated successfully!')

