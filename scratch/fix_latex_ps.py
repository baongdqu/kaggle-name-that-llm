with open('abc PROBLEM_STATEMENT.md', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('1/18  pprox 5.55%', '1/18 \\approx 5.55%')
content = content.replace('$ pprox$', '$\\approx$')
content = content.replace('18 lớp $ pprox$ 88', '18 lớp $\\approx$ 88')
content = content.replace('+98 \text{ mẫu đúng ròng}', '+98 \\text{ mẫu đúng ròng}')
content = content.replace('+98 \text', '+98 \\text')
content = content.replace('+98 \t ext', '+98 \\text')

with open('abc PROBLEM_STATEMENT.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("FIXED LATEX ESCAPING IN abc PROBLEM_STATEMENT.md!")
