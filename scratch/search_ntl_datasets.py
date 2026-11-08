import subprocess

queries = ['name-that-llm', 'ntl']
all_datasets = {}

for q in queries:
    res = subprocess.run(['kaggle', 'datasets', 'list', '-s', q, '--sort-by', 'votes', '--page-size', '100'], capture_output=True, text=True)
    lines = res.stdout.strip().split('\n')
    if len(lines) > 2:
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 2:
                ref = parts[0]
                if ref not in all_datasets:
                    all_datasets[ref] = line

print(f"Total matching NTL datasets: {len(all_datasets)}")
for ref, line in sorted(all_datasets.items(), key=lambda x: x[0]):
    print(f"{line}")

