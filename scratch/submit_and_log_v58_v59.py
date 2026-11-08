import subprocess, time, json

def submit_kernel(slug, version, message):
    cmd = [
        'kaggle', 'competitions', 'submit',
        '-c', 'name-that-llm',
        '-k', slug,
        '-f', 'submission.parquet',
        '-v', str(version),
        '-m', message
    ]
    print('Submitting:', ' '.join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print('STDOUT:', res.stdout)
    print('STDERR:', res.stderr)
    return res.returncode == 0

def check_status(slug):
    res = subprocess.run(['kaggle', 'kernels', 'status', slug], capture_output=True, text=True)
    return 'COMPLETE' in res.stdout

slug_v58 = 'baonguyenduyquoc/ntl-champion-v58-likelihood-titan'
slug_v59 = 'baonguyenduyquoc/ntl-champion-v59-prompt-aware-sovereign'

v58_done = check_status(slug_v58)
v59_done = check_status(slug_v59)
print(f'V58 Done: {v58_done}, V59 Done: {v59_done}')

if v58_done:
    submit_kernel(
        slug_v58, 1,
        'Champion V58 Multi-Family Likelihood Titan (V50 Golden Anchor + ATLAS 22% Sweet Spot + LogSumExp Sibling Conservation Gate + Damped Sinkhorn 0.45)'
    )

if v59_done:
    submit_kernel(
        slug_v59, 1,
        'Champion V59 Prompt-Aware Relative Sovereign (V50 Golden Anchor + Empirical Length Bias Z-Score + Soft Injective Collision Dispersion + Damped Sinkhorn 0.45)'
    )

