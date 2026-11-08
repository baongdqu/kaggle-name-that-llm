import subprocess, time, json

def get_status(slug):
    res = subprocess.run(['kaggle', 'kernels', 'status', slug], capture_output=True, text=True)
    return res.stdout

slug_v58 = 'baonguyenduyquoc/ntl-champion-v58-likelihood-titan'
slug_v59 = 'baonguyenduyquoc/ntl-champion-v59-prompt-aware-sovereign'

print('V58 status:', get_status(slug_v58).strip())
print('V59 status:', get_status(slug_v59).strip())

