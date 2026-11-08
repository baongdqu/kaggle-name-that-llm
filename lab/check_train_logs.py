import kaggle
import json

api = kaggle.KaggleApi()
api.authenticate()
slug = 'baonguyenduyquoc/ntl-train-sovereign-modernbert-31k'

logs = api.kernels_logs(slug)
print('Type of logs:', type(logs))
if isinstance(logs, str):
    try:
        parsed = json.loads(logs)
        for p in parsed:
            print(f"[{p.get('stream_name')}] {p.get('data')}", end='')
    except Exception as e:
        print('Raw string logs:\n', logs)
elif isinstance(logs, list):
    for p in logs:
        if isinstance(p, dict):
            print(f"[{p.get('stream_name')}] {p.get('data')}", end='')
        else:
            print(p)

