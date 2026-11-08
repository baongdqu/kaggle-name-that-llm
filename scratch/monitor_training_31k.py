import time, datetime
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

slug = 'baonguyenduyquoc/ntl-train-sovereign-modernbert-31k'
print(f"[{datetime.datetime.now()}] Monitoring {slug} for training completion...")

for attempt in range(120): # Check up to 60 minutes
    status_obj = api.kernels_status(slug)
    status = getattr(status_obj, 'status', None)
    failure = getattr(status_obj, 'failure_message', None)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Attempt {attempt+1}/120: status = {status}, failure = {failure}", flush=True)
    if 'COMPLETE' in str(status).upper():
        print(f"[{datetime.datetime.now()}] Training COMPLETE! Artifacts are ready on Kaggle!")
        break
    elif 'ERROR' in str(status).upper():
        print(f"[{datetime.datetime.now()}] Training failed with ERROR: {failure}")
        break
    time.sleep(30)

