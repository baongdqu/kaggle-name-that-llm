import time, datetime
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

slug = 'baonguyenduyquoc/ntl-champion-v60-apex-synergy'
msg = 'Champion V60 Apex Synergy (0.662 V59 Prompt-Aware Core + Relative Length Bias + Soft Typography Guards + LogSumExp Sibling Sharpening + Sinkhorn 0.45)'

print(f"[{datetime.datetime.now()}] Monitoring {slug} for completion...")
for attempt in range(60):
    status_obj = api.kernels_status(slug)
    status = getattr(status_obj, 'status', None)
    failure = getattr(status_obj, 'failure_message', None)
    print(f"Attempt {attempt+1}/60: status = {status}, failure = {failure}")
    if str(status).upper() == 'COMPLETE' or 'COMPLETE' in str(status).upper():
        print(f"[{datetime.datetime.now()}] Kernel COMPLETE! Submitting to competition...")
        try:
            res = api.competition_submit_code(
                file_name='submission.parquet',
                message=msg,
                competition='name-that-llm',
                kernel=slug,
                kernel_version=1
            )
            print("-> SUCCESS! Submission response:", res)
            break
        except Exception as e:
            print("Submission error:", e)
            break
    elif str(status).upper() == 'ERROR' or 'ERROR' in str(status).upper():
        print("Kernel failed with ERROR!")
        break
    time.sleep(10)
