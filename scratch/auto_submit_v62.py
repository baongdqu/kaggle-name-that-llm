import time, datetime
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

slug = 'baonguyenduyquoc/ntl-champion-v62-anchor-sovereign-apex'
msg = 'Champion V62 Anchor Sovereign Apex (0.662 V59 Anchor + Restored Qwen2.5-14B w=1.0 + Sinkhorn 0.45 + Latin-Square Anchor-Protected Kuhn-Munkres 100% Win-Rate + Beta-Bernoulli Typo Clip 1.2 + Prompt-Aware Length Bias)'

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
        print("Kernel failed with ERROR!", failure)
        break
    time.sleep(10)

