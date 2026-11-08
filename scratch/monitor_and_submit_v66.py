import time
import datetime
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

slug = 'baonguyenduyquoc/ntl-champion-v66-n-gram-semantic-clustering-apex'
comp = 'name-that-llm'
sub_msg = 'Champion V66 N-Gram Semantic Clustering Apex (N-Gram 1-2 Clustering ARI 0.954 sweet-spot thr=0.08 + Meta Llama-3.1-8B w=2.4 dp=True + Adaptive Hungarian >=5 + Sinkhorn 0.45)'

print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting monitoring for {slug}...")

while True:
    st = api.kernels_status(slug)
    status_str = str(getattr(st, 'status', ''))
    failure_msg = getattr(st, 'failure_message', None)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Status: {status_str}", flush=True)
    
    if 'COMPLETE' in status_str:
        print(f"Kernel completed! Initiating competition code submission...", flush=True)
        res = api.competition_submit_code(
            file_name='submission.parquet',
            message=sub_msg,
            competition=comp,
            kernel=slug,
            kernel_version=1
        )
        print("-> SUBMISSION SUCCESS! Response:", res, flush=True)
        with open('scratch/v66_submission_result.txt', 'w', encoding='utf-8') as f:
            f.write(str(res))
        break
    elif 'ERROR' in status_str or 'CANCEL' in status_str:
        print(f"Kernel failed with status: {status_str}, error: {failure_msg}", flush=True)
        with open('scratch/v66_submission_error.txt', 'w', encoding='utf-8') as f:
            f.write(f"Error: {status_str}, msg: {failure_msg}")
        break
    
    time.sleep(20)

