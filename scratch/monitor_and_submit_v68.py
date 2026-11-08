import time
import datetime
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

slug = 'baonguyenduyquoc/ntl-champion-v68-orthogonal-style-apex'
comp = 'name-that-llm'
sub_msg = 'Champion V68 Orthogonal Style Apex (V63 0.662 Anchor + Orthogonal Relative Word Length Bias w=0.05 on size>=6 + Latin-Square Hungarian >=5 + Sinkhorn 0.45)'

print(f"Starting monitoring for {slug} at {datetime.datetime.now().strftime('%H:%M:%S')}...", flush=True)

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
        with open('scratch/v68_submission_result.txt', 'w', encoding='utf-8') as f:
            f.write(str(res))
        break
    elif 'ERROR' in status_str or 'CANCEL' in status_str:
        print(f"Kernel failed with status: {status_str}, error: {failure_msg}", flush=True)
        with open('scratch/v68_submission_error.txt', 'w', encoding='utf-8') as f:
            f.write(f"Error: {status_str}, msg: {failure_msg}")
        break
    
    time.sleep(20)

