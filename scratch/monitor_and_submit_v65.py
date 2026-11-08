import time
import datetime
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

slug = 'baonguyenduyquoc/ntl-champion-v65-llama-amplified-apex'
comp = 'name-that-llm'
sub_msg = 'Champion V65 Llama Amplified Apex (Meta Llama-3.1-8B w=2.4 dp=True + Restored Clean mb_v5 w=2.0 + Qwen 14B w=2.8 + Adaptive Hungarian >=5 + Sinkhorn 0.45)'

print(f"Starting monitoring for {slug} at {datetime.datetime.now().strftime('%H:%M:%S')}...")

while True:
    st = api.kernels_status(slug)
    status_str = str(getattr(st, 'status', ''))
    failure_msg = getattr(st, 'failure_message', None)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Status: {status_str}")
    
    if 'COMPLETE' in status_str:
        print(f"Kernel completed! Initiating competition code submission...")
        res = api.competition_submit_code(
            file_name='submission.parquet',
            message=sub_msg,
            competition=comp,
            kernel=slug,
            kernel_version=1
        )
        print("-> SUBMISSION SUCCESS! Response:", res)
        with open('scratch/v65_submission_result.txt', 'w', encoding='utf-8') as f:
            f.write(str(res))
        break
    elif 'ERROR' in status_str or 'CANCEL' in status_str:
        print(f"Kernel failed with status: {status_str}, error: {failure_msg}")
        with open('scratch/v65_submission_error.txt', 'w', encoding='utf-8') as f:
            f.write(f"Error: {status_str}, msg: {failure_msg}")
        break
    
    time.sleep(20)

