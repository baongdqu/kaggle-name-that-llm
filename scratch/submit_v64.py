import time, datetime
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

slug = 'baonguyenduyquoc/ntl-champion-v64-sovereign-modernbert-apex'
msg = 'Champion V64 Sovereign ModernBERT Apex (Sovereign ModernBERT-Large 31k Pool w=2.2 + Calibrated T=0.684 + Adaptive Hungarian >=5 + Sinkhorn 0.45 + Qwen 1.0 + Llama 1.8 dp)'

def submit_v64():
    st = api.kernels_status(slug)
    status = getattr(st, 'status', None)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Kernel status: {status}")
    if status == 'KernelWorkerStatus.COMPLETE':
        print(f"Submitting {slug} to competition name-that-llm...")
        res = api.competition_submit_code(
            file_name='submission.parquet',
            message=msg,
            competition='name-that-llm',
            kernel=slug,
            kernel_version=1
        )
        print("-> SUBMISSION SUCCESS! Response:", res)
        return True, res
    return False, status

if __name__ == '__main__':
    ok, res = submit_v64()
