import time, datetime
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

slug = 'baonguyenduyquoc/ntl-champion-v68-orthogonal-style-apex'
msg = 'Champion V68 Orthogonal Style Apex (V63 0.662 Anchor + Orthogonal Word-Length Bias w=0.05 on size>=6 + Latin-Square Hungarian >=5 + Sinkhorn 0.45)'

def submit_v68():
    st = api.kernels_status(slug)
    status_str = getattr(st, 'status', None)
    print(f"[{datetime.datetime.now()}] Kernel status: {status_str}")
    if status_str != 'complete':
        print(f"Kernel not ready yet: {status_str}")
        return None
    print(f"Submitting {slug} to competition name-that-llm...")
    res = api.competition_submit_code(
        file_name='submission.parquet',
        message=msg,
        competition='name-that-llm',
        kernel=slug,
        kernel_version=1
    )
    print("-> SUBMISSION SUCCESS! Response:", res)
    return res

if __name__ == '__main__':
    submit_v68()
