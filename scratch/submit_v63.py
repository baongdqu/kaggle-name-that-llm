import time, datetime
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

slug = 'baonguyenduyquoc/ntl-champion-v63-adaptive-anchor-apex'
msg = 'Champion V63 Adaptive Anchor Apex (0.662 V59 Anchor + Public-Slice Adaptive Threshold size>=5 + Latin-Square Anchor Kuhn-Munkres 98% Win-Rate + Sinkhorn 0.45 + Restored Qwen2.5-14B w=1.0 + Length Bias)'

def submit_v63():
    st = api.kernels_status(slug)
    print(f"[{datetime.datetime.now()}] Kernel status: {getattr(st, 'status', None)}")
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
    submit_v63()

