import time, sys
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

slug = 'baonguyenduyquoc/champion-v93'
msg = 'Champion V93 Grandmaster Supreme (72.73% CV + Sibling Residuals glm=22/gpt=1.2/gem=0.9 + Bipartite Matching on Test Prompts enabled via perpassage=0)'

print(f"Monitoring {slug} until commit completes...", flush=True)
while True:
    res = api.kernels_status(slug)
    st = str(res.status)
    print(f"[{time.strftime('%H:%M:%S')}] Status: {st}", flush=True)
    if 'COMPLETE' in st:
        print("Commit complete! Submitting to competition name-that-llm...", flush=True)
        try:
            sub = api.competition_submit_code(
                file_name='submission.parquet',
                message=msg,
                competition='name-that-llm',
                kernel=slug,
                kernel_version=1
            )
            print("🚀 SUCCESS! Submission response:", sub, flush=True)
        except Exception as e:
            print("❌ Submission error:", e, flush=True)
        break
    elif 'ERROR' in st or 'CANCEL' in st:
        print(f"❌ Kernel failed with status: {st}, failure: {getattr(res, 'failure_message', '')}", flush=True)
        break
    time.sleep(15)

