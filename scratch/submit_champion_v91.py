import sys
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

slug = 'baonguyenduyquoc/champion-v91'
msg = 'Champion V91 Judith-Sovereign Apex (54-Forensic Invariants + Clean Claude Gate + Sibling Meta-Learner + Sinkhorn 0.35)'

print(f"Submitting {slug} version 1 to competition name-that-llm...")
try:
    res = api.competition_submit_code(
        file_name='submission.parquet',
        message=msg,
        competition='name-that-llm',
        kernel=slug,
        kernel_version=1
    )
    print("🚀 SUCCESS! Submission response:", res)
except Exception as e:
    print("❌ Submission error:", e)

