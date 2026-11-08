"""
Script to automatically submit Champion V58 and V59 as soon as Kaggle daily quota resets (00:00 UTC / 07:00 AM Local).
"""
import time, datetime
from kaggle.api.kaggle_api_extended import KaggleApi
import requests

api = KaggleApi()
api.authenticate()

def try_submit(slug, name, msg):
    print(f"[{datetime.datetime.now()}] Attempting submission for {name} ({slug})...")
    try:
        res = api.competition_submit_code(
            file_name='submission.parquet',
            message=msg,
            competition='name-that-llm',
            kernel=slug,
            kernel_version=1
        )
        print(f"-> SUCCESS for {name}! Response: {res}")
        return True
    except requests.exceptions.HTTPError as e:
        print(f"-> Failed for {name}: {e.response.text}")
        return False
    except Exception as e:
        print(f"-> Error for {name}: {e}")
        return False

if __name__ == '__main__':
    v58_slug = 'baonguyenduyquoc/ntl-champion-v58-likelihood-titan'
    v58_msg = 'Champion V58 Multi-Family Likelihood Titan (V50 Golden Anchor + ATLAS 22% Sweet Spot + LogSumExp Sibling Conservation Gate + Damped Sinkhorn 0.45)'
    
    v59_slug = 'baonguyenduyquoc/ntl-champion-v59-prompt-aware-sovereign'
    v59_msg = 'Champion V59 Prompt-Aware Relative Sovereign (V50 Golden Anchor + Empirical Length Bias Z-Score + Soft Injective Collision Dispersion + Damped Sinkhorn 0.45)'
    
    s58 = try_submit(v58_slug, 'Champion V58', v58_msg)
    s59 = try_submit(v59_slug, 'Champion V59', v59_msg)
    print(f"Result: V58 submitted: {s58}, V59 submitted: {s59}")

