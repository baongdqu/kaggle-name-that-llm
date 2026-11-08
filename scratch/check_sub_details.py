from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
subs = api.competition_submissions('name-that-llm')
for s in subs[:6]:
    err = getattr(s, 'error_description', None)
    print(f"Ref: {s.ref} | Status: {s.status} | Score: {s.public_score} | Error: {err}")

