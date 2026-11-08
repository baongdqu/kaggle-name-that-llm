import time, datetime
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
slug = 'baonguyenduyquoc/ntl-train-sovereign-modernbert-31k'

st = api.kernels_status(slug)
status = getattr(st, 'status', None)
print(f'[{datetime.datetime.now().strftime("%H:%M:%S")}] Training Kernel status: {status}')

