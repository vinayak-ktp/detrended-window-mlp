import os
import sys

# sys.path.append('..')
# print(sys.path)
import numpy as np
import pandas as pd

from data_pipeline.preprocessing import normalize_features, split_timeseries

os.makedirs('data/splits', exist_ok=True)

df = pd.read_csv('data/processed/combined_data.csv', parse_dates=['time'])

train, val, test = split_timeseries(df, train_size=0.7, val_size=0.15)
train_norm, val_norm, test_norm, stats = normalize_features(train, val, test, ['sp', 't2m'])

train_norm.to_csv('data/splits/train.csv', index=False)
val_norm.to_csv('data/splits/val.csv', index=False)
test_norm.to_csv('data/splits/test.csv', index=False)
