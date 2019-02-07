import pandas as pd
import random
import numpy as np
df = pd.read_csv('works/dataset/preprocess.csv')

# shuffle records
df = df.sample(frac=1)

dfTest = df[-len(df)//5:]
dfTrain = df[:-len(df)//5]


dfTrain.sort_values(by=['userId']).to_csv(
    f'works/dataset/train.csv', index=None)
dfTest.sort_values(by=['userId']).to_csv(f'works/dataset/test.csv', index=None)
