import pandas as pd
import random
import numpy as np
df = pd.read_csv('works/dataset/preprocess.csv')

userIds = list(df['userId'].unique())
testIds = set(random.sample(userIds, len(userIds)//5))
trainIds = set(userId for userId in userIds if userId not in testIds)


for index, chunk_test_ids in enumerate(np.array_split(list(testIds), 10)):
    chunks = set(chunk_test_ids.tolist())
    df[df['userId'].apply(lambda x:x in chunks)].to_csv(
        f'works/dataset/test_{index:03d}.csv', index=None)

for index, chunk_train_ids in enumerate(np.array_split(list(trainIds), 10)):
    print('make trian dataset', index)
    chunks = set(chunk_train_ids.tolist())
    df[df['userId'].apply(lambda x:x in chunks)].to_csv(
        f'works/dataset/train_{index:03d}.csv', index=None)
