import numpy as np
from scipy.sparse import lil_matrix as Sparse
import pandas as pd
import json
import pickle
import gzip
import glob
from scipy.sparse import save_npz

user_index = json.load(open('works/defs/user_index.json'))
movie_index = json.load(open('works/defs/smovie_index.json'))


for TYPE in ['test', 'train']:
    X = []
    fn = f'./works/dataset/{TYPE}.csv'
    df = pd.read_csv(fn)
    userIds = df['userId'].unique()
    print(len(userIds))
    for masterIndex, (userId, subDf) in enumerate(df.groupby(by=['userId'])):
        print(userId)
        uindex = user_index[str(userId)]
        for movieId, score in zip(subDf['movieId'].tolist(), subDf['score'].tolist()):
            mindex = movie_index[str(movieId)]
            #print(uindex, mindex, score)
            X.append([uindex, mindex, score])
        if masterIndex > 10000:
            break
    print('try to compress')
    pickle.dump(X, open(
        f'works/dataset/{TYPE}_triples.pkl', 'wb'))
