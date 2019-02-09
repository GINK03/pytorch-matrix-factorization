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


for TYPE in ['train', 'test']:
    for index, fn in enumerate(glob.glob(f'./works/dataset/{TYPE}_*.csv')):
        df = pd.read_csv(fn)
        userIds = df['userId'].unique()
        print(len(userIds))
        sparse = Sparse((len(userIds), len(movie_index)),
                        dtype=np.float)

        for masterIndex, (userId, subDf) in enumerate(df.groupby(by=['userId'])):
            print(userId)
            for movieId, score in zip(subDf['movieId'].tolist(), subDf['score'].tolist()):
                mindex = movie_index[str(movieId)]
                sparse[masterIndex, mindex] = float(score)
        print('try to compress')
        pickle.dump(sparse, open(f'works/dataset/{TYPE}_{index:02d}.pkl', 'wb'))
        #save_npz(f'works/dataset/{TYPE}_{index:02d}.csc', sparse)
