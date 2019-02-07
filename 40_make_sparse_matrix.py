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
    for index, fn in enumerate(glob.glob(f'./works/dataset/{TYPE}.csv')):
        df = pd.read_csv(fn)
        userIds = df['userId'].unique()
        print(len(userIds))

        movies = Sparse((len(userIds), len(movie_index),),
                           dtype=np.float)
        users = Sparse((len(userIds), len(user_index),),
                           dtype=np.float)
        for masterIndex, (userId, subDf) in enumerate(df.groupby(by=['userId'])):
            print(userId)
            uindex = user_index[str(userId)]
            users[masterIndex, uindex] = 1.0
            for movieId, score in zip(subDf['movieId'].tolist(), subDf['score'].tolist()):
                    mindex = movie_index[str(movieId)]
                    movies[masterIndex, mindex] = float(score)
        print('try to compress')
        pickle.dump(movies, open(
            f'works/dataset/{TYPE}_movies.pkl', 'wb'))
        pickle.dump(users, open(
            f'works/dataset/{TYPE}_users.pkl', 'wb'))
        #save_npz(f'works/dataset/{TYPE}_movies.lil', movies)
        #save_npz(f'works/dataset/{TYPE}_users.lil', users)
