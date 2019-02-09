import pandas as pd
import json
from pathlib import Path

Path('works/defs').mkdir(exist_ok=True, parents=True)

df = pd.read_csv('works/dataset/preprocess.csv')

user_index = {userid: index for index,
              userid in enumerate(df['userId'].unique().tolist())}
movie_index = {movie: index for index,
               movie in enumerate(df['movieId'].unique().tolist())}

json.dump(user_index, fp=open('works/defs/user_index.json', 'w'), indent=2)
json.dump(movie_index, fp=open('works/defs/smovie_index.json', 'w'), indent=2)
