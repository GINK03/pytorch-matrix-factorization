from pathlib import Path
import glob
from io import StringIO
import pandas as pd


def get_movie(fn):
    lines = open(fn).readlines()
    movie = lines.pop(0).strip()
    csv = ''.join(lines)
    csv = StringIO(csv)
    df = pd.read_csv(csv, header=None, sep=',')
    df.columns = ['userId', 'score', 'date']
    df['movieId'] = movie.replace(':', '')
    df = df.drop(['date'], axis=1)
    # print(df.head())
    return df


dfs = []
files = glob.glob('./download/training_set/*.txt')
for index, fn in enumerate(files):
    print(index, len(files), fn)
    df = get_movie(fn)
    dfs.append(df)

df = pd.concat(dfs, axis=0)
Path('works/dataset').mkdir(exist_ok=True, parents=True)
df.to_csv('works/dataset/preprocess.csv', index=None)
