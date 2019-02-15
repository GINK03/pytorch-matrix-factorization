# PyTorchでMatrix Factorization

## Matrix Factorization

## PyTorchでの実装

## 他のフレームワークでの実装
 - [DEEP BEERS: Playing with Deep Recommendation Engines Using Keras](https://medium.com/data-from-the-trenches/deep-beers-playing-with-deep-recommendation-engines-using-keras-part-1-1efc4779568f)

## データ・セット
 - [NetFlix Prize](http://academictorrents.com/details/9b13183dc4d60676b773c9e2cd6de5e5542cee9a)


## 前処理と学習
**断片になったcsvを結合**  
```console
$ python3 10_preprocessing.py
```
**userとmovieにインデックスをふる**
```console
$ python3 20_make_movieId_userId_defs.py
```
**trainとtestデータに分割**
```console
$ python3 30_split_train_test.py
```
**Pythonの標準データフレームに変換**  
```console
$ python3 40_make_sparse_matrix.py
```
**学習**
```console
$ python3 60_matrix_factorization.py
```

## 結果
