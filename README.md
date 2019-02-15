# PyTorchでMatrix Factorization

## Matrix Factorization
気持ちはこうで、実際にはすべてを同一に最適化できないので、ミニバッチを切り出して順次学習していく
<div align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/4949982/52837133-f0397780-3130-11e9-82f9-a9fda4764063.png">
</div>
一つのデータセットの粒度は、下図のようにEmbedding Layerで整数値をEmbeddingしたあと、内積を計算して映画の評価点を予想する。  
<div align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/4949982/52838383-d9495400-3135-11e9-9385-f0ff2aefad6c.png">
</div>

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
