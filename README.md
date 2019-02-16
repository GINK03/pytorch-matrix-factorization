# PyTorchでMatrix Factorization

10年前のNetflix Prizeで確立された（？）、Matrix Factrizationは多くの場合、SVDというアルゴリズムで解くことができるが、ロジックと数式をぼんやりと見ていたら、Deep Learningでもできるっぽいなと思った。  

ググると、Pytorchでの実装をここなっている人[1], Kerasでの実装を行っている人[2]を見つけることができた。[2]によると、内積を計算することを最終目標とするのであるが、どうやらその内部は非線形であってもいいらしく、表現力を高めるような深いネットワークの構成でも性能がでるようである。  

Pytorchで実装を行い、簡単に性能をそれなりに出せたので忘備録として残しておく。  

## Matrix Factorization
気持ちはこうで、実際にはすべてを同一に最適化できないので、ミニバッチを切り出して順次学習していく
<div align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/4949982/52837133-f0397780-3130-11e9-82f9-a9fda4764063.png">
</div>
一つのデータセットの粒度は、下図のようにEmbedding Layerで整数値をEmbeddingしたあと、内積を計算して映画の評価点を予想する。  
<div align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/4949982/52838383-d9495400-3135-11e9-9385-f0ff2aefad6c.png">
</div>

## 具体的な実装
厳密なMatrix Factrizationでの定義であるところのコサイン類似度を計算して、UserとItemの近さを出してもよいが、せっかくDeepLearningフレームワークを利用するので、内積ではなく、全結合を利用することもできる。  

そして、実際に性能が良いようである。  

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
NetFlix Prizeのロスは0.8前後なのでここまでは下がらないけど、シンプルなモデルでそここそこの結果を出せる。  

```console
$ python3 60_matrix_factorization.py 
user size 480189 item size 17770
item size 17770
rmse 3.736581492357837
rmse 1.9679039708406718
rmse 1.8339715235914231
rmse 1.723884858665941
rmse 1.6683048474397226
rmse 1.6050250868125016
rmse 1.5326026325846749
rmse 1.5022485451718044
rmse 1.436461702330561
rmse 1.4173315269913291
rmse 1.4158676689106966
rmse 1.3402933459666742
rmse 1.2894482545594244
rmse 1.261695349458166
rmse 1.2595036448705352
rmse 1.3063188623813087
...
rmse 1.080400405348696
```

## 参考
 - [1][PyTorchでもMatrix Factorizationがしたい！](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwiU-dHIjsDgAhWSFYgKHS8xBlwQFjAAegQIChAB&url=https%3A%2F%2Ftakuti.me%2Fnote%2Fpytorch-mf%2F&usg=AOvVaw3daJmWblCK3v9ksv0DEpAA)
 - [2][DEEP BEERS: Playing with Deep Recommendation Engines Using Keras](https://medium.com/data-from-the-trenches/deep-beers-playing-with-deep-recommendation-engines-using-keras-part-1-1efc4779568f)
