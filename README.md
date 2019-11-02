# python-r-stan-bayesian-model
"RとStanではじめるベイズ統計モデリングによるデータ分析入門" (馬場真哉) を Pythonで書籍の内容を記述しなおしたコードです。
最近はデータ分析にPythonをもちいることが多く、各種機械学習、深層学習やベイズ統計モデルなどの各種手法をスムースに分析できることが多いため、
全コードをPythonで再実装しました。

## 第１章
こちらは理論的な説明の章ですので割愛します。

## 第２章
データ分析に最低限必要なテクニックを解説する章です。また、Stanについても最適現の知識が解説されます。

- 2-1.py 四則演算、pandas DataFrame、Seriesの基本的な使い方、乱数生成の基礎
- 2-2.py データの読み込み、グラフ描画（ヒストグラム、カーネル密度推定、共分散、相関係数、コレログラム）
- 2-3.py グラフ描画、グラフ描画（ヒストグラム、カーネル密度推定、ボックスプロット、バイオリンプロット、ペアプロット、ラインプロット）
- 2-4.py 説明変数無しの正規分布のMCMC推定(2-4-1-calc-mean-variance.stan)
- 2-5.py 事後予測分布の推定
- 2-5-2.py 事後予測分布とサンプルの分布比較
- 2-6.py Stanの文法とコーディング
- 2-6-2.py 事後予測分布による平均値の差の比較

### 第３章
基本的な一般化線形モデル（GLM Generalized Linear Models）を学ぶ章です。本書ではライブラリbrmsで記述されているパートもすべてStanで記述しなおしています。
応答変数、説明変数、線形予測子の組み合わせでモデルを構築できるようになります。

- 3-2.py 単回帰モデル
- 3-3.py 単回帰モデルを用いた予測
- 3-5.py 回帰直線の図示
- 3-6.py ダミー変数と分散分析モデル(質的変数の正規分布モデル)
- 3-7.py 正規線形モデル（複数の説明変数、応答変数が正規分布モデル）
- 3-8.py ポアソン回帰モデル
- 3-9.py ロジスティック回帰モデル（2値データ用標準手法）
- 3-10.py 交互作用(カテゴリxカテゴリ)
- 3-10-2.py 交互作用（カテゴリx数量）
- 3-10-3.py 交互作用（数量x数量）
