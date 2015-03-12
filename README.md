# Bayesian Optimization

ベイズ最適化の勉強用コード

## ベイズ最適化とは

Black-box 関数 f(x) をガウス過程を用いて最適化する手法（たぶん）  
x_opt = argmax_x f(x) を効率よく探索して見つけたい。
f(x) の評価に時間がかかると仮定。
手順は大体以下の通り。

0. t = 0, D_t = {}
1. x_t = argmax A(x | D_t)
2. y_t = f(x_t)
3. D_{t+1} = D_t \cup {(x_t, y_t)}
4. t=t+1 として 1. へ

つまり直接最適化が難しい f(x) の代わりに、  
直接最適化が楽な（可能な）A(x | D_t) を繰り返し最適化する。  
A(x) は Acquisition function と呼ばれ、大体以下のやつらが有名。

1. Maximum Mean (MM)
   * GP の事後分布の mean を最大化する点を x_t とする
2. Probaiblity of Improvement (PI)
   * これまでのベストを更新する確率が最大となる点を x_t とする
3. Expected Improvement (EI)
   * これまでのベストに対してどれだけ更新できそうかの期待値を最大化する点を x_t とする
4. Thompson Sampling (TS)
   * GP の事後分布から関数をサンプリングし、それを最大化する点を x_t とする

評価基準は以下の２つ。  
ただし普通は計算できない。

1. 累積リグレット
   * f(x_opt) - f(x_t) の総和
2. 最適識別リグレット
   * f(x_opt) - f(x_t^+) where x_t^+ = t までの最高の点

## プログラム

### animation.py

適当な関数に対して実際にベイズ最適化を適用した要すをアニメーションとして  
プロット or gif ファイルに保存する。  
使い方は以下。

    $ ./animation.py -h
    Usage: animation.py [options]

    Options:
       -h, --help          show this help message and exit
       -f GIF, --file=GIF  gif file name
       -a ts, --acq=ts     Acquisition Function
       -n 50, --nfs=50     # frames
       --fps=5             Frames per second
       -N 10               # candidate of kernel parameters

