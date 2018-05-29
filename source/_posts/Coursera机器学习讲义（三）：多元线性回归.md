---
title: Coursera机器学习讲义（三）：多元线性回归
date: 2017-01-16 20:42:49
tags: [线性回归,梯度下降,机器学习,Coursera]
categories: [Coursera机器学习讲义]
description: 最小二乘法的变式，多元线性回归
mathjax: true
toc: true
---

## 模型描述

具有多个输入变量的线性回归叫做多元线性回归。下面是我们在描述任意数量的输入变量时所用的表示方法：

$\begin{split}x_j^{(i)}=第i组训练样本中的第j个特征值x^{(i)}\end{split}$
$\begin{split}=第i组训练样本的所有特征组成的列向量m\end{split}$
$\begin{split}=训练样本的组数n\end{split}$
$\begin{split}=|x^{(i)}|(每组样本中特征的数量)\end{split}$

对于多元线性回归我们的假设方程相应地变为：
$$h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+\theta_3x_3+...+\theta_nx_n$$
为了形象地理解这个函数的意义，我们可以把$\theta_0$想成是房子的基础价格，$\theta_1$想成是房价跟面积的系数，$\theta_2$看成房价跟楼层高低的关系，$\theta_3$看成房价跟房屋所在地段（离地铁站远近之类的）的关系等等...总之就是房屋的各种属性与房价之间关系的参数。

以上的公式表示为矩阵形式为：
$$
\begin{align}h_\theta(x) =\begin{bmatrix}\theta_0 \hspace{2em} \theta_1 \hspace{2em} ... \hspace{2em} \theta_n\end{bmatrix}\begin{bmatrix}x_0 \newline x_1 \newline \vdots \newline x_n\end{bmatrix}= \theta^T x\end{align}
$$
**注意：**$x_0^{(i)}=1;i=1,...,m$.这是为了方便$\theta^T$与$x$之间进行矩阵运算(列数和行数要匹配)。

每一个训练样本作为整个训练集$X$的一行来存储，举个栗子：
$$
\begin{align}X = \begin{bmatrix}x^{(1)}_0 & x^{(1)}_1 \newline x^{(2)}_0 & x^{(2)}_1 \newline x^{(3)}_0 & x^{(3)}_1 \end{bmatrix}=\begin{bmatrix}1 & x^{(1)}_1 \newline 1 & x^{(2)}_1 \newline1& x^{(3)}_1 \end{bmatrix},\theta = \begin{bmatrix}\theta_0 \newline \theta_1 \newline\end{bmatrix}\end{align}
$$
那么对于整个训练集，假设函数的向量化表示就是：
$$
h_\theta(X) = X \theta
$$

## 多元变量的梯度下降

梯度下降算法的形式其实并没有改变，只是每次迭代都将更新 ‘ $n$ ’ 个特征所对应的参数:

$$\begin{align} & \text{repeat until convergence:} \; \lbrace \newline \; & \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_0^{(i)}\newline \; & \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_1^{(i)} \newline \; & \theta_2 := \theta_2 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_2^{(i)} \newline & \cdots \newline \rbrace \end{align}$$

即

$$\begin{align}& \text{repeat until convergence:} \; \lbrace \newline \; & \theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \; & \text{for j := 0...n}\newline \rbrace\end{align}$$

## 特征缩放

如果保证训练集中的所有特征都在一个相对一致的范围之内，我们将能够加快梯度下降算法的收敛。

对于每一个特征方向，梯度下降的“步长”都是一样的，如果不同的特征之间数量级相差较大，会让梯度下降相对于数量级较小的特征的方向更快的下降，数量级较大的特征方向上下降缓慢。范围参差不齐的特征变量会使得梯度下降算法到达最优解的效率变得低下。

解决办法就是保证各个输入变量都大致在一个大小相似的范围之内：$±0.5-±1$之间即可（并不要求太过精确）。

两种方法：特征缩放和平均归一化

### 特征缩放

特征值除以该特征值的大小范围：
$$
x_i:={x_i\over s_i}
$$

### 平均归一化

特征值减去该特征的平均值再除以该特征的大小范围（或者标准差）：
$$
x_i:={x_i-\mu_i\over s_i}
$$

## 特征组合与多项式回归

通过对已有特征变量的组合和变换，我们可以得到非线性的假设函数：

{% asset_img  polynomial-regression.png%}

## 一般方程

即通过使代价函数导数为零，直接解得$\theta$:

$$\begin{align}
\alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_0^{(i)}=0
\newline \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_1^{(i)}=0
\newline \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_2^{(i)}=0\newline ...
\newline \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}=0
\end{align}$$

向量化之后得到：

$$X^T\cdot(X\cdot\theta-y)=0$$

解得

$$\theta=(X^TX)^{-1}X^Ty$$

但是由于需要计算$X^TX$，一般方程的时间复杂度是$\mathcal{O}(n^3)$，在特征数n较小的时候适合使用，但n较大时会变得缓慢。

## 向量化以及代码实现

