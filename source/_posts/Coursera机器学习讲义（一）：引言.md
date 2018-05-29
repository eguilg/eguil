---
title: Coursera机器学习讲义（一）：引言
date: 2017-01-14 20:18:32
tags: [机器学习,Coursera]
categories: [Coursera机器学习讲义]
---

## 什么是机器学习

这里提供了两种机器学习的定义。Arthur Samuel的描述为:

> "the field of study that gives computers the ability to learn without being explicitly programmed."

让计算机不被显示编程即具有学习能力的研究领域。

Tom Mitchell给出了更现代的定义:

> "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

计算机针对某一类任务$T$和与之对应的表现测量标准$P$从经验$E$中进行学习，并且它在任务$T$中的表现（由标准$P$测量），随着经验$E$而提高。

## 机器学习的分类

总的来说，所有机器学习的问题都可以被分为监督学习和无监督学习两类。

### 监督学习

在监督学习中，我们得首先有一组数据集，知道与这个数据集对应的正确输出是什么样，并且认为输入和输出之间有一定的关系。监督学习问题分为“回归”和“分类”问题。

在回归问题中，我们想要预测的结果所在的集合是连续的，即我们想要将输入的变量映射到一个连续的函数上。

在分类问题中，我们想要预测的结果是离散的，即我们想要将输入变量映射为离散的类别。

### 无监督学习

无监督学习让我们能够解决预先并不知道输出结果的问题。我们不必知道所有输入的含义，只用基于数据中变量之间的关系对数据进行聚类，就能的到数据的组成类别。无监督学习对于得到的结果不会有反馈。