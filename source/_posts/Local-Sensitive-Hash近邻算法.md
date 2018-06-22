---
title: Local Sensitive Hash近邻算法
date: 2018-06-22 08:53:10
tags: [机器学习, Spark]
categories: [DM On Spark]
description: LSH近邻查找算法的理解以及使用pyspark的01 label版本实现
mathjax: true
toc: true
---
## 问题引入
数据包含大量```(u_id, p_id)```记录，代表用户```u_id```对产品```p_id```有评分，要求根据各产品的被评分记录(或各用户对产品的评分记录)得到相似度符合某个条件的所有product pair(或user pair)，两者方法一致，本文将针对查找相似product pair展开讨论。

## Jaccard相似度
Jaccard相似度表示两个集合之间的相似度，用其交并比表示：
$$SIM(S, T) = \frac{|S\cap T|}{|S\cup T|}$$
根据原始数据，每一个product可以由所有评价过该产品的用户集合来描述，即
$$S_{p\_id} = \{u\_id_1, u\_id_2, u\_id_3...\}$$
所以两个产品之间的相似度可以表示为：
$$SIM(S_{p\_id_1}, S_{p\_id_2}) = \frac{|S_{p\_id_1}\cap S_{p\_id_2}|}{|S_{p\_id_1}\cup  S_{p\_id_2}|}$$
求Jaccard相似度python实现：
```python
def jaccard_similarity(set_a, set_b):
    if len(set_a) == 0 and len(set_b) == 0:
        return 0
    return float(len(set_a.intersection(set_b)))/len(set_a.union(set_b))
```
##  数据的矩阵表示
可以使用稀疏矩阵来存储用户对产品的评价关系，矩阵表示如下：

user\product | $$S_{p\_id_1}$$ | $$S_{p\_id_2}$$| $$S_{p\_id_3}$$| ...
 :-: | :-: | :-:| :-:|
$$u\_id_1$$ | 1 | 0 |0|...
$$u\_id_2$$ | 0 | 1 |0|...
$$u\_id_3$$ | 1 | 0 |0|...
 ...|...|...|...|...
 
 其中每一列都是一个根据评分用户集合来描述产品的向量。
## MinHashing
本节介绍的minhash方法会将每个产品所对应的用户向量映射为一个更短小的signature，并且仍然可以用于区分各个产品的相似度。对于给定的signature长度```n```，随机产生```n```个对于用户向量元素顺序的重新排列，可以由```n```个哈希函数实现。signature的各个元素就是原用户向量通过相应的哈希函数重排后的第一个非0元素的index。
#### minHash和Jaccard相似度
minhash算法与待哈希集合的Jaccard相似度有着密切的联系。
> 两个集合通过minhash函数得到相同值的概率等于它们的Jaccard相似度。

分析两个产品的评分用户向量，对于某一用户(矩阵的某一列)，会有以下三种情况：

- a. ```(1, 1)```即该用户对两个产品都有评分
- b. ```(0, 1)```(或```(1, 0)```)该用户只对其中一个产品有评分
- c. ```(0, 0)```该用户对两个产品均没有评分

对于任意一个重排方法，只分析两列用户向量中第一个有评分的index，则该index位置的评分情况非a即b，其中a情况即为上述所说的通过minhash函数得到相同的值。
对这两个产品都有评分的用户数为```x```，对这两个产品其中一个有评分的用户数为```y```，通过哈希函数重排后两个用户向量第一个非零元素index相同的概率为
$$\frac{x}{x+y} = SIM(S_{p\_id_1}, S_{p\_id_2})= \frac{|S_{p\_id_1}\cap S_{p\_id_2}|}{|S_{p\_id_1}\cup  S_{p\_id_2}|}$$
#### minHash Signature计算	
实际上，随机生成userid的重排计算量巨大，但该过程可以使用hash函数代替。即
	$$u\_id_{new} = ((a\ *\ u\_id + b)\ \%\ p)\ \%\ m$$
其中a、b为随机数，p为随机质数，m为用户总量。输出即为输入的用户索引在重排之后的索引值。对于一个产品的用户向量，使用每个哈希函数求出其中有评分的用户的重排索引值中的最小值(即第一个非零索引)，得到的向量即为该产品的minhash signature。

下面是针对一个产品计算minhash的实现, 输入为**[对该产品有评分的用户索引列表]**和**[哈希函数列表]**：
```python
def gen_min_hash(userid_list, hash_fn_list):
    """
    generate minhash for one product
    :param userid_list: the rated user id list
    :param hash_fn_list: list of hash function parameters, format of each element
    is [a, b, p, m] while a and b is random numbers, p is an random prime number
    and m is the maximum of user_id index
    :return: min hash signature 
    """
    min_hash = []
    for i in range(0, len(hash_fn_list)):
        hash_list = []
        for first_user_id in range(0, len(userid_list)):
            a, b, p, m = hash_fn_list[i]
            new_index = ((a * first_user_id + b) % p) % m
            hash_list.append(new_index)
        min_hash.append(min(hash_list))
    return min_hash
```
## Local Sensitive Hashing





 
 




