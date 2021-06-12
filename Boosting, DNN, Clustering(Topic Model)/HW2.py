#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 16:49:31 2021

@author: wangziwen
"""
from decimal import Decimal
from scipy.special import factorial
import pandas as pd
import numpy.matlib 
import numpy as np
import random
import math
import operator
from functools import reduce

"""
read data
"""
df_corpus = pd.read_table('/Users/wangziwen/Documents/Graduation/1st/Statistic Learning/HW/20news/20news.libsvm',header=None)
df_vocab = pd.read_table('/Users/wangziwen/Documents/Graduation/1st/Statistic Learning/HW/20news/20news.vocab',header=None)

df_vocab.rename({0:'id',
                 1:'word',
                 2:'freq'},axis=1,inplace = True)

df_corpus.rename({0:'docid',
                  1:'id_count'},axis=1,inplace = True)
df_corpus.iloc[18270,1] #each line is the bag-of-words representation of a document, of the format

"""
given T
"""
T = np.matlib.zeros((18271,83735))
# multi
w = 83735
d = 18271

n_d = T.sum(axis=1) #dx1
p = np.zeros((d,w)) #dxw
for i in range(d):
    for j in range(w):
        p[i,j] = (Decimal(math.factorial(T[i,j])))
n_d_f = np.zeros(d)
for i in range(d):
    n_d_f[i] = (Decimal(math.factorial(n_d[i])))
T_prod = np.prod(p, axis = 1)
ma = n_d_f/T_prod #dx1

"""
EM 這裡因為T,n_d,w,已知，故function僅調整k
"""
#k=10 set k
def multi(k,mu,pi) :      
    q = np.zeros((d,k,w))
    for i  in range(w):
        for j in range(k):
            for u in range(d):
                q[u][j][i] = mu[i,j]**T[u,i]  

    q_dk = np.zeros((d,k))
    for i in range(d):
        for j in range(k):
            q_dk[i,j] = np.prod(q[i][j]) #dxk

    multi = np.zeros((d,k))
    for i in range(d):
        for j in range(k):
            multi[i,j] = ma[i]*q_dk[i,j] 
    join = np.zeros((d,k))
    for i in range(d):
        for j in range(k):
            join[i,j] = pi[j]*q_dk[i,j]  #dxk
    margin = join.sum(axis=1)
    return multi,join,margin

#E
def E_step(k,mu,pi):
    multi,join,margin = multi(k,mu,pi)
    post = np.zeros((d,k))
    for i in range(d):
        for j in range(k):
            post[i,j] = join[i,j]/margin[i]
    return post
 
#M  
def M_step(k,post):
    pi =  (post/d).sum(axis=0)
    mu = np.zeros((w,k))
    for i in range(w):
        for j in range(k):
            for u in range(d):
                mu[i,j] = post[u,j]*T[u,i]
    mu = mu/d
    return mu, pi
        
def EM_Train(k,iter):
    pi = np.zeros(k) #create pi
    for i in range(k):
        pi[k] = 1/k
    mu = np.zeros((w,k)) #create mu
    for i in range(w):
        for j in range(k):
            mu[i,j] = 1/(w*k)
    # 开始迭代
    step = 0    
    while (step < iter):
    # 每次进入一次迭代后迭代次数加1
        step += 1
        # E步
        post = E_step(k,mu,pi) #post
        # M步
        mu, pi = M_step(k,post) 

    # 迭代结束后将更新后的各参数返回
    return mu,pi

# problem 5
freq = pd.DataFrame(mu.max(0))
freq = pd.DataFrame(mu.max(0))
freq.rename(columns = {0: 'id'},inplace = True)
result = freq.merge(df_vocab,how='left', on='id')[['id','word']]