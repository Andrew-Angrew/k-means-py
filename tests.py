# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:02:18 2017

@author: Andrew
"""

import numpy as np
from numpy.random import random, randn
import csv
from dummy import dist
from classic import classic_k_means
from yinyang import yinyang_k_means
from my_k_means import my_k_means
from my_k_means_turbo import turbo
from sklearn.cluster import KMeans
from time import clock

def generate_data(n, d, seed=42, true_k=None, true_d=None, 
                  noise=0, cluster_sparsity=1/10):
    np.random.seed(seed)
    true_k = true_k or n
    true_d = true_d or d
    true_clusters = random((true_k, true_d))
    if true_k == n:
        data = true_clusters
    else:
        true_assignments = np.random.randint(true_k, size = n)
        data = np.array([true_clusters[true_assignments[i]] + 
                              randn(true_d)*cluster_sparsity*np.sqrt(1/6)
                              for i in range(n)])
    if true_d < d:
        embedding = randn(true_d,d)
        data = data.dot(embedding)
    return data + noise * randn(n,d)
    
def load_mnist(n, d, noise=0, seed=42):
    np.random.seed(seed)
    assert d <= 784
    data = []
    with open("C:/Andrew/data/mnist/train.csv") as f:
        csv_f = csv.reader(f)
        t = 0
        for row in csv_f:
            if t > 0:
                data.append([float(w)/256 for w in row[1:]])
            t += 1
            if t > n:
                break
    data = np.array(data)
    v1 = data.var(axis = 0)
    v2 = v1.copy()
    v1.sort()
    indices = [i for i in range(784) if v2[i] >= v1[-d]]
    data = data[:,indices]
    return data + noise * randn(*data.shape)

def make_num(w):
    try:
        return float(w)
    except:
        return 0.

def load_kegg_net(noise=0.001,seed=42):
    np.random.seed(seed)
    data = []
    with open("C:/Andrew/data/Reaction Network (Undirected).data") as f:
        csv_f = csv.reader(f)
        data = np.array([[make_num(w) for w in row[1:]] for row in csv_f])
    data -= np.mean(data, axis = 0)
    data /= np.std(data, axis = 0)
    return data + noise * randn(*data.shape)
    
    
n, k, d, seed = 1000, 32, 32, 42
#data1 = generate_data(n, d, seed = seed)
#data2 = generate_data(n, d, true_d = 6, true_k = 200, noise = 0.025, seed = seed)
#data3 = load_mnist(n, d, noise = 0.025, seed = seed)
data4 = load_kegg_net(seed = seed)[:n]

print("n,d = ", data4.shape)
for k in [4]:
    
    print("k = {}:".format(k))
    
    start = clock()
    clas = classic_k_means(data4, k, empty_strat = 'spare')
    clas.fit()
    clas_t = clock() - start
    print("iter = ", clas.t, "{:.4}".format(clas_t), end = " | ")

    start = clock()
    yin = yinyang_k_means(data4, k)
    yin_t = clock() - start
    print("{:.4}".format(clas_t/yin_t), end = " | ")
        
    start = clock()
    my = my_k_means(data4, k, groups_strat = 'clustered', 
                     empty_strat = 'spare')
    my.fit()
    my_t = clock() - start
    print("{:.4}".format(clas_t/my_t), end = " | " )
    
    start = clock()
    tur = turbo(data4, k, groups_strat = 'clustered', 
                     empty_strat = 'spare')
    tur.fit()
    tur_t = clock() - start
    print("{:.4}".format(clas_t/tur_t), end = " | ")

    start = clock()
    standard = KMeans(n_clusters = k, init = data4[:k].copy(), 
                      algorithm = 'full', n_init = 1, 
                      precompute_distances = False).fit(data4)
    std_t = clock() - start
    print("{:.4}".format(std_t), end = " | ")
    
    start = clock()
    Elkan = KMeans(n_clusters = k, init = data4[:k].copy(), 
                   algorithm = 'elkan', n_init = 1, 
                   precompute_distances = False).fit(data4) 
    Elkan_t = clock() - start
    print("{:.4}".format(std_t/Elkan_t))
    
    print(max([dist(clas.clusters[i],yin[0][i]) for i in range(k)]), 
           sum(clas.best == yin[1]))
    
    print(max([dist(clas.clusters[i],my.clusters[-1][i]) for i in range(k)]), 
           sum(clas.best == my.best))
    
    print(max([dist(clas.clusters[i],tur.clusters[-1][i]) for i in range(k)]), 
           sum(clas.best == tur.best))

    print(max([dist(clas.clusters[i],standard.cluster_centers_[i]) for i in range(k)]), 
           sum(clas.best == standard.labels_))
    
    print(max([dist(clas.clusters[i],Elkan.cluster_centers_[i]) for i in range(k)]), 
           sum(clas.best == Elkan.labels_))
    
    
#n, k, d, seed = 16000, 256, 64, 42
#data1 = generate_data(n, d, seed = seed)
#data2 = generate_data(n, d, true_d = 6, true_k = 200, noise = 0.025, seed = seed)
#data3 = load_mnist(n, d, noise = 0.025, seed = seed)
#for data in [data1, data2, data3]:    
#    start = clock()
#    standard = KMeans(n_clusters = k, init = data[:k].copy(), 
#                      algorithm = 'full', n_init = 1, 
#                      precompute_distances = False).fit(data)
#    std_t = clock() - start
#    print("standard: time = {:.3}".format(clock() - start))
#    
#    start = clock()
#    Elkan = KMeans(n_clusters = k, init = data[:k].copy(), 
#                   algorithm = 'elkan', n_init = 1, 
#                   precompute_distances = False).fit(data) 
#    elkan_t = clock() - start
#    print("Elkan   : time = {:.3}".format(clock() - start))
#    print("speedup = {:.3}".format(std_t/elkan_t))    

   
''' 
bugs to fix in other version:
    got it! you need update lbG for group with previous best cluster!
    update lb[x] properly! if it failed lb[x]=Inf
    ?counting of iterations 
    you should carefully maintain invariants when reassign best[x] in farthest
        point strategy
    
'''    
    
    
