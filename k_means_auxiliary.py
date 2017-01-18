# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:33:38 2017

@author: Andrew
"""

import numpy as np
from math import fsum
from numpy.random import random, randn
import csv
from itertools import product

class infinite_number(object):
    
    def __init__(self):
        self.forced_val = 100500.
        pass
    
    def __gt__(self, value):
        return True
        
    def __ge__(self, value):
        return True
        
    def __lt__(self, value):
        return False
        
    def __le__(self, value):
        return False
        
    def __float__(self):
        return self.forced_val
        
Inf = infinite_number()

def compute_new_clusters(data, clusters, old_best, best, cluster_sizes):
    n = len(best)
    k,d = clusters.shape
    new_clusters = np.zeros((k,d))
    new_cluster_sizes = np.zeros(k, int)
    #print(best)
    for x in range(n):
        new_cluster_sizes[best[x]] += 1
    for c in range(k):
        if new_cluster_sizes[c] > 0:
            new_clusters[c] = clusters[c] * cluster_sizes[c]
        else:
            new_clusters[c] = clusters[c]
    for x in range(n):
        if best[x] != old_best[x]:
            new_clusters[best[x]] += data[x]
            if old_best[x] < k and new_cluster_sizes[old_best[x]] > 0:
                new_clusters[old_best[x]] -= data[x]
    for c in range(k):
        if new_cluster_sizes[c] > 0:
            new_clusters[c] /= new_cluster_sizes[c]
    return (new_clusters, new_cluster_sizes)

def generate_data(n, d, seed = 42, true_k = None, true_d = None, 
                  noise = 0, cluster_sparsity = 1/10):
    np.random.seed(seed)
    if true_k == None:
        true_k = n
    if true_d == None:
        true_d = d
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
    
def load_mnist(n, d, noise = 0, seed = 42):
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
    for i,j in product(range(n),range(784)):
        if type(data[i,j]) != np.float64:
            print("auchting! ",i,j,data[i,j], type(data[i,j]))
    v1 = data.var(axis = 0)
    v2 = v1.copy()
    v1.sort()
    indices = [i for i in range(784) if v2[i] >= v1[-d]]
    data = data[:,indices]
    return data + noise * randn(*data.shape)
       
def dist(x,y):
    dist.count += 1
    return np.sqrt(np.sum((x-y)**2))
    
def norm(x):
    return np.sqrt(np.sum(x**2))
    
    