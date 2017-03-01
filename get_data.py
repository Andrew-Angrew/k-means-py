import numpy as np
from numpy.random import random, randn
import csv

def generate(n, d, seed=42, true_k=None, true_d=None, 
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
    
def mnist(n, d, noise=0, seed=42):
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

def kegg_net(n, noise=0,seed=42):
    np.random.seed(seed)
    data = []
    with open("C:/Andrew/data/Reaction Network (Undirected).data") as f:
        csv_f = csv.reader(f)
        t = 0
        for row in csv_f:
            if t > 0:
                data.append([make_num(w) for w in row[1:]])
            t += 1
            if t > n:
                break
    data -= np.mean(data, axis = 0)
    data /= np.std(data, axis = 0)
    return data + noise * randn(*data.shape)


