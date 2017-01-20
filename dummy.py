# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:33:38 2017

@author: Andrew
"""

import numpy as np
from time import clock

    
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

class dummy:
    def __init__(self, data, k, empty_strat='spare', report=False, steps=Inf):
        assert empty_strat in ['spare', 'farthest point']
        self.empty_strat = empty_strat
        self.report = report
        self.steps = steps
        self.data, self.k = data, k
        self.n, self.d = data.shape
        self.clusters = data[:k].copy()
        if empty_strat == 'farthest point':
            self.farthest_points = np.array(range(k))
        self.old_best = np.ones(self.n, int) * k  #(see else compute_new_clusters)
        self.best = np.zeros(self.n, int)
        self.cluster_sizes = np.zeros(k, int)
        self.ub = np.ones(self.n)
        self.sb = np.zeros(self.n, int)  
        self.stop = False
        self.t = 0
        
    def compute_new_clusters(self, curr_clusters=None):
        if curr_clusters is None:
            curr_clusters = self.clusters
        (n, k, d, ub, best, old_best, sb) = (self.n, self.k, self.d, 
        self.ub, self.best, self.old_best, self.sb)
        new_clusters = np.zeros((k,d))
        new_cluster_sizes = np.zeros(k, int)
        for x in range(n):
            new_cluster_sizes[best[x]] += 1
        if self.empty_strat == 'farthest point':
            if np.min(new_cluster_sizes) == 0:
                for x in range(n):
                    if ub[self.farthest_points[best[x]]] < ub[x]:
                        self.farthest_points[best[x]] = x
                clusters_by_size = new_cluster_sizes.argsort()
                l,r = 0, k-1
                while (new_cluster_sizes[clusters_by_size[r]] == 0 and 
                       new_cluster_sizes[clusters_by_size[l]] > 1 and l < r):
                    best[self.farthest_points[clusters_by_size[l]]] = \
                        clusters_by_size[r]
                    new_cluster_sizes[clusters_by_size[l]] -= 1
                    new_cluster_sizes[clusters_by_size[r]] = 1
                    ub[self.farthest_points[clusters_by_size[l]]] = 0
                    sb[self.farthest_points[clusters_by_size[l]]] = self.t
                    l += 1
                    r -= 1                
        for x in range(n):
            if best[x] != old_best[x]:
                new_clusters[best[x]] += self.data[x]
                if (old_best[x] < k and 
                    new_cluster_sizes[old_best[x]]) > 0:
                    new_clusters[old_best[x]] -= self.data[x]
        for c in range(k):
            if new_cluster_sizes[c] > 0:
                new_clusters[c] += curr_clusters[c] * self.cluster_sizes[c]
                new_clusters[c] /= new_cluster_sizes[c]
            else:
                new_clusters[c] = curr_clusters[c]

        return (new_clusters, new_cluster_sizes)
        
    def fit(self):
        start = clock()
        while self.steps > self.t and not self.stop:
            self.step()
        if self.report:
            print(self.name + ": iter = {}, time = {:.3}".format(
                  self.t, clock() - start))
       
def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))
    
def norm(x):
    return np.sqrt(np.sum(x**2))
    

