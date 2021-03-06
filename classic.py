# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:44:21 2017

@author: Andrew
"""

import numpy as np
from numpy import newaxis
from k_means_auxiliary import dist, Inf, compute_new_clusters

def classic_k_means(data, k, steps = Inf, report = False):
    n, d = data.shape
    best = np.zeros(n, int)
    old_best = np.ones(n, int) * k  #nasty trick (see compute_new_clusters)
    clusters = data[:k].copy()
    cluster_sizes = np.zeros(k, int)
    
    stop = False
    it_num = 0
    dist.count = 0
    while (not stop) and it_num < steps:
        it_num += 1 
        
        # assignment step
        for x in range(n):
            best[x] = np.argmin([dist(data[x], c) for c in clusters])
                
        #center update step
        clusters, cluster_sizes = \
            compute_new_clusters(data, clusters ,old_best, best, cluster_sizes)
        if np.all(best == old_best):
            stop = True
        best, old_best = old_best, best
    
    if report:
        print("classic   : iter = %i, dist. calcs = %i, " 
              % (it_num, dist.count), end = "")
    return (clusters, best)
