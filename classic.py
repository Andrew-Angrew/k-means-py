# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:44:21 2017

@author: Andrew
"""

import numpy as np
from k_means_auxiliary import dist

def classic_k_means(data, k):
    n, d = data.shape
    best = np.zeros(n, int)
    new_best = np.zeros(n, int)
    clasters = data[:k].copy()
    new_clasters = np.zeros((k,d))
    claster_sizes = np.zeros(k, int)
    new_claster_sizes = np.zeros(k, int)
    
    stop = False
    it_num = 0
    while not stop:
        it_num += 1 
        
        # assignment step
        for x in range(n):
            new_best[x] = np.argmin([dist(data[x], c) for c in clasters])
                
        #center update step
        stop = True
        new_claster_sizes = np.zeros(k, int)
        for x in range(n):
            if new_best[x] != best[x]:
                stop = False
            new_claster_sizes[new_best[x]] += 1
        for c in range(k):
            if new_claster_sizes[c] > 0:
                new_clasters[c] = clasters[c] * claster_sizes[c]
            else:
                new_clasters[c] = clasters[c]
        for x in range(n):
            if new_best[x] != best[x]:
                new_clasters[new_best[x]] += data[x]
                if new_claster_sizes[best[x]] > 0:
                    new_clasters[best[x]] -= data[x]
        for c in range(k):
            if new_claster_sizes[c] > 0:
                new_clasters[c] /= new_claster_sizes[c]

        clasters, new_clasters = new_clasters, clasters
        best, new_best = new_best, best
        claster_sizes = new_claster_sizes
    
    print(it_num)
    return (clasters, best)
