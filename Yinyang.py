# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:40:13 2017

@author: Andrew
"""

import numpy as np
from k_means_auxiliary import dist, infinite_number
Inf = infinite_number()

def yinyang_k_means(data, k):
    assert k % 10 == 0
    n, d = data.shape
    m = int(k/10)
    clasters = data[:k].copy()
    new_clasters = np.zeros((k,d))
    old_best = np.zeros(n, int)
    best = np.zeros(n, int)
    ub = np.ones(n)
    dc = np.zeros(k)
    dG = np.zeros(m)
    lb = np.zeros(n)
    lbG = np.zeros((n,m))
    parents = np.zeros(k,int)
    children = np.zeros((m,10),int)
    claster_sizes = np.zeros(k, int)
    new_claster_sizes = np.zeros(k, int)
    
    for c in range(k):
        parents[c] = int(c/10)
        children[int(c/10), c % 10] = c

    stop = False
    it_num = 0
    while not stop:  
        it_num += 1
        
        #assignment step
        for x in range(n):
            if ub[x] > lb[x]:
                ub[x] = dist(data[x], clasters[best[x]])
                if ub[x] > lb[x]:
                    for G in range(m):
                        if ub[x] > lbG[x,G] - dG[G]:
                            #update lbG
                            first = (best[x], ub[x])
                            second = (-1, Inf)
                            for c in children[G]:
                                if c != best[x]:
                                    if second[1] > lbG[x,G] - dc[c]:
                                        dist_x_c = dist(data[x], clasters[c])
                                        if second[1] > dist_x_c:
                                            if first[1] > dist_x_c:
                                                second = first
                                                first = (c, dist_x_c)
                                            second = (c, dist_x_c)
                            if first[0] != best[x]:
                                best[x] = first[0]
                                ub[x] = first[1]
                            lbG[x,G] = second[1]
                            lb[x] = min(lb[x], second[1])
                        else:
                            lbG[x,G] -= dG[G]
                            lb[x] = min(lb[x], lbG[x,G] - dG[G])
                else:
                    for G in range(m):
                        lbG[x,G] -= dG[G]
            else:
                for G in range(m):
                    lbG[x,G] -= dG[G]
        

        #center update step
        stop = True
        new_claster_sizes = np.zeros(k, int)
        for x in range(n):
            if old_best[x] != best[x]:
                stop = False
            new_claster_sizes[best[x]] += 1
        for c in range(k):
            if new_claster_sizes[c] > 0:
                new_clasters[c] = clasters[c] * claster_sizes[c]
            else:
                new_clasters[c] = clasters[c]
                dc[c] = 0
        for x in range(n):
            if old_best[x] != best[x]:
                new_clasters[best[x]] += data[x]
                if new_claster_sizes[old_best[x]] > 0:
                    new_clasters[old_best[x]] -= data[x]
        for c in range(k):
            if new_claster_sizes[c] > 0:
                new_clasters[c] /= new_claster_sizes[c]
                dc[c] = dist(new_clasters[c], clasters[c])

        clasters, new_clasters = new_clasters, clasters
        old_best = best.copy()
        claster_sizes = new_claster_sizes
        
        for G in range(m):
            dG[G] = max([dc[c] for c in children[G]])
        delta = max(dG)
        for x in range(n):
            ub[x] += dc[best[x]]
            lb[x] -= delta     

    print(it_num)
    return (clasters, best)


