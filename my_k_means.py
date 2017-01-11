# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:40:13 2017

@author: Andrew
"""

import numpy as np
from numpy import newaxis
from k_means_auxiliary import dist, infinite_number
Inf = infinite_number()

def my_k_means(data, k):
    assert k % 10 == 0
    n, d = data.shape
    m = int(k/10)
    clasters = data[:k].copy()[newaxis,]
    new_clasters = np.zeros((k,d))
    old_best = np.zeros(n, int)
    best = np.zeros(n, int)
    ub = np.ones(n)
    dc = np.zeros(k)[newaxis,] #смещения кластеров
    dG = np.zeros(m)[newaxis,] #смещения групп
    dmax = np.array([0])       #максимальные смещения кластеров с момента s
    lb = np.zeros(n)
    lbG = np.zeros((n,m))
    sb = np.zeros(n, int)      #момент точности ub
    sG = np.zeros((n,m), int)  #момент точности lbG(x,G)
    sA = np.zeros(n, int)      #момент измерения lb
    parents = np.zeros(k,int)
    children = np.zeros((m,10),int)
    claster_sizes = np.zeros(k, int)
    new_claster_sizes = np.zeros(k, int)
    
    for c in range(k):
        parents[c] = int(c/10)
        children[int(c/10), c % 10] = c

    stop = False
    t = 0
    while not stop:
        #assignment step
        for x in range(n):
            if ub[x] + dc[sb[x], best[x]] > lb[x] - dmax[sA[x]]:
                ub[x] = dist(data[x], clasters[t, best[x]])
                sb[x] = t
                if ub[x] > lb[x] - dmax[sA[x]]:
                    sA[x] = t
                    for G in range(m):
                        if ub[x] > lbG[x,G] - dG[sG[x,G], G]:
                            #update lbG
                            first = (best[x], ub[x])
                            second = (-1, Inf)
                            for c in children[G]:
                                if c != best[x]:
                                    if second[1] > lbG[x,G] - dc[sG[x,G], c]:
                                        dist_x_c = dist(data[x], clasters[t,c])
                                        if second[1] > dist_x_c:
                                            if first[1] > dist_x_c:
                                                second = first
                                                first = (c, dist_x_c)
                                            second = (c, dist_x_c)
                            if first[0] != best[x]:
                                best[x] = first[0]
                                ub[x] = first[1]
                            lbG[x,G] = second[1]
                            sG[x,G] = t
                            lb[x] = min(lb[x], second[1])
                        else:
                            lb[x] = min(lb[x], lbG[x,G] - dG[sG[x,G], G])
        

        #center update step
        stop = True
        new_claster_sizes = np.zeros(k, int)
        for x in range(n):
            if old_best[x] != best[x]:
                stop = False
            new_claster_sizes[best[x]] += 1
        new_clasters = np.zeros((k,d))
        for c in range(k):
            if new_claster_sizes[c] > 0:
                new_clasters[c] = clasters[t, c] * claster_sizes[c]
            else:
                new_clasters[c] = clasters[t, c]
        for x in range(n):
            if old_best[x] != best[x]:
                new_clasters[best[x]] += data[x]
                if new_claster_sizes[old_best[x]] > 0:
                    new_clasters[old_best[x]] -= data[x]
        for c in range(k):
            if new_claster_sizes[c] > 0:
                new_clasters[c] /= new_claster_sizes[c]

        clasters = np.vstack([clasters, new_clasters[newaxis, ]])
        old_best = best.copy()
        claster_sizes = new_claster_sizes
        
        t += 1 
        dc = np.array([[dist(clasters[s,c], clasters[t,c]) 
                            for c in range(k)] for s in range(t)])
        dG = np.transpose(np.array( [ dc[:,children[G]].max(axis = 1) 
                                        for G in range(m) ] ))
        dmax = dG.max(axis = 1)

    print(t)
    return (clasters[t], best)


