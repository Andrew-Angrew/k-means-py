# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:40:13 2017

@author: Andrew
"""

import numpy as np
from numpy import newaxis
from copy import deepcopy
from k_means_auxiliary import dist, Inf, compute_new_clusters
from sub_func import make_groups

def my_k_means(data, k):
    n, d = data.shape
    m = int(np.ceil(k/10))
    clusters = [data[:k].copy()]
    old_best = np.ones(n, int) * k  #nasty trick (see compute_new_clusters)
    best = np.zeros(n, int)
#    parents = np.array([int(c/10) for c in range(k)])
#    children = [[[10 * i + j for j in range(10)] for i in range(m)]]
    children, parents = make_groups(clusters[0], m);
    m = len(children)
    children = [children]
    ub = np.ones(n)
    dc = np.zeros(k)[newaxis,] #смещения кластеров
    dG = np.zeros(m)[newaxis,] #смещения групп
    dmax = np.array([0])       #максимальные смещения кластеров с момента s
    lb = np.zeros(n)
    lbG = np.zeros((n,m))
    lbc = np.zeros((n,k))
    sb = np.zeros(n, int)      #момент точности ub
    sG = np.zeros((n,m), int)  #момент точности lbG(x,G)
    sc = np.zeros((n,k), int)  #момент точности lbc(x,c)
    sA = np.zeros(n, int)      #момент измерения lb
    cluster_sizes = np.zeros(k, int)
    
    stop = False
    t = 0
    dist.count = 0
    
    while not stop:
        #assignment step
        for x in range(n):
            if ub[x] + dc[sb[x], best[x]] > lb[x] - dmax[sA[x]]:
                ub[x] = dist(data[x], clusters[t][best[x]])
                lbc[x, best[x]] = ub[x]
                sc[x, best[x]] = t
                sb[x] = t
                if ub[x] > lb[x] - dmax[sA[x]]:
                    lb[x] = Inf
                    sA[x] = t
                    G_best = parents[best[x]]
                    for i in range(m):
                        G = (G_best + i) % m
                        if ub[x] > lbG[x,G] - dG[sG[x,G], G]:
                            #update lbG
                            first = (best[x], ub[x])
                            second = (-1, Inf)
                            for c in children[sG[x,G]][G]:
                                if c != best[x]:
                                    if sc[x,c] != t:  #to avoid index error 
                                        delta_c = dc[sc[x,c], c]
                                    else:
                                        delta_c = 0
                                    if second[1] > lbc[x,c] - delta_c:
                                        dist_x_c = dist(data[x], clusters[t][c])
                                        lbc[x,c] = dist_x_c
                                        sc[x,c] = t
                                        if second[1] > dist_x_c:
                                            if first[1] > dist_x_c:
                                                if parents[first[0]] == G:
                                                    second = first
                                                first = (c, dist_x_c)
                                            second = (c, dist_x_c)
                            if first[0] != best[x]:
                                lbG[x,parents[best[x]]] = ub[x]
                                sG[x,parents[best[x]]] = t
                                best[x] = first[0]
                                ub[x] = first[1]
                            lbG[x,G] = second[1]
                            sG[x,G] = t
                            lb[x] = min(lb[x], second[1])
                        else:
                            lb[x] = min(lb[x], lbG[x,G] - dG[sG[x,G], G])
        

        #center update step
        if np.all(best == old_best):
            stop = True
        else:
            new_clusters, cluster_sizes = compute_new_clusters(data, 
                                    clusters[t] ,old_best, best, cluster_sizes)
    
            old_best = best.copy()
            clusters.append(new_clusters)  
            
            t += 1 
            dc = np.array([[dist(clusters[s][c], clusters[t][c]) 
                                for c in range(k)] for s in range(t)])
            dG = np.transpose(np.array( [ dc[:,children[0][G]].max(axis = 1) 
                                            for G in range(m) ] ))
            if t > 1:
                children.append(deepcopy(children[-1]))
            for s in range(t):
                for G in range(m):
                    children[s][G].sort(key = lambda c:dc[s,c], reverse = True)
            dmax = dG.max(axis = 1)
        
    #print([len(G) for G in children[0]])
    print("my_k_means: iter = %i, dist. calcs = %i, " % (t, dist.count), 
          end = "")
    return (clusters[t], best)


