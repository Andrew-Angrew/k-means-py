# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:40:13 2017

@author: Andrew
"""

import numpy as np
from k_means_auxiliary import dist, Inf, compute_new_clusters
from sub_func import make_groups

def yinyang_k_means(data, k):
    n, d = data.shape
    m = int(np.ceil(k/10))
    clusters = data[:k].copy()
    old_best = np.ones(n, int) * k  #nasty trick (see compute_new_clusters)
    best = np.zeros(n, int)
#    parents = np.array([int(c/10) for c in range(k)])
#    children = [[10 * i + j for j in range(10)] for i in range(m)]
    children, parents = make_groups(clusters, m);
    m = len(children)
    ub = np.ones(n)
    dc = np.zeros(k)
    dG = np.zeros(m)
    lb = np.zeros(n)
    lbG = np.zeros((n,m))
    cluster_sizes = np.zeros(k, int)

    stop = False
    it_num = 0
    dist.count = 0
    while not stop:         
        #assignment step
        for x in range(n):
            if ub[x] > lb[x]:
                ub[x] = dist(data[x], clusters[best[x]])
                if ub[x] > lb[x]:
                    lb[x] = Inf
                    G_best = parents[best[x]]
                    for i in range(m):
                        G = (G_best + i) % m
                        if ub[x] > lbG[x,G] - dG[G]:
                            #update lbG
                            first = (best[x], ub[x])
                            second = (-1, Inf)
                            for c in children[G]:
                                if c != best[x]:
                                    if second[1] > lbG[x,G] - dc[c]:
                                        dist_x_c = dist(data[x], clusters[c])
                                        if second[1] > dist_x_c:
                                            if first[1] > dist_x_c:
                                                if parents[first[0]] == G:
                                                    second = first
                                                first = (c, dist_x_c)
                                            second = (c, dist_x_c)
                            if first[0] != best[x]:
                                lbG[x, parents[best[x]]] = ub[x]
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
        if np.all(best == old_best):
            stop = True
        else:
            it_num += 1
            new_clusters, cluster_sizes = \
                compute_new_clusters(data, clusters ,old_best, best, cluster_sizes)
            
            old_best = best.copy()
            
            for c in range(k):
                if cluster_sizes[c] == 0:
                    dc[c] = 0
                else:
                    dc[c] = dist(clusters[c], new_clusters[c])
            clusters = new_clusters
            for G in range(m):
                dG[G] = max([dc[c] for c in children[G]])
            delta = max(dG)
            for x in range(n):
                ub[x] += dc[best[x]]
                lb[x] -= delta     

    print("yinyang   : iter = %i, dist. calcs = %i, " % (it_num, dist.count), end = "")
    return (clusters, best)


