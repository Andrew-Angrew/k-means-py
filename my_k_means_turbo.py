# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:40:13 2017

@author: Andrew
"""

import numpy as np
from numpy import newaxis
from copy import deepcopy
from dummy import dist, Inf, dummy
from make_groups import*
from time import clock

class turbo(dummy):
    def __init__(self, data, k, empty_strat='spare', report=False, 
                 groups_strat='alphabet'):
        self.name = "turbo     "            
        dummy.__init__(self, data, k, empty_strat, report)
        self.clusters = [self.clusters]
        self.m = (k - 1)//10 + 1
        if groups_strat == 'alphabet':
            self.children, self.parents = \
            make_alphabet_groups(self.clusters[0], self.m)
        elif groups_strat == 'clustered':
            self.children, self.parents = \
            make_clustered_groups(self.clusters[0], self.m)
        self.m = len(self.children)
        self.children = [self.children]
        self.dc = np.zeros(k)[newaxis,] #смещения кластеров
        self.dG = np.zeros(self.m)[newaxis,] #смещения групп
        self.dmax = np.array([0])       #смещение всех кластеров с момента s
        self.lb = np.zeros(self.n)
        self.lbG = np.zeros((self.n,self.m))
        self.lbc = np.zeros((self.n,k))
        self.sG = np.zeros((self.n,self.m), int)  #момент точности lbG(x,G)
        self.sc = np.zeros((self.n,k), int)  #момент точности lbc(x,c)
        self.sA = np.zeros(self.n, int)      #момент измерения lb
        
    def step(self):        
        #i'm sorry, but i need do so)
        (m, sb, children, best, k, ub, lbG, dmax, lbc, sc, 
        dc, t, sG, dG, sA, lb, clusters, n, parents, data) = (
        self.m, self.sb, self.children, self.best, 
        self.k, self.ub, self.lbG, self.dmax, self.lbc, self.sc, 
        self.dc, self.t, self.sG, self.dG, 
        self.sA, self.lb, self.clusters, self.n, self.parents, self.data)
        
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
                    best_G = parents[best[x]]
                    for i in range(m):
                        #begin with best_G to update lbG[x,best_G] properly
                        #if x will change best[x]
                        G = (best_G + i) % m  
                        if ub[x] > lbG[x,G] - dG[sG[x,G], G]:
                            self.update_lbG(x,G)                            
                        else:
                            lb[x] = min(lb[x], lbG[x,G] - dG[sG[x,G], G])
                
        #center update step
        if np.all(best == self.old_best):
            self.stop = True
        else:
            t += 1
            self.t = t
            new_clusters, self.cluster_sizes = \
                self.compute_new_clusters(curr_clusters = clusters[t-1])            
            self.old_best = best.copy()
            clusters.append(new_clusters)  
            
            del dc, dG  #is it really spares something?
            self.dc = np.array([[dist(clusters[s][c], clusters[t][c]) 
                                for c in range(k)] for s in range(t)])
            self.dc = np.vstack((self.dc, np.zeros(k)))
            self.dG = np.transpose(np.array(
                [self.dc[:, children[0][G]].max(axis = 1) for G in range(m)]))
            if t > 1:
                children.append(deepcopy(children[-1]))
            for s in range(t):
                for G in range(m):
                    children[s][G].sort(key = lambda c:self.dc[s,c],
                                        reverse = True)
            self.dmax = self.dG.max(axis = 1)
            
    def update_lbG(self, x,G):
        (children, best, ub, lbG, sc, lbc, 
        dc, t, sG, lb, clusters, parents, data) = (
        self.children, self.best, 
        self.ub, self.lbG, self.sc, self.lbc, 
        self.dc, self.t, self.sG, 
        self.lb, self.clusters, self.parents, self.data)
        
        first = (best[x], ub[x])
        second = (-1, Inf)
        for c in children[sG[x,G]][G]:
            if c != best[x]:  #hate this if
                if second[1] > lbc[x,c] - dc[sc[x,c], c]:
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
            lbG[x, parents[best[x]]] = ub[x]
            sG[x, parents[best[x]]] = t
            best[x] = first[0]
            ub[x] = first[1]            
        lbG[x,G] = second[1]
        sG[x,G] = t
        lb[x] = min(lb[x], second[1])
        
    def run(self):
        dist.count = 0
        start = clock()
        while not self.stop:
            self.step()
        if self.report:
            print(self.name + ": iter = %i, dist. calcs = %i, time = %f.3" % 
              (self.t, dist.count, clock() - start))
        return (self.clusters[self.t], self.best)


