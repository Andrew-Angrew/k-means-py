import numpy as np
from dummy import dist, Inf, dummy
from make_groups import make_clustered_groups, make_alphabet_groups

class yinyang_k_means(dummy):
    def __init__(self, data, k, empty_strat='spare', groups_strat='alphabet'):
        self.name = "yinyang"            
        dummy.__init__(self, data, k, empty_strat)
        self.m = (k - 1)//10 + 1
        if groups_strat == 'alphabet':
            self.children, self.parents = \
            make_alphabet_groups(self.clusters, self.m)
        elif groups_strat == 'clustered':
            self.children, self.parents = \
            make_clustered_groups(self.clusters, self.m)
        self.m = len(self.children)
        self.dc = np.zeros(k) #смещения кластеров
        self.dG = np.zeros(self.m) #смещения групп
        self.lb = np.zeros(self.n)
        self.lbG = np.zeros((self.n,self.m))
    
    def update_lbG(self, x, G):
        (m, children, best, k, ub, lbG, 
        dc, t, dG, lb, clusters, n, parents, data) = (
        self.m, self.children, self.best, 
        self.k, self.ub, self.lbG, 
        self.dc, self.t, self.dG, self.lb, self.clusters, 
        self.n, self.parents, self.data)        
        
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
        

    def reassign_points(self):
        (m, children, best, k, ub, lbG, 
        dc, t, dG, lb, clusters, n, parents, data) = (
        self.m, self.children, self.best, 
        self.k, self.ub, self.lbG, 
        self.dc, self.t, self.dG, self.lb, self.clusters, 
        self.n, self.parents, self.data)  
        
        for x in range(n):
            if ub[x] > lb[x]:
                ub[x] = dist(data[x], clusters[best[x]])
                if ub[x] > lb[x]:
                    lb[x] = Inf
                    best_G = parents[best[x]]
                    for i in range(m):
                        #begin with best_G to update lbG[x,best_G] properly
                        #if x will change best[x]
                        G = (best_G + i) % m 
                        if ub[x] > lbG[x,G] - dG[G]:
                            self.update_lbG(x, G)
                        else:
                            lbG[x,G] -= dG[G]
                            lb[x] = min(lb[x], lbG[x,G] - dG[G])
                else:
                    for G in range(m):
                        lbG[x,G] -= dG[G]
            else:
                for G in range(m):
                    lbG[x,G] -= dG[G]


    def update_centers(self):
        (m, children, best, k, ub, lbG, 
        dc, t, dG, lb, clusters, n, parents, data) = (
        self.m, self.children, self.best, 
        self.k, self.ub, self.lbG, 
        self.dc, self.t, self.dG, self.lb, self.clusters, 
        self.n, self.parents, self.data)  
        
        new_clusters, self.cluster_sizes = self.compute_new_clusters()
        self.old_best = best.copy()
        
        for c in range(k):
            dc[c] = dist(clusters[c], new_clusters[c])
        self.clusters = new_clusters
        for G in range(m):
            dG[G] = max([dc[c] for c in children[G]])
        delta = max(dG)
        for x in range(n):
            ub[x] += dc[best[x]]
            lb[x] -= delta            
        
        
        