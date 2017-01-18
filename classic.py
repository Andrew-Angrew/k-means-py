# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:44:21 2017

@author: Andrew
"""

import numpy as np
from numpy import newaxis
from dummy import dist, Inf, dummy

class classic_k_means(dummy):
    def __init__(self, data, k, empty_strat='spare', report=False, steps=Inf):
        dummy.__init__(self, data, k, empty_strat, report, steps)
        self.name = "classic   "
        
    def step(self):        
        # assignment step
        for x in range(self.n):
            self.ub[x] = Inf
            for c in range(self.k):
                dist_x_c = dist(self.data[x], self.clusters[c])
                if self.ub[x] > dist_x_c:
                    self.ub[x], self.best[x] = dist_x_c, c
        #center update step
        if np.all(self.best == self.old_best):
            self.stop = True
        else:
            self.clusters, self.cluster_sizes = self.compute_new_clusters()
            self.old_best = self.best.copy()
            self.t += 1 
