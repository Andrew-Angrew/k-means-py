# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 17:15:57 2017

@author: Andrew
"""

from classic import classic_k_means
import numpy as np

def make_clustered_groups(clusters, m, steps=3):
    k, d = clusters.shape
    assert k >= m
    cl = classic_k_means(clusters, m, empty_strat = 'farthest point',
                              steps = steps)
    cl.fit()
    parents = cl.best
    group_sizes = np.zeros(m)
    for G in parents:
        group_sizes[G] += 1
    old_labels = group_sizes.argsort()[::-1]
    new_m = m
    while group_sizes[old_labels[new_m-1]] == 0:
        new_m -= 1
    new_labels = old_labels.argsort()
    parents = np.array([new_labels[G] for G in parents])
    children = list([[] for G in range(new_m)])
    for c,G in zip(range(k), parents):
        children[G].append(c)  
    return (children, parents)

def make_alphabet_groups(clusters, m):
    k, d = clusters.shape
    assert k >= m
    min_size = k // m
    big = k % m
    c = 0
    children = []
    parents = []
    for G in range(m):
        size = min_size + int(G < big)
        children.append(list(range(c,c + size)))
        parents += [G] * size
        c += size
    return (children, parents)
    

