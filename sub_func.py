# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 17:15:57 2017

@author: Andrew
"""

from classic import classic_k_means
import numpy as np

def make_groups(clusters, m, steps = 3):
    k, d = clusters.shape
    parents = classic_k_means(clusters, m, steps = steps)[1]
    group_sizes = np.zeros(m)
    for G in parents:
        group_sizes[G] += 1
    old_labels = group_sizes.argsort()[::-1]
    if group_sizes[old_labels[0]] > 3 * k / m:
        print("big groups!")
    new_m = m
    while group_sizes[old_labels[new_m-1]] == 0:
        new_m -= 1
    new_labels = old_labels.argsort()
    parents = np.array([new_labels[G] for G in parents])
    children = list([[] for G in range(new_m)])
    for c,G in zip(range(k), parents):
        children[G].append(c)  
    return (children, parents)


