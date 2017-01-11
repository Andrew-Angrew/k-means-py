# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:02:18 2017

@author: Andrew
"""


import numpy as np
from time import clock
from k_means_auxiliary import dist, generate_data, load_mnist
from classic import classic_k_means
from Yinyang import yinyang_k_means
from my_k_means import my_k_means

n, k, d = 10000, 100, 100
#data = generate_data(n, d, seed = 44)
#data = generate_data(n, d, true_d = 20, true_k = n, noise = 0.1, seed = 43)
data = load_mnist(n, d)

print("start")
start = clock()
ans_c = classic_k_means(data, k)
print("time =%f." % (clock() - start))

start = clock()
ans_y = yinyang_k_means(data, k)
print("time = %f." % (clock() - start))

start = clock()
ans_m = my_k_means(data, k)
print("time = %f." % (clock() - start))

print(max([dist(ans_c[0][i],ans_y[0][i]) for i in range(k)]), 
       sum(ans_c[1] == ans_y[1]))

print(max([dist(ans_c[0][i],ans_m[0][i]) for i in range(k)]), 
       sum(ans_c[1] == ans_y[1]))
