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

n, k, d, seed = 1000, 32, 32, 42
data1 = generate_data(n, d, seed = seed)
data2 = generate_data(n, d, true_d = 6, true_k = 200, noise = 0.025, seed = seed)
data3 = load_mnist(n, d, noise = 0.025, seed = seed)
for data in [data1, data2, data3]:
    
    start = clock()
    ans_c = classic_k_means(data, k, report = True)
    print("time = %f." % (clock() - start))

    start = clock()
    ans_y = yinyang_k_means(data, k)
    print("time = %f." % (clock() - start))
  
    start = clock()
    ans_m = my_k_means(data, k)
    print("time = %f." % (clock() - start))

     
    print(max([dist(ans_c[0][i],ans_y[0][i]) for i in range(k)]), 
           sum(ans_c[1] == ans_y[1]))
    
    print(max([dist(ans_c[0][i],ans_m[0][i]) for i in range(k)]), 
           sum(ans_c[1] == ans_m[1]))
    
    print(max([dist(ans_y[0][i],ans_m[0][i]) for i in range(k)]), 
           sum(ans_y[1] == ans_m[1]))

