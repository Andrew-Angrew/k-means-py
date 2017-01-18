## Algorithms:
my realizations in python 3.5:

classic - classic k-means

yinyang - algorithm from the paper "Yinyang K-means: A Drop-in Replacement of the Classic K-means with Consistent Speedup", Yufei Ding et al., ICML, 2015.

my_k_means - my improvement of yinyang

turbo - variation of my_k_means, using O(nk) additional memory

realizations from sklearn: standard, Elcan

##Data-sets

n, k, d, seed = 16000, 256, 64, 42

1 Random points

2 Points cnterd around true_k random clusters in space of dimension true_d and imbedded into d-dimensional space via random matrix

3 mnist data-set with dropped coordinates of lesser variation with 0.025*normal noise.

##Results
###data1
algorithm | iter | distance calcultions | time (s) | speedup over classic
------------ | ------------- | ---------- | ------------- | ------------- |
classic   | 28 | 118784000 | 4679.550236 |
yinyang   | 28 | 43944931 | 1832.094502 | 2.554
my_k_means| 28 | 31669670 | 1227.240892 | 3.813
turbo     | 28 | 22810904 | 971.634182 | 4.816
standard  |           |          | 15.5
Elkan     |           |          | 6.23        | 2.48

###data2
algorithm | iter | distance calcultions | time (s) | speedup over classic
------------ | ------------- | ---------- | ------------- | ------------- |
classic   | 13 | 57344000 | 1769.680204 |
yinyang   | 13 |  6218541 | 251.735244 | 7.029
my_k_means| 13 |  5228257 | 219.911093 | 8.047
turbo     | 13 | 4647194 | 206.548091 | 8.567
standard  |           |          | 7.28 |
Elkan     |           |          |  2.26 | 3.23

###data3 (mnist)
algorithm | iter | distance calcultions | time (s) | speedup over classic
------------ | ------------- | ---------- | ------------- | ------------- |
classic   | 47 | 196608000 | 6064.409036
yinyang   | 47 | 19867172 | 835.799325 | 7.255
my_k_means| 47 | 13495695 | 625.046024 | 9.702
turbo     | 47 | 9869174 | 519.508485 | 11.673
standard  |           |          | 24.3 |
Elkan     |           |          | 8.31 | 2.92
