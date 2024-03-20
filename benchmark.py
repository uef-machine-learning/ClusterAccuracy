#!/usr/bin/python3
import sys
from Accuracy import accuracy 
import numpy as np
import time
import random

# from scipy.optimize import linear_sum_assignment
# from ReadFilePointsPa import readFilePa
# import sklearn
from sklearn.metrics import accuracy_score

def timecomplexity_benchmark(use_matching):
	np.random.seed(1030920293)
	random.seed(1030920293)
	N=20000
	global debugk
	for N in [10000,20000,30000,40000,50000]:
		for i in range(1,int(N/100)):
			debugk=i*100
			timecomplexity_benchmark_k(i*100,N,use_matching)

def timecomplexity_benchmark_k(k,N,use_matching):
	kA = kB = k
	La=np.random.randint(1,kA+1,N)
	Lb=np.random.randint(1,kB+1,N)
	
	# To make sure that each label between 1..k appears at least once.
	for i in range(1,k+1):
		La[i] = i
		Lb[i] = i

	start_time = time.time()
	if use_matching:
		accuracy(La, Lb, use_matching=use_matching)
		accuracy(Lb, La, use_matching=use_matching)
	else:
		accuracy(La, Lb, use_matching=use_matching)
	ttime = time.time() - start_time
	print("N=%d k=%d time=%f" % (N,k,ttime),flush=True)


if sys.argv[1] == "matching":
	timecomplexity_benchmark(True)
if sys.argv[1] == "pairing":
	timecomplexity_benchmark(False)


