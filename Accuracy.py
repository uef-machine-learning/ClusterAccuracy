#!/usr/bin/python3
# Implemented by Martin Dautriche and Sami Sieranoja
# version 0.2 (2022-09-16)

# Example commands:
# python3 Accuracy.py -p -i data/s1-label.pa data/s1_clustering_result.pa  
# python3 Accuracy.py -m -i data/s1-label.pa data/s1_clustering_result.pa  

import numpy as np
import argparse
import time
import random

from collections import defaultdict

import pdb
from scipy.optimize import linear_sum_assignment
from ReadFilePointsPa import readFilePa
import sklearn
from sklearn.metrics import accuracy_score



# Y: true cluster label vector; preY: predicted cluster label vector
# Y: partition label array
# preY: partition label array
def accuracy(Y, predY, use_matching=True):
	res = mapping_labels(Y, predY, use_matching)
	ACC = (find_matlab(Y,res)/len(Y))
	return ACC

# Equivalent to find function in Matlab
# Return how many time Y[i] is equals with res[i]
def find_matlab(Y, res):
	k = 0
	for i in range(len(Y)):
		if Y[i] == res[i]:
			k = k+1
	return k

# Matching labels using contingency matrix
def matching_Labels(contingencyMatrix):
	match_labels = []
	for i in range(len(contingencyMatrix)):
		match_labels.append(contingencyMatrix[i].tolist().index(max(contingencyMatrix[i])))
	return match_labels
	
def matching_Labels2(contingencyMatrix):
	match_labels = []
	i=1
	for i in contingencyMatrix:
		v = list(contingencyMatrix[i].values())
		k = list(contingencyMatrix[i].keys())
		maxind = k[v.index(max(v))]
		match_labels.append(maxind)
	return match_labels

def ContingencyMatrix(La,Lb,kA,kB,use_matching):
	if use_matching: # Mapping, using hash table
		contg = defaultdict(lambda: 0)
		for y in range(kA):
			contg[y] = defaultdict(lambda: 0)
	else: # Pairing
		wh = max([kA,kB])
		contg = [[0 for x in range(wh)] for y in range(wh)]
	for i in range(len(La)):
		contg[La[i]-1][Lb[i]-1] += 1   
	return contg


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


# Mapping labels : Pairing or Matching
# L1 and L2 -> Labels List
def mapping_labels(L1,L2,use_matching):
	if len(L1) != len(L2):
		print("Error !!!")
		
	Label1 = np.unique(L1)
	nClass1 = len(Label1)
	Label2 = np.unique(L2)
	nClass2 = len(Label2)

	if nClass1 > nClass2:
		to_add = [*range(max(Label2+1),max(Label1+1))]
		Label2 = [*Label2, *to_add]
		 
	elif nClass2 > nClass1:
		to_add = [*range(max(Label1+1),max(Label2+1))]
		Label1 = [*Label1, *to_add]

	if use_matching:
		cg2 = ContingencyMatrix(L2,L1,nClass2,nClass1,use_matching)
		col_ind = matching_Labels2(cg2)
	else:
		# Pairing using Hungarian Algo
		cg1 = ContingencyMatrix(L2,L1,nClass2,nClass1,use_matching)
		contingencyMatrix = np.array(cg1)
		_, col_ind = linear_sum_assignment(-contingencyMatrix) #Hungarian algorithm (Variant Jonker–Volgenant algorithm)
	newL2 = np.zeros(len(L2))
	
	for i in range(nClass2):
		newL2[L2 == Label2[i]] = Label1[col_ind[i]]
	return newL2

if __name__ == "__main__":
	# timecomplexity_benchmark(False)
	# timecomplexity_benchmark(True)
	# exit()

	argparser = argparse.ArgumentParser()

	argparser.add_argument("-i", "--inputFile", required=True,
						   help="Needs two files as input. First one is considered the ground truth.", nargs="*", type=str)

	group = argparser.add_mutually_exclusive_group(required=True)
	group.add_argument("-m", "--Matching", required=False,
						   help="Matching labels (1 => to many mapping)", action='store_true')
	group.add_argument("-p", "--Pairing", required=False,
						   help="Pairing labels with the Hungarian Algorithm (1:1 mapping)", action='store_true')

	args = vars(argparser.parse_args())

	inputFilename = args["inputFile"]
	if len(args["inputFile"]) != 2:
		raise argparser.ArgumentError(None, 'Nb input file must be 2')
	
	# use_matching is True with (-m) matching option
	# use_matching is False with (-p) pairing option 
	use_matching = False
	if args["Matching"]:
		use_matching = True
	elif args["Pairing"]:
		use_matching = False

	if use_matching:
		ACCbtoa = accuracy(readFilePa(inputFilename[0]), readFilePa(inputFilename[1]), use_matching=use_matching)
		ACCatob = accuracy(readFilePa(inputFilename[1]), readFilePa(inputFilename[0]), use_matching=use_matching)
		accmin = min([ACCatob, ACCbtoa])
		accmean = np.mean([ACCatob, ACCbtoa])
		print("ACC with Matching labels = %f\n    (B=>A):%f (A=>B):%f min:%f mean:%f " %(accmin, ACCbtoa, ACCatob, accmin, accmean))
	else:
		ACC = accuracy(readFilePa(inputFilename[0]), readFilePa(inputFilename[1]), use_matching=use_matching)
		print("ACC with Pairing (Hungarian) labels = "+str(ACC))


