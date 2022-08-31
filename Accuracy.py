import numpy as np
import argparse
import pdb
from scipy.optimize import linear_sum_assignment
from ReadFilePointsPa import readFilePa
from Kmeans_Functions import KmeansDataAccuracy
import sklearn
from sklearn.metrics import accuracy_score

# Implemented by Martin Dautriche and Sami Sieranoja

# Y: true cluster label vector; preY: predicted cluster label vector
# Y: partition label array
# preY: partition label array
def accuracy(Y, predY, mapping_method=True):
    res = mapping_labels(Y, predY, mapping_method)
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
    
# ContingencyMatrix(La,Lb,N,kA,kB,mappingMethod):
# IF mappingMethod == "pairing"
  # contg = matrixOfZeros(MAX{kA,kB},MAX{kA,kB})
# ELSE
  # contg = matrixOfZeros(kA,kB)
# FOR i=1..N
  # # Number of points that belong to both clusters La[i] and Lb[i]
  # contg[La[i]][Lb[i]] += 1   
  
def ContingencyMatrix(La,Lb,kA,kB,mapping_method):
	if mapping_method: # Mapping
		contg = [[0 for x in range(kB)] for y in range(kA)]
	else: # Pairing
		wh = max([kA,kB])
		contg = [[0 for x in range(wh)] for y in range(wh)]
	for i in range(len(La)):
		contg[La[i]-1][Lb[i]-1] += 1   
	return contg

# Mapping labels : Pairing or Matching
# L1 and L2 -> Labels List
def mapping_labels(L1,L2,mapping_method):
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

    contingencyMatrix = sklearn.metrics.cluster.contingency_matrix(L2,L1)
    cg2 = ContingencyMatrix(L2,L1,nClass2,nClass1,mapping_method)
    
    for i in range(0,nClass1):
        for j in range(0,nClass2):
            if int(cg2[j][i]) != int(contingencyMatrix[j][i]):
                print("ERROR a=%f b=%d \n" %( cg2[j][i], contingencyMatrix[j][i]))
        # print("\n")
       	
    contingencyMatrix = np.array(cg2)

    if mapping_method:
        col_ind = matching_Labels(contingencyMatrix)
    else:
        # Pairing using Hungarian Algo
        _, col_ind = linear_sum_assignment(-contingencyMatrix) #Hungarian algorithm (Variant Jonker–Volgenant algorithm)
    newL2 = np.zeros(len(L2))
    
    # breakpoint()
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[col_ind[i]]
    return newL2

if __name__ == "__main__":

    # python3 Accuracy.py -k 15 -m -i "../datasets/S-set/s1-label.pa" "../datasets/S-set/s1_data.txt"
    # python3 Accuracy.py -k 15 -p -i "../datasets/S-set/s1-label.pa" "../datasets/S-set/s1_data.txt"
    # python3 Accuracy.py -m -i "../datasets/S-set/s1-label.pa" "../datasets/S-set/s1-label.pa"
    # python3 Accuracy.py -p -i "../datasets/S-set/s1-label.pa" "../datasets/S-set/s1-label.pa"

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
        raise argparse.ArgumentError(None, 'Nb input file must be 2')
    
    # mapping_method is True with (-m) matching option
    # mapping_method is False with (-p) pairing option 
    mapping_method = False
    if args["Matching"]:
        mapping_method = True
    elif args["Pairing"]:
        mapping_method = False

    if mapping_method:
        ACCbtoa = accuracy(readFilePa(inputFilename[0]), readFilePa(inputFilename[1]), mapping_method=mapping_method)
        ACCatob = accuracy(readFilePa(inputFilename[1]), readFilePa(inputFilename[0]), mapping_method=mapping_method)
        accmin = min([ACCatob, ACCbtoa])
        accmean = np.mean([ACCatob, ACCbtoa])
        print("ACC with Matching labels:\n    (B=>A):%f (A=>B):%f min:%f mean:%f " %(ACCbtoa, ACCatob, accmin, accmean))
    else:
        ACC = accuracy(readFilePa(inputFilename[0]), readFilePa(inputFilename[1]), mapping_method=mapping_method)
        print("ACC with Pairing (Hungarian) labels = "+str(ACC))

    # accuracy(readFilePa(inputFilename[0]), readFilePa(inputFilename[1]), mapping_method=mapping_method)
