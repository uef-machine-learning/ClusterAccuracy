
# Implemented by Martin Dautriche

import numpy as np
import argparse
from scipy.optimize import linear_sum_assignment
from ReadFilePointsPa import readFilePa
from Kmeans_Functions import KmeansDataAccuracy
import sklearn
from sklearn.metrics import accuracy_score
"""
Matlab Code
function ACC = accuracy(Y, predY)
% Y: true cluster label vector; preY: predicted cluster label vector
% Y: partition label array
% preY: partition label array
res = bestMap(Y, predY); -> renamed mapping_labels below
% find(Y == res) -> find if same index has the same value in the two array 
ACC = length(find(Y == res))/length(Y);
"""

# Y: true cluster label vector; preY: predicted cluster label vector
# Y: partition label array
# preY: partition label array
def accuracy(Y, predY, mapping_method=True):
    res = mapping_labels(Y, predY, mapping_method)
    ACC = (find_matlab(Y,res)/len(Y))
    if mapping_method:
        print("ACC with Matching labels = "+str(ACC))
    else:
        print("ACC with Pairing (Hungarian) labels = "+str(ACC))

# Equivalent to find function in Matlab
# Return how many time Y[i] is equals with res[i]
def find_matlab(Y, res):
    k = 0
    for i in range(len(Y)):
        if Y[i] == res[i]:
            k = k+1
    return k

"""
# Return contingency matrix
def contingency_matrix_function(L1,L2,Label1_value,Label2_value):
    k = 0
    for i in range(len(L1)):
        if  ((L1[i] == Label1_value) and (L2[i] == Label2_value)):
            k = k+1
    return k
"""

# Matching labels using contingency matrix
def matching_Labels(contingencyMatrix):
    match_labels = []
    for i in range(len(contingencyMatrix)):
        match_labels.append(contingencyMatrix[i].tolist().index(max(contingencyMatrix[i])))
    return match_labels

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
        Label2 = Label2 + ([0]*(nClass1-nClass2))
    elif nClass2 > nClass1:
        Label1 = [*Label1,*([0]*(nClass2-nClass1))]

    """
    nClass = max(nClass1,nClass2)
    contingencyMatrix = np.zeros((nClass,nClass))
    for i in range(0,nClass1):
        for j in range(0,nClass2):
            # contingencyMatrix(i,j) = length(find(L1 == Label1(i) & L2 == Label2(j))); MatLab
            contingencyMatrix[i,j] = contingency_matrix_function(L1,L2,Label1[i],Label2[j])
    """
    
    contingencyMatrix = sklearn.metrics.cluster.contingency_matrix(L2,L1)

    if mapping_method:
        col_ind = matching_Labels(contingencyMatrix)
    else:
        # Pairing using Hungarian Algo
        _, col_ind = linear_sum_assignment(-contingencyMatrix) #Hungarian algorithm (Variant Jonker–Volgenant algorithm)
    newL2 = np.zeros(len(L2))
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[col_ind[i]]
    return newL2

if __name__ == "__main__":

    # python3 Accuracy.py -k 15 -m -i "../datasets/S-set/s1-label.pa" "../datasets/S-set/s1_data.txt"
    # python3 Accuracy.py -k 15 -p -i "../datasets/S-set/s1-label.pa" "../datasets/S-set/s1_data.txt"
    # python3 Accuracy.py -m -i "../datasets/S-set/s1-label.pa" "../datasets/S-set/s1-label.pa"
    # python3 Accuracy.py -p -i "../datasets/S-set/s1-label.pa" "../datasets/S-set/s1-label.pa"

    # ARGUMENTS FOR THE EXTRACTOR
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-k", "--NbClusters", required=False,
                           nargs="?", const=0, type=int, help="Nb Cluster")
    argparser.add_argument("-i", "--inputFile", required=True,
                           help="Number of path to input file must be 2", nargs="*", type=str)

    group = argparser.add_mutually_exclusive_group(required=True)
    group.add_argument("-m", "--Matching", required=False,
                           help="Matching labels", action='store_true')
    group.add_argument("-p", "--Pairing", required=False,
                           help="Pairing labels with Hungarian Algo", action='store_true')


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

    if args["NbClusters"] is not None:
        nbClusters = args["NbClusters"]
    else:
        nbClusters = -1

    if nbClusters == -1:
        accuracy(readFilePa(inputFilename[0]), readFilePa(inputFilename[1]), mapping_method=mapping_method)
    else:
        accuracy(readFilePa(inputFilename[0]), KmeansDataAccuracy(inputFilename[1], nbClusters), mapping_method=mapping_method)
