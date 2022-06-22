import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from ReadFilePointsPa import readFileDataTxt

"""Read 2D points file .txt"""
def readFileTxt(filename):
    # Reading form a file
    with open(filename, "r+") as file:
        arrayRet = np.empty((0,2),int)
        for line in file:
            tmp = line.rstrip().replace("\t", " ").split(" ")
            tmp = [e for e in tmp if e]
            tmp[0] = int(tmp[0])
            tmp[1] = int(tmp[1])
            arrayRet = np.append(arrayRet, np.array([[tmp[0],tmp[1]]]),axis=0)
    return arrayRet

#"../datasets/S-set/s1_data.txt"
"""Compute K-means centroid and partition file

Data format:
 [  x     y      pa]
[[53920 42968     9]
 [52019 42206     9]
 [52570 42476     9]
 ...
 [41140 18409    20]
 [37752 19891    20]
 [40164 17389    20]]

"""
def KmeansDataGCI(filenameData,nbCluster):
    datasetFile = readFileDataTxt(filenameData)

    # cluster the dataset into the pre-defined number of clusters
    kmeans = KMeans(n_clusters=nbCluster,init="random").fit(datasetFile)

    # read classification and add it in a dictonary
    clusters = defaultdict(list)
    for i, l in enumerate(kmeans.labels_):
        clusters[l].append(datasetFile[i])

    # create array with coords and partition label
    #arrayKmeansPointsPa = np.empty((0, 3), int)
    boolInitArray = True
    for cluster_name in sorted(clusters):
        for data in clusters[cluster_name]:
            if boolInitArray:
                arrayKmeansPointsPa = np.empty((0, len(data)+1), int)
                boolInitArray = False
            arrayKmeansPointsPa = np.append(arrayKmeansPointsPa, np.array(
                [[*data, int(cluster_name)+1]]), axis=0)
    return arrayKmeansPointsPa

""" Compute K-means centroids
"""
def KmeansDataCI(filenameData, nbCluster):
    datasetFile = readFileDataTxt(filenameData)

    # cluster the dataset into the pre-defined number of clusters
    kmeans = KMeans(n_clusters=nbCluster,init="random").fit(datasetFile)

    centroids = kmeans.cluster_centers_
    dict_centroid = {}
    i = 1
    for cent in centroids:
        dict_centroid["C"+str(i)] = [int(cent[0]),int(cent[1])]
        i += 1
    return dict_centroid


def KmeansDataAccuracy(filenameData,nbCluster):
    datasetFile = readFileDataTxt(filenameData)

    # cluster the dataset into the pre-defined number of clusters
    kmeans = KMeans(n_clusters=nbCluster,init="random").fit(datasetFile)

    return kmeans.labels_
