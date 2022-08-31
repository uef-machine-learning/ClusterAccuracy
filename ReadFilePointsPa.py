import numpy as np
import re

"""Read 2D points file .txt

Data Format:
 [  x     y  ]
[[53920 42968]
 [52019 42206]
 [52570 42476]
 ...
 [41140 18409]
 [37752 19891]
 [40164 17389]]

"""
def readFileDataTxt(filename):
    with open(filename, "r+") as file:
        # Reading form a file
        boolInitArray = True
        for line in file:
            tmp = line.rstrip().replace("\t", " ").split(" ")
            tmp = [e for e in tmp if e]
            if boolInitArray:
                arrayPoints = np.empty((0, len(tmp)), int)
                boolInitArray = False
            for i in range(len(tmp)):
                tmp[i] = float(tmp[i])
            arrayPoints = np.append(arrayPoints, np.array(
                [tmp]), axis=0)
    return arrayPoints


"""Read centroid .txt datafile"""
def readCentroidFileTxt(filename):
    # Dictionary contain cluster's names + 2D coords
    dict_centroid = {}
    # Reading form a file
    with open(filename, "r+") as file:
        i = 1
        for line in file:
            # Remove spaces
            tmp = line.rstrip().replace("   ", " ").replace("\t", " ").split(" ")
            # Remove empty value list
            tmp = [e for e in tmp if e]

            for k in range(len(tmp)):
                tmp[k] = float(tmp[k])

            dict_centroid["C"+str(i)] = tmp
            i += 1
    return dict_centroid


"""Read partition datafile

Data Format: List of parition label [.,.,.,...]

"""
def readFilePa(filename):
    text_file = open(filename, "r")
    data = text_file.read()
    d2 = re.sub(re.compile(".*--\n",re.MULTILINE|re.DOTALL),"",data)
    X = np.fromstring(d2,dtype=int,sep="\n")
    minx = min(X)
    maxx = max(X)
    for i in range(len(X)):
        X[i] = X[i] - minx + 1
    if len(np.unique(X)) != maxx:
        raise argparse.ArgumentError(None, 'Input labels must be consequtive integers each on its own line.')
    return X


"""Create an array with data from points file and partition file

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
def readFileDataPa(filenamePoints,filenamePa):
    arrayData = readFileDataTxt(filenamePoints)
    arrayPartition = np.array([readFilePa(filenamePa)]).transpose()
    arrayPointsPar = np.append(arrayData, arrayPartition, axis=1)
    return arrayPointsPar



