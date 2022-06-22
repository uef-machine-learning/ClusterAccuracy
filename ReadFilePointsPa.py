import numpy as np

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
    with open(filename, "r+") as file:
        # Reading form a file
        listPartitionLabel = []
        firstLineBool = False
        for line in file:
            if firstLineBool:
                listPartitionLabel.append(int(line.rstrip()))
            if '-' in line:
                firstLineBool = True
    return listPartitionLabel

# "../datasets/S-set/s1-label.pa"
"../datasets/S-set/s1_data.txt"

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



