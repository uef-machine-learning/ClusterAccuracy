Implementation of clustering accuracy in Python. 

Using pairing (Hungarian algorithm):

    python3 Accuracy.py -p -i data/s1-label.pa data/s1_clustering_result.pa  
    ACC with Pairing (Hungarian) labels = 0.8244


Using matching:

    python3 Accuracy.py -m -i data/s1-label.pa data/s1_clustering_result.pa  
    ACC with Matching labels = 0.824400
    (B=>A):0.824400 (A=>B):0.873800 min:0.824400 mean:0.849100 
   
A=data/s1-label.pa (ground truth)
B=data/s1_clustering_result.pa  
(B=>A) signifies the direction of mapping, i.e. map result to ground truth.

# Input format

Labels as integers each on its own line. See file data/s1_clustering_result.pa as example. Header can be omitted. 
