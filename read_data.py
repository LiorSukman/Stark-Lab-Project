import numpy as np
import time
from clusters import Spike, Cluster
import scipy.io

NUM_BYTES = 2 # number of bytes to read each time from the spk files, based on the data format of 16 bit integers
NUM_CHANNELS = 8
NUM_SAMPLES = 32

def get_next_spike(spkFile):
    """
    input:
    spkFile: file descriptor; the file from which to read

    return:
    spike: Spike object; containing the next spike in the file
    """
    data = np.zeros((NUM_CHANNELS, NUM_SAMPLES))
    for i in range(NUM_SAMPLES):
        for j in range(NUM_CHANNELS):
            num = spkFile.read(NUM_BYTES) 
            if not num:
                return None

            data[j, i] = int.from_bytes(num, "little", signed = True) 
    spike = Spike()
    spike.data = data
    return spike   

def get_next_cluster_num(cluFile):
    """
    input:
    cluFile: file descriptor; the file from which to read

    return:
    num: int; cluster ID
    """
    num = cluFile.readline()
    assert num != ''
    return int(num)

def find_indices_in_filenames(targetName, cellClassMat):
    """
    Finds the relevant slice in the spv infornation

    input:
    filenamtargetName: string; recording name
    cellClassMat: list; spv information

    return:
    startIndex, endIndex: integer tupple; start and end indices of the relevant data in the spv
    """
    filenamesArr = cellClassMat['filename'][0][0]

    index = 0
    startIndex = 0
    # find first occurance of targetName
    for filenameArr in filenamesArr:
        filename = filenameArr[0][0] 
        if filename == targetName:
            startIndex = index
            break
        index += 1

    # find last occurance of targetName
    for i in range(startIndex, len(filenamesArr)):
        if filenamesArr[i][0][0] != targetName:
            return startIndex, i

    return startIndex, len(filenamesArr)

def find_cluster_index_in_shankclu_vector(startIndex, endIndex, shankNum, cluNum, cellClassMat):
    """
    Finds the index in the spv for the data

    input:
    startIndex: int; start of relevant data
    endIndex: int; end of relevant data
    shankNum: int shank number
    cluNum: int; cluster ID
    cellClassMat: list; spv information

    return:
    index: integer; relevant index in the spv information, None if not found
    """
    shankCluVec = cellClassMat['shankclu'][0][0]
    for i in range(startIndex, endIndex):
        shankCluEntry = shankCluVec[i]
        if shankCluEntry[0] == shankNum and shankCluEntry[1] == cluNum: # found
            return i
    return None

def determine_cluster_label(filename, shankNum, cluNum, cellClassMat):
    """
    Determines cluster's label based on the spv information

    input:
    filename: string; recording name
    shankNum: int; shank number
    cluNum: int; cluster ID
    cellClassMat: list; spv information

    return:
    label: integer; see function's body for specification
    """
    startIndex, endIndex = find_indices_in_filenames(filename, cellClassMat)
    cluIndex = find_cluster_index_in_shankclu_vector(startIndex, endIndex, shankNum, cluNum, cellClassMat)
    isAct = cellClassMat['act'][0][0][cluIndex][0]
    isExc = cellClassMat['exc'][0][0][cluIndex][0]
    isInh = cellClassMat['inh'][0][0][cluIndex][0]

    if cluIndex == None:
        return -2

    # 0 = PV
    # 1 = Pyramidal
    # -3 = both (pyr and PV) which means it will be discarded
    # -1 = untagged
    # -2 = clusters that appear in clu file but not in shankclu
    if isExc == 1: 
        if isAct == 1 or isInh == 1: # check if both conditions apply (will be discarded)
            return -3
        
        return 1

    if isAct == 1 or isInh == 1:
            return 0
    return -1


def create_cluster(name, cluNum, shankNum, cellClassMat):
    """
    Cluster creator
    
    input:
    name: string; recording name
    cluNum: integer; cluster ID
    shankNum: integer; shank numer
    cellClassMat: list; containig the spv information

    return:
    cluster: new Cluster object
    """
    # get cluster's label
    label = determine_cluster_label(name, shankNum, cluNum, cellClassMat)

    # Check if the cluster doesn't appear in shankclu
    if label == -2:
        return None

    cluster = Cluster(label = label, filename = name, numWithinFile = cluNum, shank = shankNum)
    return cluster

def read_directory(path, cellClassMat, i):
    """
    The reader function.

    input:
    path: string; path to the to the recording directory
    cellClassMat: list; the spv information
    i: integer; the shank number

    return:
    clusters_list: list of Cluster objects
    """
    clusters = dict()
    clusters_list = []
    name = path.split("\\")[-1]
    
    start = time.time()
    try:
        spkFile = open(path + "\\" + name + ".spk." + str(i), 'rb') # file containing spikes
        cluFile = open(path + "\\" + name + ".clu." + str(i)) # file containing cluster mapping of spikes
    except FileNotFoundError: # if shank recording doesn't exsist exit
        print(path + "\\" + name + ".spk." + str(i) + ' and/or ' + path + "\\" + name + ".clu." + str(i) + ' not found')
        return []

    # Read the first line of the cluster file (contains num of clusters)
    get_next_cluster_num(cluFile)
    spike = get_next_spike(spkFile)    
    while(spike is not None): # for each spike
        cluNum = get_next_cluster_num(cluFile) # cluster ID 

        # clusters 0 and 1 are artefacts and noise by convention
        if cluNum == 0 or cluNum == 1:
            spike = get_next_spike(spkFile)  
            continue

        assert cluNum != None
        fullName = name + "_" + str(i) + "_" + str(cluNum) # the format is filename_shankNum_clusterNum

        # Check if cluster exists in dictionary and create if not
        if fullName not in clusters:
            new_cluster = create_cluster(name, cluNum, (i), cellClassMat)

            # Check to see if the cluster we are trying to create is one that doesn't appear in shankclu (i.e has a label of -2)
            if new_cluster is None:
                spike = get_next_spike(spkFile) 
                continue

            clusters[fullName] = new_cluster
            clusters_list.append(new_cluster)

        clusters[fullName].add_spike(spike)
        spike = get_next_spike(spkFile)
        

    print("Finished File %s with index %d" % (name, i))
    spkFile.close()
    cluFile.close()

    end = time.time()
    print(str(end - start) + " total")

    return clusters_list

def read_all_directories(path_to_dirs_file, path_to_mat):
    """
    The main function of the file, called from the pipeline.
    This is a generator function 

    input:
    path_to_dirs_file: string; path to the file containing the paths to the data directories
    path_to_mat: string; path to the mat file containing the spv information

    return:
    dir_clusters: list of Cluster objects; each time all the clusters from a single shank from a single recording
    """
    cellClassMat = scipy.io.loadmat(path_to_mat)['sPV']
    dirsFile = open(path_to_dirs_file)
    for line in dirsFile:
        split_line = line.split()
        data_dir, remove_inds = split_line[0], split_line[1:] 
        print("reading " + str(data_dir))
        for i in range(1,5):
            if str(i) in remove_inds: # skip shanks according to instruction in dirs file
                print('Skipped shank %d in file %s' % (i, data_dir))
                continue
            dir_clusters = read_directory(data_dir, cellClassMat, i) # read the data files of shank i
            print("the number of clusters is: " + str(len(dirClusters))) 
            yield dir_clusters
    dirsFile.close()
