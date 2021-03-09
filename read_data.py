import numpy as np
import time
from clusters import Spike, Cluster
import scipy.io

NUM_BYTES = 2  # number of bytes to read each time from the spk files, based on the data format of 16 bit integers
NUM_CHANNELS = 8
NUM_SAMPLES = 32


def get_next_spike(spk_file):
    """
    input:
    spkFile: file descriptor; the file from which to read

    return:
    spike: Spike object; containing the next spike in the file
    """
    data = np.zeros((NUM_CHANNELS, NUM_SAMPLES))
    for i in range(NUM_SAMPLES):
        for j in range(NUM_CHANNELS):
            num = spk_file.read(NUM_BYTES)
            if not num:
                return None

            data[j, i] = int.from_bytes(num, "little", signed=True)
    spike = Spike()
    spike.data = data
    return spike


def get_next_cluster_num(clu_file):
    """
    input:
    cluFile: file descriptor; the file from which to read

    return:
    num: int; cluster ID
    """
    num = clu_file.readline()
    assert num != ''
    return int(num)


def find_indices_in_filenames(target_name, cell_class_mat):
    """
    Finds the relevant slice in the spv information

    input:
    target_name: string; recording name
    cell_class_mat: list; spv information

    return:
    start_index, end_index: integer tuple; start and end indices of the relevant data in the spv
    """
    file_name_arr = cell_class_mat['filename'][0][0]

    index = 0
    start_index = 0
    # find first occurrence of targetName
    for filenameArr in file_name_arr:
        filename = filenameArr[0][0]
        if filename == target_name:
            start_index = index
            break
        index += 1

    # find last occurrence of targetName
    for i in range(start_index, len(file_name_arr)):
        if file_name_arr[i][0][0] != target_name:
            return start_index, i

    end_index = len(file_name_arr)

    return start_index, end_index


def find_cluster_index_in_shankclu_vector(start_index, end_index, shank_num, clu_num, cell_class_mat):
    """
    Finds the index in the spv for the data

    input:
    start_index: int; start of relevant data
    end_index: int; end of relevant data
    shank_num: int shank number
    clu_num: int; cluster ID
    cell_class_mat: list; spv information

    return:
    index: integer; relevant index in the spv information, None if not found
    """
    shank_clu_vec = cell_class_mat['shankclu'][0][0]
    for i in range(start_index, end_index):
        shank_clu_entry = shank_clu_vec[i]
        if shank_clu_entry[0] == shank_num and shank_clu_entry[1] == clu_num:  # found
            return i
    return None


def determine_cluster_label(filename, shank_num, clu_num, cell_class_mat):
    """
    Determines cluster's label based on the spv information

    input:
    filename: string; recording name
    shank_num: int; shank number
    clu_num: int; cluster ID
    cell_class_mat: list; spv information

    return:
    label: integer; see function's body for specification
    """
    start_index, end_index = find_indices_in_filenames(filename, cell_class_mat)
    clu_index = find_cluster_index_in_shankclu_vector(start_index, end_index, shank_num, clu_num, cell_class_mat)
    is_act = cell_class_mat['act'][0][0][clu_index][0]
    is_exc = cell_class_mat['exc'][0][0][clu_index][0]
    is_inh = cell_class_mat['inh'][0][0][clu_index][0]

    if clu_index is None:
        return -2

    # 0 = PV
    # 1 = Pyramidal
    # -3 = both (pyr and PV) which means it will be discarded
    # -1 = untagged
    # -2 = clusters that appear in clu file but not in shankclu
    if is_exc == 1:
        if is_act == 1 or is_inh == 1:  # check if both conditions apply (will be discarded)
            return -3

        return 1

    if is_act == 1 or is_inh == 1:
        return 0
    return -1


def create_cluster(name, clu_num, shank_num, cell_class_mat):
    """
    Cluster creator
    
    input:
    name: string; recording name
    clu_num: integer; cluster ID
    shank_num: integer; shank number
    cell_class_mat: list; containing the spv information

    return:
    cluster: new Cluster object
    """
    # get cluster's label
    label = determine_cluster_label(name, shank_num, clu_num, cell_class_mat)

    # Check if the cluster doesn't appear in shankclu
    if label == -2:
        return None

    cluster = Cluster(label=label, filename=name, num_within_file=clu_num, shank=shank_num)
    return cluster


def read_directory(path, cell_class_mat, i):
    """
    The reader function.

    input:
    path: string; path to the to the recording directory
    cell_class_mat: list; the spv information
    i: integer; the shank number

    return:
    clusters_list: list of Cluster objects
    """
    clusters = dict()
    clusters_list = []
    name = path.split("\\")[-1]

    start = time.time()
    try:
        spkFile = open(path + "\\" + name + ".spk." + str(i), 'rb')  # file containing spikes
        cluFile = open(path + "\\" + name + ".clu." + str(i))  # file containing cluster mapping of spikes
    except FileNotFoundError:  # if shank recording doesn't exsist exit
        print(path + "\\" + name + ".spk." + str(i) + ' and/or ' + path + "\\" + name + ".clu." + str(i) + ' not found')
        return []

    # Read the first line of the cluster file (contains num of clusters)
    get_next_cluster_num(cluFile)
    spike = get_next_spike(spkFile)
    while spike is not None:  # for each spike
        clu_num = get_next_cluster_num(cluFile)  # cluster ID

        # clusters 0 and 1 are artefacts and noise by convention
        if clu_num == 0 or clu_num == 1:
            spike = get_next_spike(spkFile)
            continue

        assert clu_num is not None
        full_name = name + "_" + str(i) + "_" + str(clu_num)  # the format is filename_shankNum_clusterNum

        # Check if cluster exists in dictionary and create if not
        if full_name not in clusters:
            new_cluster = create_cluster(name, clu_num, (i), cell_class_mat)

            # Check to see if the cluster we are trying to create is one that doesn't appear in shankclu (i.e has a
            # label of -2)
            if new_cluster is None:
                spike = get_next_spike(spkFile)
                continue

            clusters[full_name] = new_cluster
            clusters_list.append(new_cluster)

        clusters[full_name].add_spike(spike)
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
    cell_class_mat = scipy.io.loadmat(path_to_mat)['sPV']
    dirs_file = open(path_to_dirs_file)
    for line in dirs_file:
        split_line = line.split()
        data_dir, remove_inds = split_line[0], split_line[1:]
        print("reading " + str(data_dir))
        for i in range(1, 5):
            if str(i) in remove_inds:  # skip shanks according to instruction in dirs file
                print('Skipped shank %d in file %s' % (i, data_dir))
                continue
            dir_clusters = read_directory(data_dir, cell_class_mat, i)  # read the data files of shank i
            print("the number of clusters is: " + str(len(dir_clusters)))
            yield dir_clusters
    dirs_file.close()
