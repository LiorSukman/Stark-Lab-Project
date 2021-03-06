import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np
import glob

from constants import SAMPLE_RATE

"""sst = 'Data/es04feb12_1/es04feb12_1.sst'
stim = 'Data/es25nov11_3/es25nov11_3.stm.sin'
cell_class_mat = io.loadmat(sst, appendmat=False, simplify_cells=True)

# relevant for sst
shankclu = cell_class_mat['sst']['shankclu']
print(shankclu[0])
ach = cell_class_mat['sst']['ach'].T
plt.bar(np.arange(len(ach[0][len(ach[0])//2 - 1:])), ach[0][len(ach[0])//2 - 1:]/ach[0][len(ach[0])//2 - 1:].max())
plt.title('sst')
plt.show()"""

def get_idnds(time_lst, pairs, remove_lights):
    i, j = 0, 0
    inds = []

    while i < len(time_lst):
        if j >= len(pairs):
            if remove_lights:
                inds.append(i)
            i += 1
        elif pairs[j][1] < time_lst[i]:
            j += 1
        elif pairs[j][0] <= time_lst[i] <= pairs[j][1]:
            if not remove_lights:
                inds.append(i)
            i += 1
        else:  # time_lst[i] < pairs[j][0]
            if remove_lights:
                inds.append(i)
            i += 1

    return inds

def remove_light(cluster, remove_lights, data_path='Data/', margin=10, pairs=None):
    file_lst = glob.glob(data_path + f'{cluster.filename}/{cluster.filename}.stm.*')
    if pairs is None:
        pairs_temp = []
        for f in file_lst:
            p = get_pair_lst(f, margin)
            if p is None:
                continue
            pairs_temp.append(p)
        pairs = combine_list(pairs_temp)

    timings = cluster.timings
    inds = get_idnds(timings, pairs, remove_lights)

    left_prop = len(inds) / len(timings)
    time_prop = (np.convolve(pairs.flatten(), [1, -1], mode='valid')[::2] - 2 * margin).sum() /\
                (timings.max() - timings.min())

    return inds, left_prop, time_prop, pairs

def get_pair_lst(path, margin):
    mat = io.loadmat(path, appendmat=False, simplify_cells=True)
    try:
        pairs = mat['stim']['times']
    except KeyError:
        print(f'Skipping {path} for missing stim and/or times fields')
        return None
    if len(pairs) == 0:
        return None

    return (pairs / (SAMPLE_RATE / 1000)) + [-margin, margin]

def combine_list(lsts):
    if len(lsts) == 0:
        return np.array([[-2, -1]])
    pairs = np.concatenate(lsts, axis=0)
    inds = np.argsort(pairs[:, 0])
    ret = []
    min_p, max_p = pairs[inds][0]
    for pair in pairs[inds][1:]:
        if min_p <= pair[0] <= max_p:
            max_p = max(pair[1], max_p)
        else:
            ret.append([min_p, max_p])
            min_p, max_p = pair
    ret.append([min_p, max_p])
    return np.array(ret)
