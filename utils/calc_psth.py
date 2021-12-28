from light_removal import combine_list
from preprocessing_pipeline import load_cluster
import glob
import scipy.io as io
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from enum import Enum
from constants import PV_COLOR, LIGHT_PV

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

SAVE_PATH = '../../../data for figures/New/'


def gaussian_func(sig, x):
    return np.exp(- (x ** 2) / (2 * sig ** 2))

def create_window(n, sig, sup):
    assert n - 1 == sig * sup * 2
    xs = np.linspace(-sup * sig, sup * sig + 1, n)
    wind = gaussian_func(sig, xs)
    return wind / wind.sum()

def mirror_edges(arr, wind_size):
    left_edge = arr[1: wind_size // 2][::-1]
    right_edge = arr[-wind_size // 2 - 1: -1]

    return np.concatenate((left_edge, arr, right_edge), axis=0)

class STIM_MODE(Enum):
    ALL = 1  # Any case
    SIM = 2  # Only simultaneous in all
    CUR = 3  # Only in current

def get_pair_lst(path, mode=STIM_MODE.ALL):
    mat = io.loadmat(path, appendmat=False, simplify_cells=True)
    try:
        types = mat['stim']['types']  # Only need pulse
        stm_pairs = mat['stim']['times']
        amps = mat['stim']['vals']
        indexes = None
        if mode == STIM_MODE.ALL:
            indexes = np.ones(len(types))
        elif mode == STIM_MODE.SIM:
            indexes = 1 - mat['stim']['index']
        elif mode == STIM_MODE.CUR:
            indexes = mat['stim']['index']
        else:
            raise NotImplementedError
        durs = np.array([b-a for a, b in stm_pairs])
        mask = (types == 'PULSE') * (durs > 900) * (durs < 1100) * indexes * (amps > 0.059)
        stm_pairs = stm_pairs[mask.astype('bool')]

    except KeyError:
        print(f'Skipping {path} for missing stim and/or times fields')
        return None
    if len(stm_pairs) == 0:
        return None

    return stm_pairs

def calc_hist(spike_train, stims, bins):
    spike_train = spike_train * 20_000 / 1000
    wind_size = int(20 * 3.5 * 2 + 1)
    g_wind = create_window(wind_size, 20, 3.5)
    ret = np.zeros((len(stims), len(bins) - 1))
    for i, stim in enumerate(stims):
        ref_time_list = spike_train - stim
        mask = (ref_time_list >= bins[0]) * (ref_time_list < bins[-1])
        ref_time_list = ref_time_list[mask]
        hist, _ = np.histogram(ref_time_list, bins=bins) 
        mirr_hist = mirror_edges(hist, wind_size)
        conv_hist = np.convolve(mirr_hist, g_wind, mode='valid')
        ret[i] = conv_hist

    ret *= (20_000 / len(stims))  # TODO this normalization is only valid for 1 sample bin

    return ret.mean(axis=0), stats.sem(ret, axis=0)


def check_stim_shank(shank, f):
    try:
        f_shank = int(f.split('.')[-1]) - 32
        return f_shank == shank
    except ValueError:
        return False


def main(data_path, cluster_name, temp_state, ax, mode=STIM_MODE.ALL):
    cluster = load_cluster(temp_state, cluster_name)

    file_lst = glob.glob(data_path + f'{cluster.filename}/{cluster.filename}.stm.*')
    pairs_temp = []
    for f in file_lst:
        if not check_stim_shank(cluster.shank, f):
            continue
        p = get_pair_lst(f, mode)
        if p is None:
            continue
        pairs_temp.append(p)
    pairs = combine_list(pairs_temp)

    bins = np.linspace(-50 * 20, 100 * 20, 150 * 20 + 1)

    spike_train = cluster.timings

    hist = calc_hist(spike_train, [start for start, end in pairs], bins)

    wind_size = 141
    mirr_hist = mirror_edges(hist, wind_size)

    g_wind = create_window(wind_size, 20, 3.5)
    conv_hist = np.convolve(mirr_hist, g_wind, mode='valid')

    rect = patches.Rectangle((0, 0), 50, conv_hist.max(), facecolor='c', alpha=0.2)

    # Add the patch to the Axes
    ax.add_patch(rect)
    ax.plot(np.linspace(-50, 100, 150 * 20), conv_hist)
    ax.set_ylabel('avg spike/second')
    ax.set_xlabel('ms')
    ax.set_title(cluster_name)

    return ax


if __name__ == "__main__":
    data_path = '../Data/'
    cluster_name = 'es25nov11_13_3_11'

    cluster = load_cluster('../temp_state/', cluster_name)

    file_lst = glob.glob(data_path + f'{cluster.filename}/{cluster.filename}.stm.*')
    pairs_temp = []
    print('reading stm files')
    for f in file_lst:
        try:
            if cluster.shank != int(f.split('.')[-1]) - 32:
                continue
        except ValueError:
            continue
        p = get_pair_lst(f)
        if p is None:
            continue
        pairs_temp.append(p)
    pairs = combine_list(pairs_temp)

    c = 20  # TODO allow differnet c values, currently hard coded in calc_hist

    bins = np.linspace(-50 * 20, 100 * 20, 150 * c + 1)

    spike_train = cluster.timings

    hist, sem = calc_hist(spike_train, [start for start, end in pairs], bins)

    """wind_size = int(c * 3.5 * 2 + 1)
    mirr_hist = mirror_edges(hist, wind_size)
    mirr_sems = mirror_edges(sems, wind_size)

    g_wind = create_window(wind_size, c, 3.5)
    conv_hist = np.convolve(mirr_hist, g_wind, mode='valid')
    conv_sem = np.convolve(mirr_sems, g_wind, mode='valid')"""

    fig, ax = plt.subplots(figsize=(30, 8))
    #rect = patches.Rectangle((0, 0), 50, hist.max(), facecolor='b', alpha=0.2)
    ax.axvline(x=0, ymax=np.max(hist + sem), color='k', linestyle='--')
    ax.axvline(x=50, color='k', linestyle='--')


    # Add the patch to the Axes
    #ax.add_patch(rect)
    ax.plot(np.linspace(-50, 100, 150 * c), hist, color=PV_COLOR)
    ax.fill_between(np.linspace(-50, 100, 150 * c), hist - sem, hist + sem, color=LIGHT_PV, alpha=0.2)
    ax.set_ylabel('avg spike/second')
    ax.set_xlabel('ms from onset')

    #plt.show()
    plt.savefig(SAVE_PATH + f"PSTH.pdf", transparent=True)
