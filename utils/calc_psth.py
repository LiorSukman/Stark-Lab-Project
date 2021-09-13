from light_removal import combine_list
from preprocessing_pipeline import load_cluster
import glob
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def gaussian_func(sig, x):
    return np.exp(- (x ** 2) / (2 * sig ** 2))

def create_window(n, sig, sup):
    xs = np.linspace(-sup * sig, sup * sig, n)
    wind = gaussian_func(sig, xs)
    return wind / wind.sum()

def mirror_edges(arr, wind_size):
    left_edge = arr[1: wind_size // 2][::-1]
    right_edge = arr[-wind_size // 2 - 1: -1]

    return np.concatenate((left_edge, arr, right_edge), axis=0)

def get_pair_lst(path):
    mat = io.loadmat(path, appendmat=False, simplify_cells=True)
    try:
        types = mat['stim']['types']  # Only need pulse
        stm_pairs = mat['stim']['times']
        amps = mat['stim']['vals']
        durs = np.array([b-a for a, b in stm_pairs])
        mask = (types == 'PULSE') * (amps > 0.064) * (amps < 0.066) * (durs > 900) * (durs < 1100)
        stm_pairs = stm_pairs[mask]
        amps = amps[mask]

    except KeyError:
        print(f'Skipping {path} for missing stim and/or times fields')
        return None
    if len(stm_pairs) == 0:
        return None

    return stm_pairs

def calc_hist(spike_train, stims, bins):
    spike_train = spike_train * 20_000 / 1000
    ret = np.zeros(len(bins) - 1)
    for stim in stims:
        ref_time_list = spike_train - stim
        # print(bins[0], bins[-1])
        mask = (ref_time_list >= bins[0]) * (ref_time_list < bins[-1])
        #  print(mask.sum())
        ref_time_list = ref_time_list[mask]
        hist, _ = np.histogram(ref_time_list, bins=bins)
        ret += hist

    return 20_000 * ret / len(stims)


if __name__ == "__main__":
    data_path = '..\\Data\\'
    cluster_name = '04feb12_1_2_22'

    cluster = load_cluster('..\\temp_state\\', cluster_name)

    file_lst = glob.glob(data_path + f'{cluster.filename}\\{cluster.filename}.stm.*')
    print(data_path + f'{cluster.filename}\\{cluster.filename}.stm.*')
    pairs_temp = []
    print('reading stm files')
    for f in file_lst:
        p = get_pair_lst(f)
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

    fig, ax = plt.subplots()
    rect = patches.Rectangle((0, 0), 50, conv_hist.max(), facecolor='r', alpha=0.2)

    # Add the patch to the Axes
    ax.add_patch(rect)
    ax.plot(np.linspace(-50, 100, 150 * 20), conv_hist)
    ax.set_ylabel('avg spike/second')
    ax.set_xlabel('ms')
    plt.show()
