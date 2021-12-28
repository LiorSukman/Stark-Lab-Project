import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from preprocessing_pipeline import load_cluster
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

SAVE_PATH = '../../../data for figures/New/'


def calc_hist(spike_train, stims, bins):
    ret = np.zeros(len(bins) - 1)
    for stim in stims:
        ref_time_list = spike_train - stim
        # print(bins[0], bins[-1])
        mask = (ref_time_list >= bins[0]) * (ref_time_list < bins[-1])
        #  print(mask.sum())
        ref_time_list = ref_time_list[mask]
        hist, _ = np.histogram(ref_time_list, bins=bins)
        ret += hist

    return 1000 * ret / len(stims)

def calc_cc(pv_name, pyr_name, temp_path, ax, loc):
    pyr_clu = load_cluster(temp_path, pyr_name)
    pv_clu = load_cluster(temp_path, pv_name)

    pyr_timings = pyr_clu.timings
    pv_timings = pv_clu.timings

    timings = np.concatenate((pyr_timings, pv_timings))
    classes = np.concatenate((np.ones(len(pyr_timings)), np.ones(len(pv_timings)) * 2))

    sort_inds = np.argsort(timings)
    timings = timings[sort_inds]
    classes = classes[sort_inds]

    N = 2 * 1 * 20 + 2
    offset = 1 / (2 * 1)
    bins = np.linspace(-20 - offset, 20 + offset, N)

    hist = calc_hist(pv_timings, pyr_timings, bins)
    ax.bar(np.linspace(-20, 20, N - 1), hist)
    ax.axvline(0, ymin=0, ymax=hist.max(), color='k', linestyle='--')
    ax.set_title(f"{loc}")
    ax.set_ylabel('avg spike/second')
    ax.set_xlabel('ms')

    return ax


if __name__ == "__main__":
    temp_path = '../temp_state/'
    pyr_name = name = 'es25nov11_13_3_3'  # pyr
    pv_name = 'es25nov11_13_3_11'  # pv

    pyr_clu = load_cluster(temp_path, pyr_name)
    pv_clu = load_cluster(temp_path, pv_name)

    pyr_timings = pyr_clu.timings
    pv_timings = pv_clu.timings

    timings = np.concatenate((pyr_timings, pv_timings))
    classes = np.concatenate((np.ones(len(pyr_timings)), np.ones(len(pv_timings)) * 2))

    sort_inds = np.argsort(timings)
    timings = timings[sort_inds]
    classes = classes[sort_inds]

    mdic = {"timings": timings * 20, "classes": classes}
    # io.savemat("cc_mat.mat", mdic)

    N = 2 * 1 * 50 + 2
    offset = 1 / (2 * 1)
    bins = np.linspace(-50 - offset, 50 + offset, N)

    hist = calc_hist(pv_timings, pyr_timings, bins)
    fig, ax = plt.subplots(figsize=(30, 8))
    ax.vlines(0, ymin=0, ymax=hist.max(), color='k', linestyle='--')
    ax.bar(np.linspace(-50, 50, N - 1), hist)

    plt.savefig(SAVE_PATH + f"CCH.pdf", transparent=True)

