import numpy as np
import scipy.signal as signal
import time
from constants import MIN_TIME_LIST, VERBOS, INF, DEBUG, ACH_WINDOW
import matplotlib.pyplot as plt
import scipy.io as io

from features.temporal_features.FET_DKL import DKL
from features.temporal_features.FET_jump_index import Jump
from features.temporal_features.FET_PSD import PSD
from features.temporal_features.FET_rise_time import RiseTime
from features.temporal_features.FET_unif_dist import UnifDist

features = [DKL(), Jump(), PSD(), RiseTime(), UnifDist()]


def get_array_length(chunks):
    counter = 0
    for chunk in chunks:
        counter += len(chunk)
    return counter

def invert_chunks(chunks):
    ret = np.zeros(get_array_length(chunks), dtype=np.int)
    for i, chunk in enumerate(chunks):
        ret[chunk] = i  # note that chunk is a list
    return ret

def calc_temporal_histogram(time_lst, bins, chunks):
    ret = np.zeros((len(chunks), len(bins) - 1))
    chunks_inv = invert_chunks(chunks)
    end_time = time_lst.max()
    for i in range(len(time_lst)):
        if time_lst[i] + ACH_WINDOW > end_time:
            break
        ref_time_list = time_lst - time_lst[i]
        mask = (ref_time_list > 0) * (ref_time_list <= ACH_WINDOW)
        ref_time_list = ref_time_list[mask]
        hist, _ = np.histogram(ref_time_list, bins=bins)
        ret[chunks_inv[i]] += hist
    if DEBUG:
        raise NotImplementedError
        print(f"ret sum {ret.sum()}")
        # TODO adapt for chunks somehow
        plt.bar(np.arange(len(ret)), ret)
        plt.title(f"num spikes is {len(time_lst)}")
        plt.show()

    return ret


def calc_temporal_features(time_lst, chunks, resolution=2, bin_range=ACH_WINDOW, upsample=8, cdf_range=50,
                           jmp_min=50, jmp_max=ACH_WINDOW):
    feature_mat_for_cluster = None

    time_lst = np.array(time_lst)

    N = resolution * bin_range + 1
    bins = np.linspace(0, bin_range, N)
    start_time = time.time()
    histograms = calc_temporal_histogram(time_lst, bins, chunks)

    sst = 'Data/es04feb12_1/es04feb12_1.sst'
    cell_class_mat = io.loadmat(sst, appendmat=False, simplify_cells=True)
    ach = cell_class_mat['sst']['ach'].T
    plt.bar(np.arange(len(ach[1][len(ach[1]) // 2 - 1:])),
            ach[1][len(ach[1]) // 2 - 1:] / ach[1][len(ach[1]) // 2 - 1:].max(), alpha=0.5, label='ach')
    plt.bar(np.arange(len(histograms[0])), histograms[0]/histograms[0].max(), alpha=0.5, label='hist')
    plt.legend()
    plt.show()
    histograms = np.array([signal.resample(histogram, upsample * N) for histogram in histograms])
    histograms = np.where(histograms >= 0, histograms, 0)
    end_time = time.time()
    if VERBOS:
        print(f"histogram creation took {end_time - start_time:.3f} seconds")
    start_band = histograms[:, :resolution * cdf_range * upsample]
    start_cdf = (np.cumsum(start_band, axis=1).T / np.sum(start_band, axis=1)).T
    # TODO: do it only in jump if not required elsewhere
    mid_band = histograms[:, resolution * jmp_min * upsample: resolution * jmp_max * upsample + 1]

    for feature in features:
        start_time = time.time()
        feature.set_fields(resolution=2, cdf_range=cdf_range, jmp_min=jmp_min, jmp_max=jmp_max)
        mat_result = feature.calculate_feature(rhs=histograms, start_cdf=start_cdf, mid_band=mid_band)
        if feature_mat_for_cluster is None:
            feature_mat_for_cluster = mat_result
        else:
            feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)
        end_time = time.time()
        if VERBOS:
            print(f"feature {feature.name} processing took {end_time - start_time:.3f} seconds")
    mat_result = np.ones((len(chunks), 1)) * len(time_lst)
    feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)
    return feature_mat_for_cluster


def get_temporal_features_names():
    names = []
    for feature in features:
        names += feature.headers
    names += ['num_spikes']
    return names
