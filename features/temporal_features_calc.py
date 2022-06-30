import numpy as np
import scipy.signal as signal
import time
from constants import VERBOS, DEBUG, ACH_WINDOW
import matplotlib.pyplot as plt

from features.temporal_features.FET_DKL import DKL
from features.temporal_features.FET_jump_index import Jump
from features.temporal_features.FET_PSD import PSD
from features.temporal_features.FET_rise_time import RiseTime
from features.temporal_features.FET_unif_dist import UnifDist
from features.temporal_features.FET_firing_rate import FiringRate

features = [DKL(), Jump(), PSD(), RiseTime(), UnifDist()]
ind_features = [FiringRate()]


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


def ordered_array_histogram(lst, bins, i0):
    max_abs = bins[-1]
    bin_size = bins[1] - bins[0]
    bin0 = len(bins) // 2
    hist = np.zeros(len(bins) - 1)

    i = i0 + 1
    while i <= len(lst) - 1 and lst[i] < max_abs:
        bin_ind = int(((lst[i] - (bin_size / 2)) // bin_size) + bin0)
        hist[bin_ind] += 1
        i += 1
    i = i0 - 1
    while i >= 0 and lst[i] >= -max_abs:
        bin_ind = int(((lst[i] - (bin_size / 2)) // bin_size) + bin0)
        hist[bin_ind] += 1
        i -= 1

    return hist

def compare_hists(a, b):
    if len(a) != len(b):
        print('length problem')
        return False
    for i, vals in enumerate(zip(a, b)):
        b1, b2 = vals
        if b1 != b2:
            print('val problem in index', i, b1, b2)
            print(a[i+1], b[i+1])
            return False
    return True

def calc_temporal_histogram(time_lst, bins, chunks):
    ret = np.zeros((len(chunks), len(bins) - 1))
    counter = np.zeros((len(chunks), 1))
    chunks_inv = invert_chunks(chunks)
    end_time = time_lst.max()
    for i in range(len(time_lst)):
        """if time_lst[i] + ACH_WINDOW > end_time:
            break"""
        ref_time_list = time_lst - time_lst[i]
        hist = ordered_array_histogram(ref_time_list, bins, i)
        ret[chunks_inv[i]] += hist
        counter[chunks_inv[i]] += 1

    ret = ret / (counter * ((bins[1] - bins[0])/1000))

    return ret


def calc_temporal_features(time_lst, chunks, resolution=2, bin_range=ACH_WINDOW, upsample=8, start_band_range=50,
                           mid_band_start=50, mid_band_end=ACH_WINDOW):
    time_lst = np.array(time_lst)

    assert (resolution > 0 and resolution % 1 == 0)
    assert (bin_range > 0 and bin_range % 1 == 0)

    N = 2 * resolution * bin_range + 2
    offset = 1 / (2 * resolution)
    bins = np.linspace(-bin_range - offset, bin_range + offset, N)
    start_time = time.time()
    histograms = calc_temporal_histogram(time_lst, bins, chunks)
    zero_bin_ind = histograms.shape[1] // 2
    histograms = (histograms[:, :zero_bin_ind + 1:][:, ::-1] + histograms[:, zero_bin_ind:]) / 2

    histograms = np.array([signal.resample_poly(histogram, upsample ** 2, upsample, padtype='line') for histogram in histograms])
    histograms = np.where(histograms >= 0, histograms, 0)
    end_time = time.time()
    if VERBOS:
        print(f"histogram creation took {end_time - start_time:.3f} seconds")
    start_band = histograms[:, :resolution * start_band_range * upsample]
    mid_band = histograms[:, resolution * mid_band_start * upsample: resolution * mid_band_end * upsample + 1]

    assert len(ind_features) == 1
    for feature in ind_features:
        feature_mat_for_cluster = feature.calculate_feature(time_lst, chunks)

    for feature in features:
        start_time = time.time()
        feature.set_fields(resolution=2 * upsample, start_band_range=start_band_range, mid_band_start=mid_band_start,
                           mid_band_end=mid_band_end)
        mat_result = feature.calculate_feature(rhs=histograms, start_band=start_band, mid_band=mid_band)

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
    for feature in ind_features:
        names += feature.headers
    for feature in features:
        names += feature.headers
    names += ['num_spikes']
    return names
