import numpy as np
import scipy.signal as signal
import time
from constants import MIN_TIME_LIST, VERBOS
import matplotlib.pyplot as plt

from features.temporal_features.FET_DKL import DKL
from features.temporal_features.FET_jump_index import Jump
from features.temporal_features.FET_PSD import PSD
from features.temporal_features.FET_rise_time import RiseTime
from features.temporal_features.FET_unif_dist import UnifDist

features = [DKL(), Jump(), PSD(), RiseTime(), UnifDist()]

def calc_temporal_histogram(time_lst, bins):
    """ret = np.zeros(len(bins) - 1)
    for i in range(len(time_lst)):
        hist, _ = np.histogram(time_lst - time_lst[i], bins=bins)
        ret += hist"""

    diffs = np.convolve(np.sort(time_lst), [1, -1], mode='valid')
    hist, _ = np.histogram(diffs, bins=bins)

    return hist

def calc_temporal_features(time_lst, resolution=2, bin_range=1500, upsample=8, cdf_range=30, jmp_min=50, jmp_max=1200):
    feature_mat_for_cluster = None

    if len(time_lst) < MIN_TIME_LIST:
        return np.zeros((1, len(get_temporal_features_names())))  # TODO: rethink the values returned here

    time_lst = np.array(time_lst)

    N = resolution * bin_range + 1
    bins = np.linspace(0, bin_range, N)
    start_time = time.time()
    histogram = calc_temporal_histogram(time_lst, bins)
    histogram = signal.resample(histogram, upsample * N)
    end_time = time.time()
    if VERBOS:
        print(f"histogram creation took {end_time - start_time:.3f} seconds")
    start_band = histogram[:resolution * cdf_range * upsample]
    start_cdf = np.cumsum(start_band) / np.sum(start_band)
    # TODO: do it only in jump if not required elsewhere
    mid_band = histogram[resolution * jmp_min * upsample: resolution * jmp_max * upsample + 1]

    """plt.plot(histogram)
    plt.show()
    plt.plot(start_cdf)
    plt.show()"""

    for feature in features:
        start_time = time.time()
        feature.set_fields(resolution=2, cdf_range=30, jmp_min=50, jmp_max=1200)
        mat_result = feature.calculate_feature(rhs=histogram, start_cdf=start_cdf, mid_band=mid_band)
        if feature_mat_for_cluster is None:
            feature_mat_for_cluster = mat_result
        else:
            feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)
        end_time = time.time()
        if VERBOS:
            print(f"feature {feature.name} processing took {end_time - start_time:.3f} seconds")

    return feature_mat_for_cluster

def get_temporal_features_names():
    # TODO: check if this works: names = [(name for name in feature.headers) for feature in features]
    names = []
    for feature in features:
        names += feature.headers
    return names
