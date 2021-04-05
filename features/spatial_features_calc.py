import numpy as np
import time
from constants import VERBOS
import scipy.signal as signal
from clusters import Spike

from features.spatial_features.FET_time_lag import TimeLagFeature
from features.spatial_features.FET_spd import SPD
from features.spatial_features.FET_da import DA
from features.spatial_features.FET_depolarization_graph import DepolarizationGraph
from features.spatial_features.FET_channel_contrast_feature import ChannelContrast
from features.spatial_features.FET_geometrical_estimation import GeometricalEstimation

features = [TimeLagFeature(), SPD(), DA(), DepolarizationGraph(), ChannelContrast(), GeometricalEstimation()]

"""def match_spike(spike, wavelet):
    best_corr = 0
    wvlt = None
    for ts in range(len(spike)):
        for std in np.arange(2, 32, 2):
            tmp_wvlt = wavelet(ts, std) * -1
            corr = np.corrcoef(spike, tmp_wvlt)[0, 1]
            if corr >= best_corr:
                best_corr = corr
                wvlt = tmp_wvlt

    if spike.mean() == 0:
        raise AssertionError
    mul = wvlt.mean() / spike.mean()
    wvlt *= mul
    return wvlt

def match_chunk(chunk, wavelet):
    ret = np.zeros(chunk.data.shape)
    for i, channel in enumerate(chunk.data):
        ret[i] = match_spike(channel, wavelet)

    chunk = Spike(data=ret)
    return chunk

def wavelet_function(center, std, length):
    w = signal.windows.gaussian(length, std)
    w = w[length // 2 - center: -1 - center]
    return w"""

def wavelet_info(length, stds, centers):
    w = np.empty((len(stds), len(centers), length // 2), dtype='float32')
    for i, std in enumerate(stds):
        for j, center in enumerate(centers):
            g = signal.windows.gaussian(length, std) * -1
            w[i, j] = g[length // 2 - center: -1 - center]

    w_norm = w - np.repeat(np.expand_dims(w.mean(axis=2), axis=2), length // 2, axis=2)

    rss = np.sqrt((w_norm ** 2).sum(axis=2))

    return w, w_norm, rss

"""def wavelet_transform(chunks):
    def wavelet(center, std): return wavelet_function(center, std, 2 * chunks[0].data.shape[-1] + 1)
    ret = []
    for i, chunk in enumerate(chunks):
        ret.append(match_chunk(chunk, wavelet))

    return ret"""

def match_spike(spike, w, w_norm, rss):
    spike = spike - spike.mean()

    corrs = np.dot(w_norm, spike) / rss
    ind = corrs.argmax()
    i, j = ind // len(spike), ind % len(spike)
    wvlt = w[i, j].copy()

    mul = 1 / spike.max()  # wvlt.max() == 1
    wvlt *= mul
    return wvlt

def match_chunk(chunk, w, w_norm, rss):
    ret = np.zeros(chunk.data.shape)
    for i, channel in enumerate(chunk.data):
        ret[i] = match_spike(channel, w, w_norm, rss)

    chunk = Spike(data=ret)
    return chunk

def wavelet_transform(chunks):
    w, w_norm, rss = wavelet_info(chunks[0].data.shape[-1] * 2 + 1, np.arange(2, 32, 2),
                                  np.arange(chunks[0].data.shape[-1]))
    ret = []
    for i, chunk in enumerate(chunks):
        ret.append(match_chunk(chunk, w, w_norm, rss))

    return ret

def calc_spatial_features(chunks):
    feature_mat_for_cluster = None
    start_time = time.time()
    print('Starting wavelet transformation...')
    wavelets = wavelet_transform(chunks)
    end_time = time.time()
    if VERBOS:
        print(f"wavelet transformation took {end_time - start_time:.3f} seconds")
    for feature in features:
        start_time = time.time()
        mat_result = feature.calculate_feature(wavelets)  # calculates the features, returns a matrix
        if feature_mat_for_cluster is None:
            feature_mat_for_cluster = mat_result
        else:
            feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)
        end_time = time.time()
        if VERBOS:
            print(f"feature {feature.name} processing took {end_time - start_time:.3f} seconds")

    return feature_mat_for_cluster

def get_spatial_features_names():
    # TODO: check if this works: names = [(name for name in feature.headers) for feature in features]
    names = []
    for feature in features:
        names += feature.headers
    return names
