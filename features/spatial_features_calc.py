import numpy as np
import time
from constants import VERBOS
import scipy.signal as signal
from scipy.optimize import curve_fit
from clusters import Spike

from features.spatial_features.FET_time_lag import TimeLagFeature
from features.spatial_features.FET_spd import SPD
from features.spatial_features.FET_da import DA
from features.spatial_features.FET_depolarization_graph import DepolarizationGraph
from features.spatial_features.FET_channel_contrast_feature import ChannelContrast
from features.spatial_features.FET_geometrical_estimation import GeometricalEstimation

pure_spatial_features = [SPD(), DA(), ChannelContrast()]
tempo_spatial_features = [TimeLagFeature(), DepolarizationGraph(), GeometricalEstimation()]


def wavelet_info(length, centers):
    w = np.empty((len(centers), length // 2), dtype='float32')
    for i, center in enumerate(centers):
        g = signal.windows.gaussian(length, 5) * -1
        w[i] = g[length // 2 - center: -1 - center]

    w_norm = w - np.repeat(np.expand_dims(w.mean(axis=1), axis=1), length // 2, axis=1)

    rss = np.sqrt((w_norm ** 2).sum(axis=1))

    return w, w_norm, rss


def match_spike(spike, w, w_norm, rss):
    mul_val = -spike.min()
    mul_val = spike.max() - spike.min()
    spike = spike - spike.mean()

    corrs = np.dot(w_norm, spike) / rss
    ind = corrs.argmax()
    wvlt = w[ind].copy() * mul_val

    return wvlt


def calc_window(a, b, n):
    b = int(b)
    g = a * signal.windows.gaussian(2 * n + 1, 5)
    g = g[n - b: -1 - b]
    return g


def sp_match_spike(channel, x_data, cons, w, w_norm, rss):
    spike = np.zeros(channel.shape)
    spike[channel.argmin()] = channel.min()
    return spike
    if not cons:
        popt, _ = curve_fit(cons_gaussian_function, x_data, channel,
                            bounds=([2 * min(-1, channel.min()), 0], [0, len(x_data)]),
                            p0=[min(-1, channel.min()), len(x_data) / 2], max_nfev=10_000)
        a, b = popt
        spike = calc_window(a, b, len(x_data))
    else:
        spike = match_spike(channel, w, w_norm, rss)
    return spike


def sp_match_chunk(chunk, cons, w, w_norm, rss):
    x_data = np.arange(chunk.data.shape[-1])
    ret = np.zeros(chunk.data.shape)
    for i, channel in enumerate(chunk.data):
        ret[i] = sp_match_spike(np.where(channel > 0, 0, channel), x_data, cons, w, w_norm, rss)

    chunk = Spike(data=ret / abs(ret.min()))
    
    return chunk

def sp_wavelet_transform(chunks, cons):
    w, w_norm, rss = wavelet_info(chunks[0].data.shape[-1] * 2 + 1, np.arange(chunks[0].data.shape[-1]))
    ret = []
    for chunk in chunks:
        ret.append(sp_match_chunk(chunk, cons, w, w_norm, rss))

    return ret

def cons_gaussian_function(x, a, b):
    return a * np.exp((-(x - b) ** 2) / 25)

def calc_spatial_features(chunks):
    feature_mat_for_cluster = None
    start_time = time.time()
    # print('Starting wavelet transformation...')
    # wavelets = sp_wavelet_transform(chunks, False)
    wavelets = sp_wavelet_transform(chunks, False)
    end_time = time.time()
    if VERBOS:
        print(f"wavelet transformation took {end_time - start_time:.3f} seconds")
    for feature in tempo_spatial_features:
        start_time = time.time()
        if isinstance(feature, GeometricalEstimation):
            mat_result = feature.calculate_feature(wavelets, chunks)  # calculates the features, returns a matrix
        else:
            mat_result = feature.calculate_feature(wavelets)  # calculates the features, returns a matrix
        if feature_mat_for_cluster is None:
            feature_mat_for_cluster = mat_result
        else:
            feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)
        end_time = time.time()

        if VERBOS:
            print(f"feature {feature.name} contains {mat_result.shape} values")
            print(f"feature {feature.name} processing took {end_time - start_time:.3f} seconds")

    for feature in pure_spatial_features:
        start_time = time.time()
        mat_result = feature.calculate_feature(chunks)  # calculates the features, returns a matrix
        if feature_mat_for_cluster is None:
            feature_mat_for_cluster = mat_result
        else:
            feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)
        end_time = time.time()

        if VERBOS:
            print(f"feature {feature.name} contains {mat_result.shape} values")
            print(f"feature {feature.name} processing took {end_time - start_time:.3f} seconds")

    return feature_mat_for_cluster


def get_spatial_features_names():
    names = []
    for feature in tempo_spatial_features + pure_spatial_features:
        names += feature.headers
    return names
