import numpy as np
import time
from constants import VERBOS

from features.spatial_features.FET_time_lag import TimeLagFeature
from features.spatial_features.FET_spd import SPD
from features.spatial_features.FET_da import DA
from features.spatial_features.FET_depolarization_graph import DepolarizationGraph
from features.spatial_features.FET_channel_contrast_feature import ChannelContrast
from features.spatial_features.FET_geometrical_estimation import GeometricalEstimation

features = [TimeLagFeature(), SPD(), DA(), DepolarizationGraph(), ChannelContrast(), GeometricalEstimation()]

def match_spike(spike, wavelet):
    best_corr = 0
    wvlt = None
    for ts in range(len(spike)):
        tmp_wvlt = wavelet(ts)
        corr = np.corrcoef(spike, tmp_wvlt)[0, 1].abs()
        if corr >= best_corr:
            best_corr = corr
            wvlt = tmp_wvlt
    mul = wvlt.mean() / spike.mean()
    wvlt *= mul
    return wvlt

def match_chunk(chunk, wavelet):
    ret = np.zeros(chunk.shape)
    for i, channel in enumerate(chunk):
        ret[i] = match_spike(channel, wavelet)

    return ret

def wavelet_function(center, length):
    pass

def wavelet_transform(chunks):
    def wavelet(center): wavelet_function(center, chunks[0].shape[-1])
    ret = []
    for i, chunk in enumerate(chunks):
        ret.append(match_chunk(chunk, wavelet))

    return ret

def calc_spatial_features(chunks):
    feature_mat_for_cluster = None
    wavelets = wavelet_transform(chunks)
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
