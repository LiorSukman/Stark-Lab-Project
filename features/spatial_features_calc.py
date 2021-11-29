import numpy as np
import time
from enum import Enum
from constants import VERBOS
from clusters import Spike


from features.spatial_features.FET_time_lag import TimeLagFeature
from features.spatial_features.FET_spd import SPD
from features.spatial_features.FET_depolarization_graph import DepolarizationGraph
from features.spatial_features.FET_geometrical_estimation import GeometricalEstimation

dep_spatial_features = [SPD(), GeometricalEstimation(), DepolarizationGraph()]
full_spatial_features = [TimeLagFeature()]

class DELTA_MODE(Enum):
    DEP = 1
    F_ZCROSS = 2
    S_ZCROSS = 3


def calc_pos(arr, start_pos, mode, debug=False):
    assert mode in [DELTA_MODE.F_ZCROSS, DELTA_MODE.S_ZCROSS]
    pos = start_pos
    if debug:
        print(arr.shape)
    while (pos >= 0) if mode == DELTA_MODE.F_ZCROSS else (pos < len(arr)):
        if debug:
            print(pos)
        if arr[pos] == 0:
            if mode == DELTA_MODE.F_ZCROSS:
                pos -= 1
            else:
                pos += 1
        else:
            break
    return pos

def match_spike(channel, cons, med):
    pos = channel.argmin()
    spike = np.zeros(channel.shape)
    if cons == DELTA_MODE.DEP:
        spike[pos] = channel.min()
    elif cons in [DELTA_MODE.F_ZCROSS, DELTA_MODE.S_ZCROSS]:
        sig_m = np.convolve(np.where(channel <= med, -1, 1), [-1, 1], 'same')
        sig_m[0] = sig_m[-1] = 1
        pos = calc_pos(sig_m, pos, cons)
        if pos == 256:
            print(sig_m)
            print(channel.shape)
            pos = calc_pos(sig_m, channel.argmin(), cons, True)
        spike[pos] = channel.min()
    else:
        raise KeyError("cons paremeter is not valid")
    return spike, pos


def match_chunk(chunk, cons, amp):
    ret = np.zeros(chunk.data.shape)
    main_c = amp.argmax()
    roll_val = 0
    med = np.median(chunk.data)
    for i, channel in enumerate(chunk.data):
        ret[i], pos = match_spike(channel, cons, med)
        if i == main_c:
            roll_val = chunk.data.shape[-1] // 2 - pos

    ret = np.roll(ret, roll_val, axis=1)
    chunk = Spike(data=ret / abs(ret.min()))
    
    return chunk

def wavelet_transform(chunks, cons, amps):
    ret = []
    for chunk, amp in zip(chunks, amps):
        ret.append(match_chunk(chunk, cons, amp))

    return ret


def calc_amps(chunks):
    ret = []
    for chunk in chunks:
        camps = chunk.data.max(axis=1) - chunk.data.min(axis=1)
        ret.append(camps / camps.max())

    return np.array(ret)

def calc_spatial_features(chunks):
    feature_mat_for_cluster = None
    start_time = time.time()
    # print('Starting wavelet transformation...')
    amps = calc_amps(chunks)

    wavelets_dep = wavelet_transform(chunks, DELTA_MODE.DEP, amps)
    wavelets_fzc = wavelet_transform(chunks, DELTA_MODE.F_ZCROSS, amps)
    wavelets_szc = wavelet_transform(chunks, DELTA_MODE.S_ZCROSS, amps)

    end_time = time.time()
    if VERBOS:
        print(f"wavelet transformation took {end_time - start_time:.3f} seconds")
    for feature in dep_spatial_features:
        start_time = time.time()
        mat_result = feature.calculate_feature(wavelets_dep)  # calculates the features, returns a matrix
        if feature_mat_for_cluster is None:
            feature_mat_for_cluster = mat_result
        else:
            feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)
        end_time = time.time()

        if VERBOS:
            print(f"feature {feature.name} contains {mat_result.shape} values")
            print(f"feature {feature.name} processing took {end_time - start_time:.3f} seconds")

    for feature in full_spatial_features:
        start_time = time.time()
        for data, dtype in zip([wavelets_dep, wavelets_fzc, wavelets_szc], ['dep', ' fzc', 'szc']):
            feature.set_data(dtype)
            mat_result = feature.calculate_feature(data, amps)  # calculates the features, returns a matrix
            if feature_mat_for_cluster is None:
                feature_mat_for_cluster = mat_result
            else:
                feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)
            end_time = time.time()

        if VERBOS:
            print(f"feature {feature.name} (data: {dtype}) contains {mat_result.shape} values")
            print(f"feature {feature.name} processing took {end_time - start_time:.3f} seconds")

    return feature_mat_for_cluster


def get_spatial_features_names():
    names = []
    for feature in dep_spatial_features:
        names += feature.headers
    for feature in full_spatial_features:
        for dtype in ['dep', ' fzc', 'szc']:
            feature.set_data(dtype)
            names += feature.headers
    return names
