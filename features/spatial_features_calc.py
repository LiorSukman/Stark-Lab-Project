import numpy as np
import time
from enum import Enum
from constants import VERBOS
from clusters import Spike


from features.spatial_features.FET_time_lag import TimeLagFeature
from features.spatial_features.FET_spd import SPD
from features.spatial_features.FET_depolarization_graph import DepolarizationGraph
from features.spatial_features.spatial_FWHM import SPAT_FWHM

dep_spatial_features = [SPD()]
fwhm_spatial = [SPAT_FWHM()]
full_spatial_features = [TimeLagFeature(), DepolarizationGraph()]

class DELTA_MODE(Enum):
    DEP = 1
    F_ZCROSS = 2
    S_ZCROSS = 3


def calc_pos(arr, start_pos, mode):
    assert mode in [DELTA_MODE.F_ZCROSS, DELTA_MODE.S_ZCROSS]
    pos = start_pos
    while (pos >= 0) if mode == DELTA_MODE.F_ZCROSS else (pos < len(arr)):
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
        spike[pos] = channel.min()
    else:
        raise KeyError("cons paremeter is not valid")
    return spike, pos


def match_chunk(chunk, cons, amp):
    ret = np.zeros(chunk.data.shape)
    main_c = amp.argmax()
    roll_val = 0
    med = np.median(chunk.data) # first version global
    for i, channel in enumerate(chunk.data):
        #med = np.median(chunk.data[i]) second version local
        #med = chunk.data[i].min() * 0.5 third version local fractional
        ret[i], pos = match_spike(channel, cons, med)
        if i == main_c:
            roll_val = chunk.data.shape[-1] // 2 - pos

    ret = np.roll(ret, roll_val, axis=1)
    chunk = Spike(data=ret / abs(ret.min()))
    
    return chunk


def match_chunk_cont(chunk, cons, amp, q):
    ret = np.zeros(chunk.data.shape)
    main_c = amp.argmax()
    roll_val = 0
    for i, channel in enumerate(chunk.data):
        #quantile = np.quantile(chunk.data[i], q)
        quantile = chunk.data[i].min() * q
        ret[i], pos = match_spike(channel, cons, quantile)
        if i == main_c:
            roll_val = chunk.data.shape[-1] // 2 - pos

    ret = np.roll(ret, roll_val, axis=1)
    chunk = Spike(data=ret / abs(ret.min()))

    return chunk

def wavelet_transform(chunks, cons, amps, q=None):
    ret = []
    if q is None:
        for chunk, amp in zip(chunks, amps):
            ret.append(match_chunk(chunk, cons, amp))
    else:
        for chunk, amp in zip(chunks, amps):
            ret.append(match_chunk_cont(chunk, cons, amp, q))

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

    """
    for feature in fwhm_spatial:
        mat_result = feature.calculate_feature(chunks, amps)  # calculates the features, returns a matrix
        if feature_mat_for_cluster is None:
            feature_mat_for_cluster = mat_result
        else:
            feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)
    """

    for feature in full_spatial_features:
        start_time = time.time()
        for data, dtype in zip([wavelets_dep, wavelets_fzc, wavelets_szc], ['dep', 'fzc', 'szc']):
            feature.set_data(dtype)
            mat_result = feature.calculate_feature(data, amps, chunks)  # calculates the features, returns a matrix
            if feature_mat_for_cluster is None:
                feature_mat_for_cluster = mat_result
            else:
                feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)
            end_time = time.time()

        if VERBOS:
            print(f"feature {feature.name} (data: {dtype}) contains {mat_result.shape} values")
            print(f"feature {feature.name} processing took {end_time - start_time:.3f} seconds")

    return feature_mat_for_cluster

def calc_spatial_cont(chunks):
    amps = calc_amps(chunks)
    feature = TimeLagFeature()

    wavelets_dep = wavelet_transform(chunks, DELTA_MODE.DEP, amps)
    feature_mat_for_cluster = feature.calculate_feature(wavelets_dep, amps)

    for mode in [DELTA_MODE.F_ZCROSS, DELTA_MODE.S_ZCROSS]:
        for q in np.linspace(0.05, 0.95, 19):
            wavelets = wavelet_transform(chunks, mode, amps, q)
            mat_result = feature.calculate_feature(wavelets, amps)  # calculates the features, returns a matrix
            feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)

    return feature_mat_for_cluster


def get_spatial_features_names():
    names = []
    for feature in dep_spatial_features:
        names += feature.headers
    # for feature in fwhm_spatial:
    #     names += feature.headers
    for feature in full_spatial_features:
        for dtype in ['dep', 'fzc', 'szc']:
            feature.set_data(dtype)
            names += feature.headers
    return names

def get_spatial_cont_names():
    names = ['dep_red', 'dep_sd']
    for mode in ['FMC', 'SMC']:
        for q in np.linspace(0.05, 0.95, 19):
            names += [f'{mode}_{q: .2g}_red', f'{mode}_{q: .2g}_sd']
    return names

