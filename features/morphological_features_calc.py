import numpy as np
from clusters import Spike
import time
from constants import VERBOS
from features.spatial_features_calc import wavelet_transform, calc_amps
import matplotlib.pyplot as plt

from features.morphological_features.FET_break import BreakMeasurement
from features.morphological_features.FET_fwhm import FWHM
from features.morphological_features.FET_get_acc import GetAcc
from features.morphological_features.FET_max_speed import MaxSpeed
from features.morphological_features.FET_peak2peak import Peak2Peak
from features.morphological_features.FET_rise_coef import RiseCoef
from features.morphological_features.FET_smile_cry import SmileCry


features = [BreakMeasurement(), FWHM(), GetAcc(), MaxSpeed(), Peak2Peak(), RiseCoef(), SmileCry()]


def get_main_chnnels(chunks):
    ret = []

    for i, chunk in enumerate(chunks):
        chunk_data = chunk.get_data()
        chunk_amp = chunk_data.max(axis=1) - chunk_data.min(axis=1)
        main_channel = np.argmax(chunk_amp)
        plt.plot(chunk_data[main_channel])
        data = chunk_data[main_channel].copy() / abs(chunk_data[main_channel].min())  # normalize
        ret.append(Spike(data=data))  # set main channel to be the one with highest peak - trough

    return ret


def get_trough_time(main_chunks):
    result = [[spike.data.argmin()] for spike in main_chunks]
    result = np.asarray(result)
    return result


def calc_morphological_features(chunks, transform=None):
    feature_mat_for_cluster = None

    if transform is not None:
        chunks = wavelet_transform(chunks, transform, calc_amps(chunks))

    main_chunks = get_main_chnnels(chunks)

    for feature in features:
        start_time = time.time()
        mat_result = feature.calculate_feature(main_chunks)
        if feature_mat_for_cluster is None:
            feature_mat_for_cluster = mat_result
        else:
            feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)
        end_time = time.time()
        if VERBOS:
            print(f"feature {feature.name} processing took {end_time - start_time:.3f} seconds")

    mat_result = get_trough_time(main_chunks)
    feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)

    return feature_mat_for_cluster


def get_morphological_features_names():
    names = []
    for feature in features:
        names += feature.headers
    return names + ['t_time']
