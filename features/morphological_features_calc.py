import numpy as np
from clusters import Spike

from features.morphological_features.FET_break import BreakMeasurement
from features.morphological_features.FET_fwhm import FWHM
from features.morphological_features.FET_max_speed import MaxSpeed
from features.morphological_features.FET_peak2peak import Peak2Peak
from features.morphological_features.FET_rise_coef import RiseCoef
from features.morphological_features.FET_smile_cry import SmileCry

# TODO: add get_acc feature


features = [BreakMeasurement(), FWHM(), MaxSpeed(), Peak2Peak(), RiseCoef(), SmileCry()]


def get_main_chnnels(chunks):
    ret = []

    for i, chunk in enumerate(chunks):
        chunk_data = chunk.get_data()
        main_channel = np.argmin(chunk_data) // chunk_data.shape[-1]
        ret.append(Spike(data=chunk_data[main_channel]))  # set main channel to be the one with highest depolariztion

    return ret


def calc_morphological_features(chunks):
    feature_mat_for_cluster = None
    main_chunks = get_main_chnnels(chunks)

    for feature in features:
        mat_result = feature.calculate_feature(main_chunks)
        if feature_mat_for_cluster is None:
            feature_mat_for_cluster = mat_result
        else:
            feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)

    return feature_mat_for_cluster


def get_morphological_features_names():
    # TODO: check if this works: names = [(name for name in feature.headers) for feature in features]
    names = []
    for feature in features:
        names += feature.headers
    return names
