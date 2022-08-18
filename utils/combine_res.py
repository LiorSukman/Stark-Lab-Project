import pandas as pd
import numpy as np

from constants import SPATIAL, feature_names_org
from constants import SPATIAL_R, feature_names_rich

# regular
PATH_C = '../ml/results_rf_290322_rich_fix_imp.csv'
PATH_RAW = '../ml/raw_imps_rf_290322_rich_fix_imp.npy'
PATH_0 = '../ml/results_rf_290322_fix_imp.csv'
DEST = '../ml/results_rf_combined.csv'

BASELINE_C = '../ml/results_rf_280722_rich_chance_balanced_fix_imp.csv'
BASELINE_RAW = '../ml/raw_imps_rf_280722_rich_chance_balanced_fix_imp.npy'
BASELINE_0 = '../ml/results_rf_100522_chance_balanced.csv'
DEST_BASELINE = '../ml/results_rf_combined_chance_balanced.csv'

# moments
DEST_MOMENTS = '../ml/results_rf_moments.csv'
DEST_MOMENTS_BASELINE = '../ml/results_rf_moments_chance_balanced.csv'

# events
DEST_EVENTS = '../ml/results_rf_events.csv'
DEST_EVENTS_BASELINE = '../ml/results_rf_events_chance_balanced.csv'

# small chunks
PATH_SPAT = '../ml/results_rf_290322_more_chunks_wo1.csv'
DEST_SPAT = '../ml/results_rf_spatial_combined.csv'

NUM_FETS = 34
NUM_MOMENTS = 5
NUM_MODALITIES = 3
NUM_CHUNKS = 8
NUM_EVENTS = 3
EVENTS = ['fzc', 'dep', 'szc']


def get_family_imp(inds, arr, all=False):
    if all:
        arr = np.nan_to_num(arr)
    arr_m = abs(arr[:, :, inds].sum(axis=2))
    fam_imps = np.asarray([a[~np.isnan(a)].mean() for a in arr_m])
    return fam_imps


if __name__ == "__main__":
    # regular
    """# actual
    imps = np.load(PATH_RAW)
    df_c = pd.read_csv(PATH_C, index_col=0)

    for i in range(NUM_FETS):
        inds = [i * (NUM_MOMENTS + 1) + j for j in range(NUM_MOMENTS + 1)]
        new_imp = get_family_imp(inds, imps)
        df_c[f'test feature new {i + 1}'] = new_imp

    drop = [f'test feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))] + [f'dev feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))]
    df_c = df_c.drop(columns=drop)
    mapper = {f'test feature new {i+1}': f'test feature {i+1}' for i in range(NUM_FETS)}
    df_c = df_c.rename(columns=mapper)

    df_0 = pd.read_csv(PATH_0, index_col=0)
    df_0 = df_0[df_0.chunk_size == 0]

    df_c[df_c.chunk_size == 0] = df_0

    df_c.to_csv(DEST)

    # baseline
    imps = np.load(BASELINE_RAW)
    df_c = pd.read_csv(BASELINE_C, index_col=0)

    for i in range(NUM_FETS):
        inds = [i * (NUM_MOMENTS + 1) + j for j in range(NUM_MOMENTS + 1)]
        new_imp = get_family_imp(inds, imps)
        df_c[f'test feature new {i + 1}'] = new_imp

    drop = [f'test feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))] + [f'dev feature {i + 1}' for i in
                                                                     range(NUM_FETS * (NUM_MOMENTS + 1))]
    df_c = df_c.drop(columns=drop)
    mapper = {f'test feature new {i + 1}': f'test feature {i + 1}' for i in range(NUM_FETS)}
    df_c = df_c.rename(columns=mapper)

    df_0 = pd.read_csv(BASELINE_0, index_col=0)
    df_0 = df_0[df_0.chunk_size == 0]

    df_c[df_c.chunk_size == 0] = df_0

    df_c.to_csv(DEST_BASELINE)"""

    """"# moments
    # actual
    imps = np.load(PATH_RAW)
    df_c = pd.read_csv(PATH_C, index_col=0)

    for i in range(NUM_MOMENTS + 1):
        inds = [j * (NUM_MOMENTS + 1) + i for j in range(NUM_FETS)]
        new_imp = get_family_imp(inds, imps, all=True)
        df_c[f'test feature new {i + 1}'] = new_imp

    drop = [f'test feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))] + [f'dev feature {i + 1}' for i in
                                                                                     range(
                                                                                         NUM_FETS * (NUM_MOMENTS + 1))]
    df_c = df_c.drop(columns=drop)
    mapper = {f'test feature new {i + 1}': f'test feature {i + 1}' for i in range(NUM_MOMENTS + 1)}
    df_c = df_c.rename(columns=mapper)

    df_c.to_csv(DEST_MOMENTS)

    # baseline
    imps = np.load(BASELINE_RAW)
    df_c = pd.read_csv(BASELINE_C, index_col=0)

    for i in range(NUM_MOMENTS + 1):
        inds = [j * (NUM_MOMENTS + 1) + i for j in range(NUM_FETS)]
        new_imp = get_family_imp(inds, imps, all=True)
        df_c[f'test feature new {i + 1}'] = new_imp

    drop = [f'test feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))] + [f'dev feature {i + 1}' for i in
                                                                                     range(
                                                                                         NUM_FETS * (NUM_MOMENTS + 1))]
    df_c = df_c.drop(columns=drop)
    mapper = {f'test feature new {i + 1}': f'test feature {i + 1}' for i in range(NUM_MOMENTS + 1)}
    df_c = df_c.rename(columns=mapper)

    df_c.to_csv(DEST_MOMENTS_BASELINE)

    # events
    # actual
    imps = np.load(PATH_RAW)
    df_c = pd.read_csv(PATH_C, index_col=0)

    spatial_inds = np.arange(imps.shape[0]).reshape((imps.shape[0] // NUM_CHUNKS, NUM_CHUNKS))[::NUM_MODALITIES].flatten()
    imps = imps[spatial_inds]
    df_c = df_c[df_c.modality == 'spatial']

    spatial_fet_names = [feature_names_rich[i] for i in SPATIAL_R[:-1]]

    for i, event in enumerate(EVENTS):
        inds = [spatial_fet_names.index(name) for name in spatial_fet_names if event in name]
        new_imp = get_family_imp(inds, imps)
        df_c[f'test feature new {i + 1}'] = new_imp

    drop = [f'test feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))] + [f'dev feature {i + 1}' for i in
                                                                                     range(
                                                                                         NUM_FETS * (NUM_MOMENTS + 1))]
    df_c = df_c.drop(columns=drop)
    mapper = {f'test feature new {i + 1}': f'test feature {i + 1}' for i in range(NUM_EVENTS)}
    df_c = df_c.rename(columns=mapper)

    df_c.to_csv(DEST_EVENTS)

    # baseline
    imps = np.load(BASELINE_RAW)
    df_c = pd.read_csv(BASELINE_C, index_col=0)
    df_c = df_c[df_c.modality == 'spatial']

    spatial_inds = np.arange(imps.shape[0]).reshape((imps.shape[0] // NUM_CHUNKS, NUM_CHUNKS))[::NUM_MODALITIES].flatten()
    imps = imps[spatial_inds]

    for i, event in enumerate(EVENTS):
        inds = [spatial_fet_names.index(name) for name in spatial_fet_names if event in name]
        new_imp = get_family_imp(inds, imps)
        df_c[f'test feature new {i + 1}'] = new_imp

    drop = [f'test feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))] + [f'dev feature {i + 1}' for i in
                                                                                     range(
                                                                                         NUM_FETS * (NUM_MOMENTS + 1))]
    df_c = df_c.drop(columns=drop)
    mapper = {f'test feature new {i + 1}': f'test feature {i + 1}' for i in range(NUM_EVENTS)}
    df_c = df_c.rename(columns=mapper)

    df_c.to_csv(DEST_EVENTS_BASELINE)"""

    # combine spatials
    df_a = pd.read_csv(PATH_0, index_col=0)
    df_b = pd.read_csv(PATH_SPAT, index_col=0)

    df_a = df_a[df_a.modality == 'spatial']
    df_b = df_b[df_b.chunk_size != 0]

    df_a = df_a.append(df_b).sort_values(by=['chunk_size'])

    df_a.to_csv(DEST_SPAT)
