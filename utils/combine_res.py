import pandas as pd
import numpy as np
import glob

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

# spatial families
DEST_SPAT_FAM = '../ml/results_rf_families.csv'

# combine spatials - small chunks
PATH_SPAT = '../ml/results_rf_290322_more_chunks_wo1.csv'
DEST_SPAT = '../ml/results_rf_spatial_combined.csv'

# regular but for chunking V2
PATH_SPAT_V2 = '../ml/results_rf_110822_rich_v2.csv'
PATH_WF_ST_V2 = '../ml/results_rf_160822_rich_v2_wf_st.csv'
DEST_V2 = '../ml/results_rf_combined_v2.csv'

# region CA1
PATH_REGION = '../ml/results_rf_080922_rich_region_imp.csv'
PATH_RAW_CA1 = '../ml/raw_imps_dev_rf_080922_rich_region_imp.npy'
DEST_REGION = '../ml/results_rf_region_ca1.csv'

# region nCX
PATH_REGION_NCX = '../ml/results_rf_110922_rich_region_imp_ncx.csv'
PATH_RAW_NCX = '../ml/raw_imps_rf_110922_rich_region_imp_ncx.npy'
DEST_REGION_NCX = '../ml/results_rf_region_ncx.csv'

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


def combine_csvs(n_iter=25, seed_per_iter=40, orig_modifier='imps', dest_comb='060922_shuffles_combined'):
    df = None
    for n in range(n_iter):
        path = f'../ml/shuffle_results/060922_shuffles_{n * seed_per_iter}_{(n + 1) * seed_per_iter}/' \
               f'results_rf_060922_shuffles_{n * seed_per_iter}_{(n + 1) * seed_per_iter}_{orig_modifier}.csv'
        df_temp = pd.read_csv(path, index_col=0)
        if df is None:
            df = df_temp
        else:
            df = df.append(df_temp)

    df.to_csv(f'../ml/shuffle_results/{dest_comb}.csv')


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

    df_c.to_csv(DEST_EVENTS_BASELINE)

    # spatial families
    imps = np.load(PATH_RAW)
    df_c = pd.read_csv(PATH_C, index_col=0)

    spatial_inds = np.arange(imps.shape[0]).reshape((imps.shape[0] // NUM_CHUNKS, NUM_CHUNKS))[::NUM_MODALITIES].flatten()
    imps = imps[spatial_inds]
    df_c = df_c[df_c.modality == 'spatial']

    spatial_families_temp = {
        'value-based': ['spatial_dispersion_count', 'spatial_dispersion_sd', 'spatial_dispersion_area'],
        'time-based': ['dep_red', 'dep_sd', 'fzc_red', 'fzc_sd', 'szc_red', 'szc_sd'],
        'graph-based': ['dep_graph_avg_speed', 'dep_graph_slowest_path', 'dep_graph_fastest_path',
                        'fzc_graph_avg_speed', 'fzc_graph_slowest_path', 'fzc_graph_fastest_path',
                        'szc_graph_avg_speed', 'szc_graph_slowest_path', 'szc_graph_fastest_path']}

    spatial_families = dict()
    for key in spatial_families_temp:
        temp_list = []
        for f in spatial_families_temp[key]:
            temp_list += [f'{f}', f'{f}_avg', f'{f}_std', f'{f}_q25', f'{f}_q50', f'{f}_q75']
        spatial_families[key] = temp_list

    spatial_fet_names = [feature_names_rich[i] for i in SPATIAL_R[:-1]]

    for i, fam in enumerate(spatial_families):
        inds = [spatial_fet_names.index(name) for name in spatial_families[fam]]
        new_imp = get_family_imp(inds, imps)
        df_c[f'test feature new {i + 1}'] = new_imp

    drop = [f'test feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))] + [f'dev feature {i + 1}' for i in
                                                                                     range(
                                                                                         NUM_FETS * (NUM_MOMENTS + 1))]
    df_c = df_c.drop(columns=drop)
    mapper = {f'test feature new {i + 1}': f'test feature {i + 1}' for i in range(NUM_EVENTS)}
    df_c = df_c.rename(columns=mapper)

    df_c.to_csv(DEST_SPAT_FAM)

    # combine spatials
    df_a = pd.read_csv(PATH_0, index_col=0)
    df_b = pd.read_csv(PATH_SPAT, index_col=0)

    df_a = df_a[df_a.modality == 'spatial']
    df_b = df_b[df_b.chunk_size != 0]

    df_a = df_a.append(df_b).sort_values(by=['chunk_size'])

    df_a.to_csv(DEST_SPAT)

    # combining chunking V2
    df_a = pd.read_csv(PATH_SPAT_V2, index_col=0)
    df_b = pd.read_csv(PATH_WF_ST_V2, index_col=0)
    df_b_wf = df_b[df_b.modality == 'morphological']
    df_b_st = df_b[df_b.modality == 'temporal']

    df_a = df_a[df_a.chunk_size != 0]
    df_b_wf = df_b_wf[df_b_wf.chunk_size != 0]
    df_b_st = df_b_st[df_b_st.chunk_size != 0]

    df_0 = pd.read_csv(PATH_0, index_col=0)
    df_0 = df_0[df_0.chunk_size == 0]
    df_0_spat = df_0[df_0.modality == 'spatial']
    df_0_wf = df_0[df_0.modality == 'morphological']
    df_0_st = df_0[df_0.modality == 'temporal']

    df_a = df_a.append(df_0_spat, ignore_index=True)
    df_b_wf = df_b_wf.append(df_0_wf, ignore_index=True)
    df_b_st = df_b_st.append(df_0_st, ignore_index=True)

    df = df_a.append((df_b_wf, df_b_st), ignore_index=True)

    df.to_csv(DEST_V2)

    # region CA1
    imps = np.load(PATH_RAW_CA1)
    df_c = pd.read_csv(PATH_REGION, index_col=0)

    for i in range(NUM_FETS):
        inds = [i * (NUM_MOMENTS + 1) + j for j in range(NUM_MOMENTS + 1)]
        new_imp = get_family_imp(inds, imps)
        df_c[f'test feature new {i + 1}'] = new_imp

    drop = [f'test feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))] + [f'dev feature {i + 1}' for i in
                                                                                     range(
                                                                                         NUM_FETS * (NUM_MOMENTS + 1))]
    df_c = df_c.drop(columns=drop)
    mapper = {f'test feature new {i + 1}': f'test feature {i + 1}' for i in range(NUM_FETS)}
    df_c = df_c.rename(columns=mapper)

    df_c.to_csv(DEST_REGION)

    # region nCX
    imps = np.load(PATH_RAW_NCX)
    df_c = pd.read_csv(PATH_REGION_NCX, index_col=0)

    for i in range(NUM_FETS):
        inds = [i * (NUM_MOMENTS + 1) + j for j in range(NUM_MOMENTS + 1)]
        new_imp = get_family_imp(inds, imps)
        df_c[f'test feature new {i + 1}'] = new_imp

    drop = [f'test feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))] + [f'dev feature {i + 1}' for i in
                                                                                     range(
                                                                                         NUM_FETS * (NUM_MOMENTS + 1))]
    df_c = df_c.drop(columns=drop)
    mapper = {f'test feature new {i + 1}': f'test feature {i + 1}' for i in range(NUM_FETS)}
    df_c = df_c.rename(columns=mapper)

    df_c.to_csv(DEST_REGION_NCX)

    # shuffle fix slow run
    root_dir = '../ml/shuffle_results/'
    a = root_dir + '060922_shuffles_640_680/results_rf_060922_shuffles_640_680_659.csv'
    b = root_dir + '060922_shuffles_660_680/results_rf_060922_shuffles_660_680.csv'
    a_df = pd.read_csv(a, index_col=0)
    b_df = pd.read_csv(b, index_col=0)
    join_df = a_df.append(b_df)
    join_df.to_csv(root_dir + '060922_shuffles_640_680/results_rf_060922_shuffles_640_680.csv')

    a = root_dir + '060922_shuffles_640_680/raw_imps_rf_060922_shuffles_640_680_659.npy'
    b = root_dir + '060922_shuffles_660_680/raw_imps_rf_060922_shuffles_660_680.npy'
    a_np = np.load(a)
    b_np = np.load(b)
    join_np = np.concatenate((a_np, b_np))
    np.save(root_dir + '060922_shuffles_640_680/raw_imps_rf_060922_shuffles_640_680.npy', join_np)

    a = root_dir + '060922_shuffles_640_680/preds_rf_060922_shuffles_640_680_659.npy'
    b = root_dir + '060922_shuffles_660_680/preds_rf_060922_shuffles_660_680.npy'
    a_np = np.load(a)
    b_np = np.load(b)
    join_np = np.concatenate((a_np, b_np))
    np.save(root_dir + '060922_shuffles_640_680/preds_rf_060922_shuffles_640_680.npy', join_np)"""

    # shuffle fix imps (features)
    """for n in range(25):
        path_csv = f'../ml/shuffle_results/060922_shuffles_{n * 40}_{(n + 1) * 40}/results_rf_060922_shuffles_{n * 40}_{(n + 1) * 40}.csv'
        path_npy = f'../ml/shuffle_results/060922_shuffles_{n * 40}_{(n + 1) * 40}/raw_imps_rf_060922_shuffles_{n * 40}_{(n + 1) * 40}.npy'
        df_temp = pd.read_csv(path_csv, index_col=0)
        imps_temp = np.load(path_npy)

        for i in range(NUM_FETS):
            inds = [i * (NUM_MOMENTS + 1) + j for j in range(NUM_MOMENTS + 1)]
            new_imp = get_family_imp(inds, imps_temp)
            df_temp[f'test feature new {i + 1}'] = new_imp

        drop = [f'test feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))] + [f'dev feature {i + 1}' for i in
                                                                                         range(NUM_FETS * (
                                                                                                 NUM_MOMENTS + 1))]
        df_temp = df_temp.drop(columns=drop)
        mapper = {f'test feature new {i + 1}': f'test feature {i + 1}' for i in range(NUM_FETS)}
        df_temp = df_temp.rename(columns=mapper)

        df_temp.to_csv(
            f'../ml/shuffle_results/060922_shuffles_{n * 40}_{(n + 1) * 40}/results_rf_060922_shuffles_{n * 40}_{(n + 1) * 40}_imps.csv')
            
        # shuffle combine all (features)
        combine_csvs(orig_modifier='imps', dest_comb='060922_shuffles_combined')

    # shuffle fix imps (moments)
    for n in range(25):
        print(f'moments: iteration {n}')
        path_csv = f'../ml/shuffle_results/060922_shuffles_{n * 40}_{(n + 1) * 40}/results_rf_060922_shuffles_{n * 40}_{(n + 1) * 40}.csv'
        path_npy = f'../ml/shuffle_results/060922_shuffles_{n * 40}_{(n + 1) * 40}/raw_imps_rf_060922_shuffles_{n * 40}_{(n + 1) * 40}.npy'
        df_temp = pd.read_csv(path_csv, index_col=0)
        imps_temp = np.load(path_npy)

        for i in range(NUM_MOMENTS + 1):
            inds = [j * (NUM_MOMENTS + 1) + i for j in range(NUM_FETS)]
            new_imp = get_family_imp(inds, imps_temp, all=True)
            df_temp[f'test feature new {i + 1}'] = new_imp

        drop = [f'test feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))] + \
               [f'dev feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))]
        df_temp = df_temp.drop(columns=drop)
        mapper = {f'test feature new {i + 1}': f'test feature {i + 1}' for i in range(NUM_MOMENTS + 1)}
        df_temp = df_temp.rename(columns=mapper)

        df_temp.to_csv(
            f'../ml/shuffle_results/060922_shuffles_{n * 40}_{(n + 1) * 40}/results_rf_060922_shuffles_{n * 40}_{(n + 1) * 40}_moments.csv')

    # shuffle combine all (moments)
    combine_csvs(orig_modifier='moments', dest_comb='060922_shuffles_moments')

    # shuffle fix imps (events)
    for n in range(25):
        print(f'events: iteration {n}')
        path_csv = f'../ml/shuffle_results/060922_shuffles_{n * 40}_{(n + 1) * 40}/results_rf_060922_shuffles_{n * 40}_{(n + 1) * 40}.csv'
        path_npy = f'../ml/shuffle_results/060922_shuffles_{n * 40}_{(n + 1) * 40}/raw_imps_rf_060922_shuffles_{n * 40}_{(n + 1) * 40}.npy'
        df_temp = pd.read_csv(path_csv, index_col=0)
        imps_temp = np.load(path_npy)

        spatial_inds = np.arange(imps_temp.shape[0]).reshape((imps_temp.shape[0] // 2, 2))[::3].flatten()
        imps_temp = imps_temp[spatial_inds]
        df_temp = df_temp[df_temp.modality == 'spatial']
        spatial_fet_names = [feature_names_rich[i] for i in SPATIAL_R[:-1]]

        for i, event in enumerate(EVENTS):
            inds = [spatial_fet_names.index(name) for name in spatial_fet_names if event in name]
            new_imp = get_family_imp(inds, imps_temp)
            df_temp[f'test feature new {i + 1}'] = new_imp

        drop = [f'test feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))] \
               + [f'dev feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))]
        df_temp = df_temp.drop(columns=drop)
        mapper = {f'test feature new {i + 1}': f'test feature {i + 1}' for i in range(NUM_EVENTS)}
        df_temp = df_temp.rename(columns=mapper)

        df_temp.to_csv(
            f'../ml/shuffle_results/060922_shuffles_{n * 40}_{(n + 1) * 40}/results_rf_060922_shuffles_{n * 40}_{(n + 1) * 40}_events.csv')

    # shuffle combine all (events)
    combine_csvs(orig_modifier='events', dest_comb='060922_shuffles_events')"""

    # shuffle fix imps (spatial families)
    spatial_families_temp = {
        'value-based': ['spatial_dispersion_count', 'spatial_dispersion_sd', 'spatial_dispersion_area'],
        'time-based': ['dep_red', 'dep_sd', 'fzc_red', 'fzc_sd', 'szc_red', 'szc_sd'],
        'graph-based': ['dep_graph_avg_speed', 'dep_graph_slowest_path', 'dep_graph_fastest_path',
                        'fzc_graph_avg_speed', 'fzc_graph_slowest_path', 'fzc_graph_fastest_path',
                        'szc_graph_avg_speed', 'szc_graph_slowest_path', 'szc_graph_fastest_path']}

    spatial_families = dict()
    for key in spatial_families_temp:
        temp_list = []
        for f in spatial_families_temp[key]:
            temp_list += [f'{f}', f'{f}_avg', f'{f}_std', f'{f}_q25', f'{f}_q50', f'{f}_q75']
        spatial_families[key] = temp_list

    spatial_fet_names = [feature_names_rich[i] for i in SPATIAL_R[:-1]]

    for n in range(25):
        print(f'families: iteration {n}')
        path_csv = f'../ml/shuffle_results/060922_shuffles_{n * 40}_{(n + 1) * 40}/results_rf_060922_shuffles_{n * 40}_{(n + 1) * 40}.csv'
        path_npy = f'../ml/shuffle_results/060922_shuffles_{n * 40}_{(n + 1) * 40}/raw_imps_rf_060922_shuffles_{n * 40}_{(n + 1) * 40}.npy'
        df_temp = pd.read_csv(path_csv, index_col=0)
        imps_temp = np.load(path_npy)

        spatial_inds = np.arange(imps_temp.shape[0]).reshape((imps_temp.shape[0] // 2, 2))[::3].flatten()
        imps_temp = imps_temp[spatial_inds]
        df_temp = df_temp[df_temp.modality == 'spatial']
        spatial_fet_names = [feature_names_rich[i] for i in SPATIAL_R[:-1]]

        for i, fam in enumerate(spatial_families):
            inds = [spatial_fet_names.index(name) for name in spatial_families[fam]]
            new_imp = get_family_imp(inds, imps_temp)
            df_temp[f'test feature new {i + 1}'] = new_imp

        drop = [f'test feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))] \
               + [f'dev feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))]
        df_temp = df_temp.drop(columns=drop)
        mapper = {f'test feature new {i + 1}': f'test feature {i + 1}' for i in range(NUM_EVENTS)}
        df_temp = df_temp.rename(columns=mapper)

        df_temp.to_csv(
            f'../ml/shuffle_results/060922_shuffles_{n * 40}_{(n + 1) * 40}/results_rf_060922_shuffles_{n * 40}_{(n + 1) * 40}_families.csv')

    # shuffle combine all (events)
    combine_csvs(orig_modifier='families', dest_comb='060922_shuffles_families')
