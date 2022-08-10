import ML_util

import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import shap

from gs_rf import grid_search as grid_search_rf
from RF import run as run_model

from constants import SPATIAL, MORPHOLOGICAL, TEMPORAL, TRANS_MORPH
from constants import SPATIAL_R, MORPHOLOGICAL_R, TEMPORAL_R
from utils.hideen_prints import HiddenPrints
from constants import INF

chunks = [0, 25, 50, 1600] #[0, 25, 50, 100, 200, 400, 800, 1600]
restrictions = ['complete']
dataset_identifier = '0.800.2'

NUM_FETS = 204

n_estimators_min = 0
n_estimators_max = 2
n_estimators_num = 3
max_depth_min = 1
max_depth_max = 2
max_depth_num = 2
min_samples_splits_min = 1
min_samples_splits_max = 5
min_samples_splits_num = 5
min_samples_leafs_min = 0
min_samples_leafs_max = 5
min_samples_leafs_num = 6

n = 5


def get_test_set(data_path, region_based=False, get_dev=False):
    train, dev, test, _, _, _ = ML_util.get_dataset(data_path)

    train_squeezed = ML_util.squeeze_clusters(train)
    dev_squeezed = ML_util.squeeze_clusters(dev)
    if (not region_based) and len(dev_squeezed) > 0:
        train_data = np.concatenate((train_squeezed, dev_squeezed))
    else:
        train_data = train_squeezed
    features, labels = ML_util.split_features(train_data)
    features = np.nan_to_num(features)
    features = np.clip(features, -INF, INF)

    scaler = StandardScaler()
    scaler.fit(features)

    test_set = test if not get_dev else dev
    if len(test_set) == 0:
        return [], []

    test_squeezed = ML_util.squeeze_clusters(test_set)
    x, y = ML_util.split_features(test_squeezed)
    x = np.nan_to_num(x)
    x = np.clip(x, -INF, INF)
    x = scaler.transform(x)

    return x, y


def get_shap_imp(clf, test, seed, skip=False):
    if skip:
        a, b = np.zeros((test.shape[-1])), np.zeros((min(1000, len(test)), test.shape[-1]))
        return a, b
    df = pd.DataFrame(test)
    df_shap = df.sample(min(1000, len(test)), random_state=int(seed) + 1)
    rf_explainer = shap.TreeExplainer(clf)  # define explainer
    shap_values = rf_explainer(df_shap)  # calculate shap values
    pyr_shap_values = shap_values[..., 1]
    return np.mean(np.abs(pyr_shap_values.values), axis=0), pyr_shap_values.values


def thr_preds(probs, thr):
    return (probs >= thr).astype('int8')


def get_predictions(clf, train, dev, test, region_based, use_dev):
    if not region_based:
        train = np.concatenate((train, dev))
    train_squeezed = ML_util.squeeze_clusters(train)
    train_features, train_labels = ML_util.split_features(train_squeezed)
    train_features = np.nan_to_num(train_features)
    train_features = np.clip(train_features, -INF, INF)

    scaler = StandardScaler()
    scaler.fit(train_features)

    preds = []

    test_set = test if not use_dev else dev
    if len(test_set) == 0:
        return None

    for cluster in test_set:
        features, labels = ML_util.split_features(cluster)
        features = np.nan_to_num(features)
        features = np.clip(features, -INF, INF)
        features = scaler.transform(features)

        prob = clf.predict_proba(features).mean(axis=0)
        pred = prob[1]

        preds.append(pred)

    return np.asarray(preds)


def get_preds(clf, data_path, region_based=False):
    train, dev, test, _, _, _ = ML_util.get_dataset(data_path)

    preds = get_predictions(clf, train, dev, test, region_based, False)
    assert preds is not None
    dev_preds = get_predictions(clf, train, dev, test, region_based, True)
    if dev_preds is None:
        dev_preds = []

    return preds, dev_preds


def calc_auc(clf, data_path, region_based=False, use_dev=False):
    train, dev, test, _, _, _ = ML_util.get_dataset(data_path)

    test_set = test if not use_dev else dev
    if len(test_set) == 0:
        return 0, [0], [0]

    targets = [row[0][-1] for row in test_set]

    preds = get_predictions(clf, train, dev, test, region_based, use_dev)
    assert preds is not None

    # calculate fpr and tpr values for different thresholds
    fpr, tpr, thresholds = roc_curve(targets, preds, drop_intermediate=False)
    auc_val = auc(fpr, tpr)
    return auc_val, fpr, tpr


def get_modality_results(data_path, seed, fet_inds, region_based=False, shuffle_labels=False):
    lists = ['aucs', 'fprs', 'tprs', 'importances', 'dev_aucs', 'dev_fprs', 'dev_tprs', 'dev_importances',
             'preds', 'dev_preds', 'raw_imps', 'dev_raw_imps']

    aucs, fprs, tprs, importances = [], [], [], []
    dev_aucs, dev_fprs, dev_tprs, dev_importances = [], [], [], []
    preds, dev_preds = [], []
    raw_imps, dev_raw_imps = [], []

    print(f"            Starting chunk size = {chunks[0]}")

    clf, n_estimators, max_depth, min_samples_split, min_samples_leaf = grid_search_rf(
        data_path + f"/0_{dataset_identifier}/", n_estimators_min,
        n_estimators_max, n_estimators_num, max_depth_min, max_depth_max, max_depth_num,
        min_samples_splits_min, min_samples_splits_max, min_samples_splits_num,
        min_samples_leafs_min, min_samples_leafs_max, min_samples_leafs_num, n, seed=seed,
        region_based=region_based, shuffle_labels=shuffle_labels)

    pred, dev_pred = get_preds(clf, data_path + f"/0_{dataset_identifier}/", region_based)

    auc, fpr, tpr = calc_auc(clf, data_path + f"/0_{dataset_identifier}/", region_based)
    dev_auc, dev_fpr, dev_tpr = calc_auc(clf, data_path + f"/0_{dataset_identifier}/", region_based, use_dev=True)

    x, _ = get_test_set(data_path + f"/0_{dataset_identifier}/", region_based)
    raw_imp = np.ones((1000, NUM_FETS)) * np.nan
    importance, raw_imp_temp = get_shap_imp(clf, x, seed)
    raw_imp[:min(1000, len(x)), fet_inds[:-1]] = raw_imp_temp

    x, _ = get_test_set(data_path + f"/0_{dataset_identifier}/", region_based, get_dev=True)
    dev_raw_imp = np.ones((1000, NUM_FETS)) * np.nan
    if len(x) > 0:
        dev_importance, dev_raw_imp_temp = get_shap_imp(clf, x, seed)
        dev_raw_imp[:min(1000, len(x)), fet_inds[:-1]] = dev_raw_imp_temp
    else:
        dev_importance = np.ones(importance.shape) * np.nan

    for var in lists:
        assignment = f"{var}.append({var[:-1]})"
        exec(assignment)

    restriction, modality = data_path.split('/')[-2:]
    restriction = '_'.join(restriction.split('_')[:-1])

    for chunk_size in chunks[1:]:
        print(f"            Starting chunk size = {chunk_size}")
        clf = run_model(n_estimators, max_depth, min_samples_split, min_samples_leaf,
                        data_path + f"/{chunk_size}_{dataset_identifier}/",
                        seed, region_based, shuffle_labels)

        auc, fpr, tpr = calc_auc(clf, data_path + f"/{chunk_size}_{dataset_identifier}/", region_based)
        dev_auc, dev_fpr, dev_tpr = calc_auc(clf, data_path + f"/{chunk_size}_{dataset_identifier}/", region_based,
                                             use_dev=True)

        print(f"\nchunk size: {chunk_size} - AUC: {auc}, dev AUC: {dev_auc}\n")

        pred, dev_pred = get_preds(clf, data_path + f"/{chunk_size}_{dataset_identifier}/", region_based)

        x, _ = get_test_set(data_path + f"/{chunk_size}_{dataset_identifier}/", region_based)
        raw_imp = np.ones((1000, NUM_FETS)) * np.nan
        importance, raw_imp_temp = get_shap_imp(clf, x, seed)
        raw_imp[:min(1000, len(x)), fet_inds[:-1]] = raw_imp_temp

        x, _ = get_test_set(data_path + f"/{chunk_size}_{dataset_identifier}/", region_based, get_dev=True)
        dev_raw_imp = np.ones((1000, NUM_FETS)) * np.nan
        if len(x) > 0:
            dev_importance, dev_raw_imp_temp = get_shap_imp(clf, x, seed)
            dev_raw_imp[:min(1000, len(x)), fet_inds[:-1]] = dev_raw_imp_temp
        else:
            dev_importance = np.ones(importance.shape) * np.nan

        for var in lists:
            assignment = f"{var}.append({var[:-1]})"
            exec(assignment)

    df = pd.DataFrame(
        {'restriction': restriction, 'modality': modality, 'chunk_size': chunks, 'seed': [str(seed)] * len(aucs),
         'auc': aucs, 'fpr': fprs, 'tpr': tprs, 'dev_auc': dev_aucs, 'dev_fpr': dev_fprs, 'dev_tpr': dev_tprs})

    features = [f"test feature {f + 1}" for f in range(NUM_FETS)]
    importances_row = np.nan * np.ones((len(aucs), NUM_FETS))
    importances_row[:, fet_inds[:-1]] = importances
    df[features] = pd.DataFrame(importances_row, index=df.index)

    features = [f"dev feature {f + 1}" for f in range(NUM_FETS)]
    importances_row = np.nan * np.ones((len(aucs), NUM_FETS))
    importances_row[:, fet_inds[:-1]] = dev_importances
    df[features] = pd.DataFrame(importances_row, index=df.index)

    return df, np.stack(tuple(preds)), np.stack(tuple(dev_preds)), np.stack(tuple(raw_imps)), np.stack(tuple(dev_raw_imps))


def get_folder_results(data_path, seed, region_based=False, shuffle_labels=False):
    df_cols = ['restriction', 'modality', 'chunk_size', 'seed', 'auc', 'fpr', 'tpr', 'dev_auc', 'dev_fpr', 'dev_tpr'] +\
              [f"test feature {f + 1}" for f in range(NUM_FETS)] + [f"dev feature {f + 1}" for f in range(NUM_FETS)]
    preds, dev_preds = None, None
    raw_imps, dev_raw_imps = None, None
    df = pd.DataFrame({col: [] for col in df_cols})
    for modality in modalities:
        print(f"        Starting modality {modality[0]}")
        modality_df, pred, dev_pred, raw_imp, dev_raw_imp = get_modality_results(
            data_path + '/' + modality[0], seed, modality[1], region_based=region_based, shuffle_labels=shuffle_labels)
        df = df.append(modality_df, ignore_index=True)
        preds = pred if preds is None else np.vstack((preds, pred))
        dev_preds = dev_pred if dev_preds is None else np.vstack((dev_preds, dev_pred))
        raw_imps = raw_imp if raw_imps is None else np.vstack((raw_imps, raw_imp))
        dev_raw_imps = dev_raw_imp if dev_raw_imps is None else np.vstack((dev_raw_imps, dev_raw_imp))

    return df, preds, dev_preds, raw_imps, dev_raw_imps


if __name__ == "__main__":
    """
    checks:
    1) weights!=balanced should be only for baselines
    2) region_based flag is updated
    3) perm_labels flag is updated (only effective when creating a *new* dataset)
    4) datas.txt is updated 
    """

    model = 'rf'
    modifier = '030822_rich_region'
    iterations = 50
    load_iter = 5
    animal_based = False
    region_based = True
    perm_labels = False  # This is done in creation of dataset
    shuffle_labels = False  # This is done in loading
    results = None if load_iter is None else pd.read_csv(f'results_{model}_{modifier}_{load_iter}.csv', index_col=0)
    preds = None if load_iter is None else np.load(f'preds_{model}_{modifier}_{load_iter}.npy')
    dev_preds = None if load_iter is None else np.load(f'preds_dev_{model}_{modifier}_{load_iter}.npy')
    raw_imps = None if load_iter is None else np.load(f'raw_imps_{model}_{modifier}_{load_iter}.npy')
    dev_raw_imps = None if load_iter is None else np.load(f'raw_imps_dev_{model}_{modifier}_{load_iter}.npy')
    save_path = f'../Datasets/data_sets_{modifier}'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if 'trans' in modifier:
        modalities = [('trans_wf', TRANS_MORPH)]
    else:
        modalities = [('spatial', SPATIAL_R), ('temporal', TEMPORAL_R), ('morphological', MORPHOLOGICAL_R)]
    for i in range(0 if load_iter is None else load_iter + 1, iterations):
        print(f"Starting iteration {i}")
        for r in restrictions:
            print(f"    Starting restrictions {r}")
            new_path = save_path + f"/{r}_{i}"
            if not os.path.isdir(new_path):
                os.mkdir(new_path)
            for name, places in modalities:
                new_new_path = new_path + f"/{name}/"
                if not os.path.isdir(new_new_path):
                    os.mkdir(new_new_path)
                keep = places
                with HiddenPrints():
                    ML_util.create_datasets(per_train=0.8, per_dev=0, per_test=0.2, datasets='datas.txt', seed=i,
                                            should_filter=True, save_path=new_new_path, verbos=False, keep=keep, mode=r,
                                            region_based=region_based, perm_labels=perm_labels)

            result, pred, dev_pred, raw_imp, dev_raw_imp = get_folder_results(new_path, i, region_based, shuffle_labels=shuffle_labels)
            if results is None:
                results = result
            else:
                results = results.append(result)

            preds = pred if preds is None else np.vstack((preds, pred))
            dev_preds = dev_pred if dev_preds is None else np.vstack((dev_preds, dev_pred))

            raw_imps = raw_imp if raw_imps is None else np.vstack((raw_imps, raw_imp))
            dev_raw_imps = dev_raw_imp if dev_raw_imps is None else np.vstack((dev_raw_imps, dev_raw_imp))

            np.save(f'preds_{model}_{modifier}_{i}', preds)
            np.save(f'preds_dev_{model}_{modifier}_{i}', dev_preds)
            np.save(f'raw_imps_{model}_{modifier}_{i}', raw_imps)
            np.save(f'raw_imps_dev_{model}_{modifier}_{i}', dev_raw_imps)
            results.to_csv(f'results_{model}_{modifier}_{i}.csv')

    results.to_csv(f'results_{model}_{modifier}.csv')
    np.save(f'preds_{model}_{modifier}', preds)
    np.save(f'preds_dev_{model}_{modifier}', dev_preds)
    np.save(f'raw_imps_{model}_{modifier}', raw_imps)
    np.save(f'raw_imps_dev_{model}_{modifier}', dev_raw_imps)
