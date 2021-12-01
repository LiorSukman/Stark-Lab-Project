import ML_util
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
# from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import pickle

from gs_rf import grid_search as grid_search_rf
from gs_svm import grid_search as grid_search_svm
from gs_gb import grid_search as grid_search_gb
from SVM_RF import run as run_model

from constants import SPATIAL, MORPHOLOGICAL, TEMPORAL, SPAT_TEMPO
from constants import TRANS_MORPH
from utils.hideen_prints import HiddenPrints
from constants import INF

chunks = [0, 500, 200]
restrictions = ['complete', 'no_small_sample']
dataset_identifier = '0.800.2'
importance_mode = 'reg'
# try_load = '../saved_models' # TODO implement
NUM_FETS = 29

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

lr_min = -3
lr_max = 1
lr_num = 5

min_gamma = -8
max_gamma = 0
num_gamma = 9
min_c = 0
max_c = 6
num_c = 7
kernel = 'rbf'

n = 5

def get_test_set(data_path):
    train, dev, test, _, _, _ = ML_util.get_dataset(data_path)

    train_squeezed = ML_util.squeeze_clusters(train)
    dev_squeezed = ML_util.squeeze_clusters(dev)
    if len(dev_squeezed) > 0:
        train_data = np.concatenate((train_squeezed, dev_squeezed))
    else:
        train_data = train_squeezed
    features, labels = ML_util.split_features(train_data)
    # np.random.shuffle(labels)
    features = np.nan_to_num(features)
    features = np.clip(features, -INF, INF)
    # features = np.random.normal(size=features.shape)

    scaler = StandardScaler()
    scaler.fit(features)

    test_squeezed = ML_util.squeeze_clusters(test)
    x, y = ML_util.split_features(test_squeezed)
    x = np.nan_to_num(x)
    x = np.clip(x, -INF, INF)
    x = scaler.transform(x)

    return x, y


def calc_auc(clf, data_path):
    train, dev, test, _, _, _ = ML_util.get_dataset(data_path)
    # test_path = data_path.replace('200', '500').replace('500', '0')
    # _, _, test, _, _, _ = ML_util.get_dataset(test_path)
    train = np.concatenate((train, dev))
    train_squeezed = ML_util.squeeze_clusters(train)
    train_features, train_labels = ML_util.split_features(train_squeezed)
    train_features = np.nan_to_num(train_features)
    train_features = np.clip(train_features, -INF, INF)

    scaler = StandardScaler()
    scaler.fit(train_features)
    _ = scaler.transform(train_features)

    preds = []
    targets = []

    for cluster in test:
        features, labels = ML_util.split_features(cluster)
        features = np.nan_to_num(features)
        features = np.clip(features, -INF, INF)
        features = scaler.transform(features)
        label = labels[0]  # as they are the same for all the cluster
        pred = clf.predict_proba(features).mean(axis=0)[1]
        preds.append(pred)
        targets.append(label)

    fpr, tpr, thresholds = roc_curve(targets, preds)  # calculate fpr and tpr values for different thresholds
    auc_val = auc(fpr, tpr)

    return auc_val, fpr, tpr


def get_modality_results(data_path, seed, model, fet_inds, importance_mode='reg'):
    accs, pyr_accs, in_accs, aucs, importances, fprs, tprs = [], [], [], [], [], [], []
    C, gamma, n_estimators, max_depth, min_samples_split, min_samples_leaf, lr = [None] * 7

    path_split = data_path.split('/')
    dest = f"../saved_models/{model}_{path_split[3]}_{path_split[2]}"

    if model == 'rf':
        clf, acc, pyr_acc, in_acc, n_estimators, max_depth, min_samples_split, min_samples_leaf = grid_search_rf(
            data_path + f"/0_{dataset_identifier}/", False, n_estimators_min, n_estimators_max, n_estimators_num,
            max_depth_min, max_depth_max, max_depth_num, min_samples_splits_min, min_samples_splits_max,
            min_samples_splits_num, min_samples_leafs_min, min_samples_leafs_max, min_samples_leafs_num, n, seed)
        auc, fpr, tpr = calc_auc(clf, data_path + f"/0_{dataset_identifier}/")
        if importance_mode == 'reg':
            importance = clf.feature_importances_
        elif importance_mode == 'perm':
            x, y = get_test_set(data_path + f"/0_{dataset_identifier}/")
            importance = permutation_importance(clf, x, y, random_state=seed + 1).importances_mean
        else:
            raise NotImplementedError

    elif model == 'svm':
        clf, _, acc, pyr_acc, in_acc, C, gamma = grid_search_svm(data_path + f"/0_{dataset_identifier}/", False, None,
                                                                 min_gamma, max_gamma, num_gamma, min_c, max_c, num_c,
                                                                 kernel, n)
        if importance_mode == 'perm':
            x, y = get_test_set(data_path + f"/0_{dataset_identifier}/")
            importance = permutation_importance(clf, x, y, random_state=seed + 1).importances_mean
        else:
            raise NotImplementedError

        auc = 0
        fpr = tpr = [0]

    elif model == 'gb':
        clf, acc, pyr_acc, in_acc, n_estimators, max_depth, lr = grid_search_gb(
            data_path + f"/0_{dataset_identifier}/", False, n_estimators_min, n_estimators_max, n_estimators_num,
            max_depth_min, max_depth_max, max_depth_num, lr_min, lr_max, lr_num, n)
        auc, fpr, tpr = calc_auc(clf, data_path + f"/0_{dataset_identifier}/")

        if importance_mode == 'perm':
            x, y = get_test_set(data_path + f"/0_{dataset_identifier}/")
            importance = permutation_importance(clf, x, y, random_state=seed + 1).importances_mean
        else:
            raise NotImplementedError

    else:
        raise Exception(f"model {model} is not suppurted, only svm, gb or rf are supported at the moment")

    accs.append(acc)
    pyr_accs.append(pyr_acc)
    in_accs.append(in_acc)
    aucs.append(auc)
    importances.append(importance)
    fprs.append(fpr)
    tprs.append(tpr)

    with open(dest + "_0", 'wb') as fid:  # save the model
        pickle.dump(clf, fid)

    restriction, modality = data_path.split('/')[-2:]
    restriction = '_'.join(restriction.split('_')[:-1])

    for chunk_size in chunks[1:]:
        clf, acc, pyr_acc, in_acc = run_model(model, None, None, None, False, None, False, True, False, gamma, C,
                                              kernel,
                                              n_estimators, max_depth, min_samples_split, min_samples_leaf, lr,
                                              data_path + f"/{chunk_size}_{dataset_identifier}/", seed)
        if model == 'rf' or model == 'gb':
            auc, fpr, tpr = calc_auc(clf, data_path + f"/{chunk_size}_{dataset_identifier}/")
            if model == 'rf':
                if importance_mode == 'reg':
                    importance = clf.feature_importances_
                elif importance_mode == 'perm':
                    x, y = get_test_set(data_path + f"/{chunk_size}_{dataset_identifier}/")
                    importance = permutation_importance(clf, x, y, random_state=seed + 1).importances_mean
                else:
                    raise NotImplementedError
            else:
                if importance_mode == 'perm':
                    x, y = get_test_set(data_path + f"/{chunk_size}_{dataset_identifier}/")
                    importance = permutation_importance(clf, x, y, random_state=seed + 1).importances_mean
                else:
                    raise NotImplementedError

        elif model == 'svm':
            if importance_mode == 'perm':
                x, y = get_test_set(data_path + f"/{chunk_size}_{dataset_identifier}/")
                importance = permutation_importance(clf, x, y, random_state=seed + 1).importances_mean
            else:
                raise NotImplementedError
            auc = 0

        accs.append(acc)
        pyr_accs.append(pyr_acc)
        in_accs.append(in_acc)
        aucs.append(auc)
        importances.append(importance)
        fprs.append(fpr)
        tprs.append(tpr)

        with open(dest + f"_{chunk_size}", 'wb') as fid:  # save the model
            pickle.dump(clf, fid)

    df = pd.DataFrame(
        {'restriction': restriction, 'modality': modality, 'chunk_size': chunks, 'seed': [str(seed)] * len(accs),
         'acc': accs, 'pyr_acc': pyr_accs, 'in_acc': in_accs, 'auc': aucs, 'fpr': fprs, 'tpr': tprs})

    features = [f"feature {f+1}" for f in range(NUM_FETS)]
    importances_row = np.nan * np.ones((len(accs), NUM_FETS))
    importances_row[:, fet_inds[:-1]] = importances
    df[features] = pd.DataFrame(importances_row, index=df.index)

    return df


def get_folder_results(data_path, model, seed):
    df_cols = ['restriction', 'modality', 'chunk_size', 'seed', 'acc', 'pyr_acc', 'in_acc', 'auc', 'fpr', 'tpr',] + \
              [f"feature {f+1}" for f in range(NUM_FETS)]
    df = pd.DataFrame({col: [] for col in df_cols})
    for modality in modalities:
        print(f"        Starting modality {modality[0]}")
        with HiddenPrints():
            modality_df = get_modality_results(data_path + '/' + modality[0], seed, model, modality[1],
                                               importance_mode=importance_mode)
        df = df.append(modality_df, ignore_index=True)

    return df


if __name__ == "__main__":
    """
    checks:
    1) weights!=balanced should be only for baselines
    2) dev should be used only when not using region data
    3) make sure nothing is permutated (keyword: shuffle) or randomly generated
    4) save_path for datasets is correct
    5) modalities are correct
    6) csv name doesn't overide anything
    7) datas.txt is updated 
    8) region_based flag is updated
    """
    model = 'rf'
    iterations = 20
    results = None
    save_path = '../data_sets_new'
    restrictions = ['complete']
    modalities = [('spatial', SPATIAL), ('temporal', TEMPORAL), ('morphological', MORPHOLOGICAL)]
    for i in range(iterations):
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
                # TODO in group split it might not be good to just change the seed like this
                with HiddenPrints():
                    ML_util.create_datasets(per_train=0.8, per_dev=0, per_test=0.2, datasets='datas.txt',
                                            should_filter=True, save_path=new_new_path, verbos=False, keep=keep, mode=r,
                                            seed=i, region_based=False)
            if results is None:
                results = get_folder_results(new_path, model, i)
            else:
                results = results.append(get_folder_results(new_path, model, i), ignore_index=True)

    results.to_csv(f'results_{model}.csv')
