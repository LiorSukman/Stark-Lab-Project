import ML_util
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
# from sklearn.utils.testing import ignore_warnings
import pickle
import shap

from gs_rf import grid_search as grid_search_rf
from gs_svm import grid_search as grid_search_svm
from gs_gb import grid_search as grid_search_gb
from SVM_RF import run as run_model

from constants import SPATIAL, MORPHOLOGICAL, TEMPORAL, SPAT_TEMPO
from constants import TRANS_MORPH
from utils.hideen_prints import HiddenPrints
from constants import INF

chunks = [0, 100, 200, 400, 800, 1600]
restrictions = ['complete', 'no_small_sample']
dataset_identifier = '0.800.2'
importance_mode = 'shap'  # reg, perm or shap (for rf only)
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


def get_test_set(data_path, region_based=False, get_dev=False):
    train, dev, test, _, _, _ = ML_util.get_dataset(data_path)

    train_squeezed = ML_util.squeeze_clusters(train)
    dev_squeezed = ML_util.squeeze_clusters(dev)
    if (not region_based) and len(dev_squeezed) > 0:
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

    test_set = test if not get_dev else dev
    if len(test_set) == 0:
        return [], []

    test_squeezed = ML_util.squeeze_clusters(test_set)
    x, y = ML_util.split_features(test_squeezed)
    x = np.nan_to_num(x)
    x = np.clip(x, -INF, INF)
    x = scaler.transform(x)

    return x, y


def get_shap_imp(clf, test, seed):
    df = pd.DataFrame(test)
    df_shap = df.sample(min(1000, len(test)), random_state=int(seed) + 1)
    rf_explainer = shap.TreeExplainer(clf)  # define explainer
    shap_values = rf_explainer(df_shap)  # calculate shap values
    pyr_shap_values = shap_values[..., 1]
    return np.mean(np.abs(pyr_shap_values.values), axis=0)


def calc_auc(clf, data_path, region_based=False, use_dev=False, calc_f1=False):
    train, dev, test, _, _, _ = ML_util.get_dataset(data_path)
    # test_path = data_path.replace('200', '500').replace('500', '0')
    # _, _, test, _, _, _ = ML_util.get_dataset(test_path)
    if not region_based:
        train = np.concatenate((train, dev))
    train_squeezed = ML_util.squeeze_clusters(train)
    train_features, train_labels = ML_util.split_features(train_squeezed)
    train_features = np.nan_to_num(train_features)
    train_features = np.clip(train_features, -INF, INF)

    scaler = StandardScaler()
    scaler.fit(train_features)
    _ = scaler.transform(train_features)

    preds = []
    bin_preds = []
    targets = []

    test_set = test if not use_dev else dev
    if len(test_set) == 0:
        return 0, [0], [0]

    for cluster in test_set:
        features, labels = ML_util.split_features(cluster)
        features = np.nan_to_num(features)
        features = np.clip(features, -INF, INF)
        features = scaler.transform(features)
        label = labels[0]  # as they are the same for all the cluster
        prob = clf.predict_proba(features).mean(axis=0)
        pred = prob[1]
        bin_pred = prob.argmax()
        bin_preds.append(bin_pred)
        preds.append(pred)
        targets.append(label)

    if calc_f1:
        precision, recall, thresholds = precision_recall_curve(targets, preds)
        f1 = f1_score(targets, bin_preds)

        return f1, precision, recall
    else:
        fpr, tpr, thresholds = roc_curve(targets, preds,
                                         drop_intermediate=False)  # calculate fpr and tpr values for different thresholds
        auc_val = auc(fpr, tpr)
        return auc_val, fpr, tpr


def get_modality_results(data_path, seed, model, fet_inds, importance_mode='reg', region_based=False):
    accs, pyr_accs, in_accs, aucs, fprs, tprs, importances = [], [], [], [], [], [], []
    dev_accs, dev_pyr_accs, dev_in_accs, dev_aucs, dev_fprs, dev_tprs, dev_importances = [], [], [], [], [], [], []
    C, gamma, n_estimators, max_depth, min_samples_split, min_samples_leaf, lr = [None] * 7

    path_split = data_path.split('/')
    dest = f"../saved_models/{model}_{path_split[3]}_{path_split[2]}"

    if model == 'rf':
        clf, acc, pyr_acc, in_acc, dev_acc, dev_pyr_acc, dev_in_acc, n_estimators, max_depth, min_samples_split, \
        min_samples_leaf = grid_search_rf(data_path + f"/0_{dataset_identifier}/", False, n_estimators_min,
                                          n_estimators_max, n_estimators_num,
                                          max_depth_min, max_depth_max, max_depth_num, min_samples_splits_min,
                                          min_samples_splits_max,
                                          min_samples_splits_num, min_samples_leafs_min, min_samples_leafs_max,
                                          min_samples_leafs_num, n, seed,
                                          region_based)
        auc, fpr, tpr = calc_auc(clf, data_path + f"/0_{dataset_identifier}/", region_based)
        dev_auc, dev_fpr, dev_tpr = calc_auc(clf, data_path + f"/0_{dataset_identifier}/", region_based, use_dev=True)

        if importance_mode == 'reg':
            dev_importance = importance = clf.feature_importances_
        elif importance_mode == 'perm':
            x, y = get_test_set(data_path + f"/0_{dataset_identifier}/", region_based)
            importance = permutation_importance(clf, x, y, random_state=seed + 1).importances_mean
            x, y = get_test_set(data_path + f"/0_{dataset_identifier}/", region_based, get_dev=True)
            if len(x) > 0:
                dev_importance = permutation_importance(clf, x, y, random_state=seed + 1).importances_mean
            else:
                dev_importance = np.zeros(importance.shape)
        elif importance_mode == 'shap':
            x, _ = get_test_set(data_path + f"/0_{dataset_identifier}/", region_based)
            importance = get_shap_imp(clf, x, seed)
            x, _ = get_test_set(data_path + f"/0_{dataset_identifier}/", region_based, get_dev=True)
            if len(x) > 0:
                dev_importance = get_shap_imp(clf, x, seed)
            else:
                dev_importance = np.zeros(importance.shape)
        else:
            raise NotImplementedError

    elif model == 'svm':
        clf, _, acc, pyr_acc, in_acc, dev_acc, dev_pyr_acc, dev_in_acc, C, gamma = \
            grid_search_svm(data_path + f"/0_{dataset_identifier}/", False, None, min_gamma, max_gamma, num_gamma,
                            min_c, max_c, num_c, kernel, n)
        if importance_mode == 'perm':
            x, y = get_test_set(data_path + f"/0_{dataset_identifier}/", region_based)
            importance = permutation_importance(clf, x, y, random_state=seed + 1).importances_mean
            x, y = get_test_set(data_path + f"/0_{dataset_identifier}/", region_based, get_dev=True)
            if len(x) > 0:
                dev_importance = permutation_importance(clf, x, y, random_state=seed + 1).importances_mean
            else:
                dev_importance = np.zeros(importance.shape)
        else:
            raise NotImplementedError

        auc, fpr, tpr = 0, [0], [0]
        dev_auc, dev_fpr, dev_tpr = 0, [0], [0]

    elif model == 'gb':
        clf, acc, pyr_acc, in_acc, dev_acc, dev_pyr_acc, dev_in_acc, n_estimators, max_depth, lr = grid_search_gb(
            data_path + f"/0_{dataset_identifier}/", False, n_estimators_min, n_estimators_max, n_estimators_num,
            max_depth_min, max_depth_max, max_depth_num, lr_min, lr_max, lr_num, n, region_based)
        auc, fpr, tpr = calc_auc(clf, data_path + f"/0_{dataset_identifier}/", region_based)
        dev_auc, dev_fpr, dev_tpr = calc_auc(clf, data_path + f"/0_{dataset_identifier}/", region_based, use_dev=True)

        if importance_mode == 'perm':
            x, y = get_test_set(data_path + f"/0_{dataset_identifier}/", region_based)
            importance = permutation_importance(clf, x, y, random_state=seed + 1).importances_mean
            x, y = get_test_set(data_path + f"/0_{dataset_identifier}/", region_based, get_dev=True)
            if len(x) > 0:
                dev_importance = permutation_importance(clf, x, y, random_state=seed + 1).importances_mean
            else:
                dev_importance = np.zeros(importance.shape)
        else:
            raise NotImplementedError

    else:
        raise Exception(f"model {model} is not suppurted, only svm, gb or rf are supported at the moment")

    accs.append(acc)
    pyr_accs.append(pyr_acc)
    in_accs.append(in_acc)
    dev_accs.append(dev_acc)
    dev_pyr_accs.append(dev_pyr_acc)
    dev_in_accs.append(dev_in_acc)

    aucs.append(auc)
    fprs.append(fpr)
    tprs.append(tpr)
    dev_aucs.append(dev_auc)
    dev_fprs.append(dev_fpr)
    dev_tprs.append(dev_tpr)

    importances.append(importance)
    dev_importances.append(dev_importance)

    with open(dest + "_0", 'wb') as fid:  # save the model
        pickle.dump(clf, fid)

    restriction, modality = data_path.split('/')[-2:]
    restriction = '_'.join(restriction.split('_')[:-1])

    for chunk_size in chunks[1:]:
        clf, acc, pyr_acc, in_acc, dev_acc, dev_pyr_acc, dev_in_acc = run_model(model, None, None, None, False, None,
                                                                                False, True, False, gamma, C,
                                                                                kernel,
                                                                                n_estimators, max_depth,
                                                                                min_samples_split, min_samples_leaf, lr,
                                                                                data_path +
                                                                                f"/{chunk_size}_{dataset_identifier}/",
                                                                                seed, region_based)
        if model == 'rf' or model == 'gb':
            auc, fpr, tpr = calc_auc(clf, data_path + f"/{chunk_size}_{dataset_identifier}/", region_based)
            dev_auc, dev_fpr, dev_tpr = calc_auc(clf, data_path + f"/0_{dataset_identifier}/", region_based,
                                                 use_dev=True)
            if model == 'rf':
                if importance_mode == 'reg':
                    dev_importance = importance = clf.feature_importances_
                elif importance_mode == 'perm':
                    x, y = get_test_set(data_path + f"/{chunk_size}_{dataset_identifier}/", region_based)
                    importance = permutation_importance(clf, x, y, random_state=seed + 1).importances_mean
                    x, y = get_test_set(data_path + f"/{chunk_size}_{dataset_identifier}/", region_based, get_dev=True)
                    if len(x) > 0:
                        dev_importance = permutation_importance(clf, x, y, random_state=seed + 1).importances_mean
                    else:
                        dev_importance = np.zeros(importance.shape)
                elif importance_mode == 'shap':
                    x, _ = get_test_set(data_path + f"/{chunk_size}_{dataset_identifier}/", region_based)
                    importance = get_shap_imp(clf, x, seed)
                    x, _ = get_test_set(data_path + f"/{chunk_size}_{dataset_identifier}/", region_based, get_dev=True)
                    if len(x) > 0:
                        dev_importance = get_shap_imp(clf, x, seed)
                    else:
                        dev_importance = np.zeros(importance.shape)
                else:
                    raise NotImplementedError
            else:
                if importance_mode == 'perm':
                    x, y = get_test_set(data_path + f"/{chunk_size}_{dataset_identifier}/", region_based)
                    importance = permutation_importance(clf, x, y, random_state=seed + 1).importances_mean
                    x, y = get_test_set(data_path + f"/{chunk_size}_{dataset_identifier}/", region_based, get_dev=True)
                    if len(x) > 0:
                        dev_importance = permutation_importance(clf, x, y, random_state=seed + 1).importances_mean
                    else:
                        dev_importance = np.zeros(importance.shape)
                else:
                    raise NotImplementedError

        elif model == 'svm':
            if importance_mode == 'perm':
                x, y = get_test_set(data_path + f"/{chunk_size}_{dataset_identifier}/", region_based)
                importance = permutation_importance(clf, x, y, random_state=seed + 1).importances_mean
                x, y = get_test_set(data_path + f"/{chunk_size}_{dataset_identifier}/", region_based, get_dev=True)
                if len(x) > 0:
                    dev_importance = permutation_importance(clf, x, y, random_state=seed + 1).importances_mean
                else:
                    dev_importance = np.zeros(importance.shape)
            else:
                raise NotImplementedError
            auc, fpr, tpr = 0, [0], [0]
            dev_auc, dev_fpr, dev_tpr = 0, [0], [0]

        accs.append(acc)
        pyr_accs.append(pyr_acc)
        in_accs.append(in_acc)
        dev_accs.append(dev_acc)
        dev_pyr_accs.append(dev_pyr_acc)
        dev_in_accs.append(dev_in_acc)

        aucs.append(auc)
        fprs.append(fpr)
        tprs.append(tpr)
        dev_aucs.append(dev_auc)
        dev_fprs.append(dev_fpr)
        dev_tprs.append(dev_tpr)

        importances.append(importance)
        dev_importances.append(dev_importance)

        with open(dest + f"_{chunk_size}", 'wb') as fid:  # save the model
            pickle.dump(clf, fid)

    df = pd.DataFrame(
        {'restriction': restriction, 'modality': modality, 'chunk_size': chunks, 'seed': [str(seed)] * len(accs),
         'acc': accs, 'pyr_acc': pyr_accs, 'in_acc': in_accs, 'auc': aucs, 'fpr': fprs, 'tpr': tprs,
         'dev_acc': dev_accs, 'dev_pyr_acc': dev_pyr_accs, 'dev_in_acc': dev_in_accs, 'dev_auc': dev_aucs,
         'dev_fpr': dev_fprs, 'dev_tpr': dev_tprs
         })

    features = [f"test feature {f + 1}" for f in range(NUM_FETS)]
    importances_row = np.nan * np.ones((len(accs), NUM_FETS))
    importances_row[:, fet_inds[:-1]] = importances
    df[features] = pd.DataFrame(importances_row, index=df.index)

    features = [f"dev feature {f + 1}" for f in range(NUM_FETS)]
    importances_row = np.nan * np.ones((len(accs), NUM_FETS))
    importances_row[:, fet_inds[:-1]] = dev_importances
    df[features] = pd.DataFrame(importances_row, index=df.index)

    return df


def get_folder_results(data_path, model, seed, region_based=False):
    df_cols = ['restriction', 'modality', 'chunk_size', 'seed', 'acc', 'pyr_acc', 'in_acc', 'dev_acc', 'dev_pyr_acc',
               'dev_in_acc', 'auc', 'fpr', 'tpr', 'dev_auc', 'dev_fpr', 'dev_tpr'] + \
              [f"test feature {f + 1}" for f in range(NUM_FETS)] + [f"dev feature {f + 1}" for f in range(NUM_FETS)]
    df = pd.DataFrame({col: [] for col in df_cols})
    for modality in modalities:
        print(f"        Starting modality {modality[0]}")
        with HiddenPrints():
            modality_df = get_modality_results(data_path + '/' + modality[0], seed, model, modality[1],
                                               importance_mode=importance_mode, region_based=region_based)
        df = df.append(modality_df, ignore_index=True)

    return df


if __name__ == "__main__":
    """
    checks:
    1) weights!=balanced should be only for baselines
    2) region_based flag is updated
    3) make sure nothing is permutated (keyword: shuffle) or randomly generated
    4) save_path for datasets is correct
    5) modalities are correct
    6) csv name doesn't overide anything
    7) datas.txt is updated 
    """
    model = 'rf'
    iterations = 20
    region_based = False
    perm_labels = True
    results = None
    save_path = '../data_sets_perm_labels'
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
                                            seed=i, region_based=region_based, perm_labels=perm_labels)
            if results is None:
                results = get_folder_results(new_path, model, i, region_based)
            else:
                results = results.append(get_folder_results(new_path, model, i), ignore_index=True)

    results.to_csv(f'results_{model}_chance_level.csv')
