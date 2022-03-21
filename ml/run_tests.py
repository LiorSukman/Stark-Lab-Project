import ML_util

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
import shap

from gs_rf import grid_search as grid_search_rf
from SVM_RF import run as run_model

from constants import SPATIAL, MORPHOLOGICAL, TEMPORAL
from utils.hideen_prints import HiddenPrints
from constants import INF

chunks = [0, 25, 50, 100, 200, 400, 800, 1600]
restrictions = ['complete']
dataset_identifier = '0.650.150.2'  # '0.800.2'

NUM_FETS = 34

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

def get_shap_imp(clf, test, seed):
    df = pd.DataFrame(test)
    df_shap = df.sample(min(1000, len(test)), random_state=int(seed) + 1)
    rf_explainer = shap.TreeExplainer(clf)  # define explainer
    shap_values = rf_explainer(df_shap)  # calculate shap values
    pyr_shap_values = shap_values[..., 1]
    return np.mean(np.abs(pyr_shap_values.values), axis=0)

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

def calc_auc(clf, data_path, region_based=False, use_dev=False, calc_f1=False):
    train, dev, test, _, _, _ = ML_util.get_dataset(data_path)

    test_set = test if not use_dev else dev
    if len(test_set) == 0:
        return 0, [0], [0]

    targets = [row[0][-1] for row in test_set]

    preds = get_predictions(clf, train, dev, test, region_based, use_dev)
    assert preds is not None

    if calc_f1:
        precision, recall, thresholds = precision_recall_curve(targets, preds)
        bin_preds = thr_preds(preds, 0.5)
        f1 = f1_score(targets, bin_preds)

        return f1, precision, recall
    else:
        fpr, tpr, thresholds = roc_curve(targets, preds,
                                         drop_intermediate=False)  # calculate fpr and tpr values for different thresholds
        auc_val = auc(fpr, tpr)
        return auc_val, fpr, tpr


def get_mcc(clf, data_path, region_based, use_dev=False):
    train, dev, test, _, _, _ = ML_util.get_dataset(data_path)

    test_set = test if not use_dev else dev
    if len(test_set) == 0:
        return 0

    targets = [row[0][-1] for row in test_set]

    preds = get_predictions(clf, train, dev, test, region_based, use_dev)
    assert preds is not None

    bin_preds = thr_preds(preds, 0.5)

    mcc = matthews_corrcoef(targets, bin_preds)

    return mcc


def get_modality_results(data_path, seed, fet_inds, region_based=False):
    lists = ['accs', 'pyr_accs', 'in_accs', 'aucs', 'fprs', 'tprs', 'importances', 'dev_accs', 'dev_pyr_accs',
             'dev_in_accs', 'dev_aucs', 'dev_fprs', 'dev_tprs', 'dev_importances', 'f1s', 'precisions', 'recalls',
             'dev_f1s', 'dev_precisions', 'dev_recalls', 'mccs', 'dev_mccs', 'preds', 'dev_preds']

    """for var in lists:
        assignment = f"{var}=[]"
        exec(assignment)"""

    accs, pyr_accs, in_accs, aucs, fprs, tprs, importances, dev_accs, dev_pyr_accs = [], [], [], [], [], [], [], [], []
    dev_in_accs, dev_aucs, dev_fprs, dev_tprs, dev_importances, f1s, precisions, recalls = [], [], [], [], [], [], [], []
    dev_f1s, dev_precisions, dev_recalls, mccs, dev_mccs, preds, dev_preds = [], [], [], [], [], [], []

    print(f"            Starting chunk size = {chunks[0]}")

    clf, acc, pyr_acc, in_acc, dev_acc, dev_pyr_acc, dev_in_acc, n_estimators, max_depth, min_samples_split, \
    min_samples_leaf = grid_search_rf(data_path + f"/0_{dataset_identifier}/", False, n_estimators_min,
                                      n_estimators_max, n_estimators_num,
                                      max_depth_min, max_depth_max, max_depth_num, min_samples_splits_min,
                                      min_samples_splits_max,
                                      min_samples_splits_num, min_samples_leafs_min, min_samples_leafs_max,
                                      min_samples_leafs_num, n, seed=seed, region_based=region_based)

    pred, dev_pred = get_preds(clf, data_path + f"/0_{dataset_identifier}/", region_based)

    auc, fpr, tpr = calc_auc(clf, data_path + f"/0_{dataset_identifier}/", region_based)
    dev_auc, dev_fpr, dev_tpr = calc_auc(clf, data_path + f"/0_{dataset_identifier}/", region_based, use_dev=True)

    f1, precision, recall = calc_auc(clf, data_path + f"/0_{dataset_identifier}/", region_based, calc_f1=True)
    dev_f1, dev_precision, dev_recall = calc_auc(clf, data_path + f"/0_{dataset_identifier}/", region_based,
                                                 use_dev=True, calc_f1=True)

    x, _ = get_test_set(data_path + f"/0_{dataset_identifier}/", region_based)
    importance = get_shap_imp(clf, x, seed)
    x, _ = get_test_set(data_path + f"/0_{dataset_identifier}/", region_based, get_dev=True)
    if len(x) > 0:
        dev_importance = get_shap_imp(clf, x, seed)
    else:
        dev_importance = np.zeros(importance.shape)

    mcc = get_mcc(clf, data_path + f"/0_{dataset_identifier}/", region_based)
    dev_mcc = get_mcc(clf, data_path + f"/0_{dataset_identifier}/", region_based, use_dev=True)

    for var in lists:
        assignment = f"{var}.append({var[:-1]})"
        exec(assignment)

    restriction, modality = data_path.split('/')[-2:]
    restriction = '_'.join(restriction.split('_')[:-1])

    for chunk_size in chunks[1:]:
        print(f"            Starting chunk size = {chunk_size}")
        clf, acc, pyr_acc, in_acc, dev_acc, dev_pyr_acc, dev_in_acc = run_model(model, None, None, None, False, None,
                                                                                False, True, False, None, None, None,
                                                                                n_estimators, max_depth,
                                                                                min_samples_split, min_samples_leaf,
                                                                                None, data_path +
                                                                                f"/{chunk_size}_{dataset_identifier}/",
                                                                                seed, region_based)

        auc, fpr, tpr = calc_auc(clf, data_path + f"/{chunk_size}_{dataset_identifier}/", region_based)
        dev_auc, dev_fpr, dev_tpr = calc_auc(clf, data_path + f"/{chunk_size}_{dataset_identifier}/", region_based,
                                             use_dev=True)

        print(f"\nchunk size: {chunk_size} - AUC: {auc}, dev AUC: {dev_auc}\n")

        pred, dev_pred = get_preds(clf, data_path + f"/{chunk_size}_{dataset_identifier}/", region_based)

        f1, precision, recall = calc_auc(clf, data_path + f"/{chunk_size}_{dataset_identifier}/", region_based,
                                         calc_f1=True)
        dev_f1, dev_precision, dev_recall = calc_auc(clf, data_path + f"/{chunk_size}_{dataset_identifier}/",
                                                     region_based, use_dev=True, calc_f1=True)

        x, _ = get_test_set(data_path + f"/{chunk_size}_{dataset_identifier}/", region_based)
        importance = get_shap_imp(clf, x, seed)
        x, _ = get_test_set(data_path + f"/{chunk_size}_{dataset_identifier}/", region_based, get_dev=True)
        if len(x) > 0:
            dev_importance = get_shap_imp(clf, x, seed)
        else:
            dev_importance = np.zeros(importance.shape)

        mcc = get_mcc(clf, data_path + f"/{chunk_size}_{dataset_identifier}/", region_based)
        dev_mcc = get_mcc(clf, data_path + f"/{chunk_size}_{dataset_identifier}/", region_based, use_dev=True)

        for var in lists:
            assignment = f"{var}.append({var[:-1]})"
            exec(assignment)

    df = pd.DataFrame(
        {'restriction': restriction, 'modality': modality, 'chunk_size': chunks, 'seed': [str(seed)] * len(accs),
         'acc': accs, 'pyr_acc': pyr_accs, 'in_acc': in_accs, 'auc': aucs, 'fpr': fprs, 'tpr': tprs,
         'dev_acc': dev_accs, 'dev_pyr_acc': dev_pyr_accs, 'dev_in_acc': dev_in_accs, 'dev_auc': dev_aucs,
         'dev_fpr': dev_fprs, 'dev_tpr': dev_tprs, 'f1': f1s, 'precision': precisions, 'recall': recalls,
         'dev_f1': dev_f1s, 'dev_precision': dev_precisions, 'dev_recall': dev_recalls, 'mcc': mccs, 'dev_mcc': dev_mccs
         })

    features = [f"test feature {f + 1}" for f in range(NUM_FETS)]
    importances_row = np.nan * np.ones((len(accs), NUM_FETS))
    importances_row[:, fet_inds[:-1]] = importances
    df[features] = pd.DataFrame(importances_row, index=df.index)

    features = [f"dev feature {f + 1}" for f in range(NUM_FETS)]
    importances_row = np.nan * np.ones((len(accs), NUM_FETS))
    importances_row[:, fet_inds[:-1]] = dev_importances
    df[features] = pd.DataFrame(importances_row, index=df.index)

    return df, np.stack(tuple(preds)), np.stack(tuple(dev_preds))


def get_folder_results(data_path, seed, region_based=False):
    df_cols = ['restriction', 'modality', 'chunk_size', 'seed', 'acc', 'pyr_acc', 'in_acc', 'dev_acc', 'dev_pyr_acc',
               'dev_in_acc', 'auc', 'fpr', 'tpr', 'dev_auc', 'dev_fpr', 'dev_tpr', 'f1', 'precision', 'recall',
               'dev_f1', 'dev_precision', 'dev_recall', 'mcc', 'dev_mcc'] + \
              [f"test feature {f + 1}" for f in range(NUM_FETS)] + [f"dev feature {f + 1}" for f in range(NUM_FETS)]
    preds, dev_preds = None, None
    df = pd.DataFrame({col: [] for col in df_cols})
    for modality in modalities:
        print(f"        Starting modality {modality[0]}")
        modality_df, pred, dev_pred = get_modality_results(data_path + '/' + modality[0], seed, modality[1], region_based=region_based)
        df = df.append(modality_df, ignore_index=True)
        preds = pred if preds is None else np.vstack((preds, pred))
        dev_preds = dev_pred if dev_preds is None else np.vstack((dev_preds, dev_pred))

    return df, preds, dev_preds


if __name__ == "__main__":
    """
    checks:
    1) weights!=balanced should be only for baselines
    2) region_based flag is updated
    3) perm_labels flag is updated (only effective when creating a *new* dataset)
    4) save_path for datasets is correct
    5) modalities are correct
    6) csv name doesn't overide anything
    7) datas.txt is updated 
    """
    model = 'rf'
    modifier = 'with_dev'
    iterations = 25
    animal_based = False
    region_based = True
    perm_labels = False
    results, preds, dev_preds = None, None, None
    save_path = '../data_sets_dev_170322'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
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
                warnings.warn('region_based is deactivated in create_dataset')
                with HiddenPrints():
                    ML_util.create_datasets(per_train=0.65, per_dev=0.15, per_test=0.2, datasets='datas.txt', seed=i,
                                            should_filter=True, save_path=new_new_path, verbos=False, keep=keep, mode=r,
                                            region_based=False, perm_labels=perm_labels,
                                            group_split=animal_based)

            result, pred, dev_pred = get_folder_results(new_path, i, region_based)
            if results is None:
                results = result
            else:
                results = results.append(result)

            preds = pred if preds is None else np.vstack((preds, pred))
            dev_preds = dev_pred if dev_preds is None else np.vstack((dev_preds, dev_pred))

            if i % 5 == 0:
                np.save(f'preds_{model}_{modifier}_{i}', preds)
                np.save(f'preds_dev_{model}_{modifier}_{i}', dev_preds)
                results.to_csv(f'results_{model}_{modifier}_{i}.csv')

    results.to_csv(f'results_{model}_{modifier}.csv')
    np.save(f'preds_{model}_{modifier}', preds)
    np.save(f'preds_dev_{model}_{modifier}', dev_preds)
