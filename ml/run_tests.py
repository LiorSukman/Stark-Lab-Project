import ML_util
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

from gs_rf import grid_search
from SVM_RF import run as run_model

from constants import SPATIAL, MORPHOLOGICAL, TEMPORAL, SPAT_TEMPO

chunks = [0, 500, 200]
modalities = ['spatial', 'morphological', 'temporal']
restrictions = ['complete', 'no_small_sample']

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

def calc_auc(clf, data_path):
    train, dev, test, _, _, _ = ML_util.get_dataset(data_path)
    train = np.concatenate((train, dev))
    train_squeezed = ML_util.squeeze_clusters(train)
    train_features, train_labels = ML_util.split_features(train_squeezed)

    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)

    preds = []
    targets = []

    for cluster in test:
        features, labels = ML_util.split_features(cluster)
        features = np.nan_to_num(features)
        features = scaler.transform(features)
        label = labels[0]  # as they are the same for all the cluster
        pred = clf.predict_proba(features).mean(axis=0)[1]
        preds.append(pred)
        targets.append(label)

    fpr, tpr, thresholds = roc_curve(targets, preds)  # calculate fpr and tpr values for different thresholds
    auc_val = auc(fpr, tpr)

    return auc_val


def get_modality_results(data_path):
    accs, pyr_accs, in_accs, aucs = [], [], [], []

    clf, acc, pyr_acc, in_acc, n_estimators, max_depth, min_samples_split, min_samples_leaf = grid_search(
        data_path + "/0_0.60.20.2/", False, n_estimators_min, n_estimators_max, n_estimators_num,
        max_depth_min, max_depth_max, max_depth_num, min_samples_splits_min, min_samples_splits_max,
        min_samples_splits_num, min_samples_leafs_min, min_samples_leafs_max, min_samples_leafs_num, n)
    auc = calc_auc(clf, data_path + "/0_0.60.20.2/")

    accs.append(acc)
    pyr_accs.append(pyr_acc)
    in_accs.append(in_acc)
    aucs.append(auc)

    restriction, modality = data_path.split('/')[-2:]

    if modality == 'temporal':
        accs = accs * len(chunks)
        pyr_accs = pyr_accs * len(chunks)
        in_accs = in_accs * len(chunks)
        aucs = aucs * len(chunks)
    else:
        for chunk_size in chunks[1:]:
            clf, acc, pyr_acc, in_acc = run_model('rf', None, None, None, False, None, False, True, False, None, None,
                                                  None,
                                                  n_estimators, max_depth, min_samples_split, min_samples_leaf,
                                                  data_path + f"/{chunk_size}_0.60.20.2/")
            auc = calc_auc(clf, data_path + f"/{chunk_size}_0.60.20.2/")

            accs.append(acc)
            pyr_accs.append(pyr_acc)
            in_accs.append(in_acc)
            aucs.append(auc)

    df = pd.DataFrame({'restriction': restriction, 'modality': modality, 'chunk_size': chunks,
                       'acc': accs, 'pyr_acc': pyr_accs, 'in_acc': in_accs, 'auc': aucs})

    return df


def get_folder_results(data_path):
    df = pd.DataFrame(
        {'restriction': [], 'modality': [], 'chunk_size': [], 'acc': [], 'pyr_acc': [], 'in_acc': [], 'auc': []})
    for modality in modalities:
        modaility_df = get_modality_results(data_path + '/' + modality)
        df = df.append(modaility_df, ignore_index=True)

    return df


def get_results(data_path):
    df = pd.DataFrame(
        {'restriction': [], 'modality': [], 'chunk_size': [], 'acc': [], 'pyr_acc': [], 'in_acc': [], 'auc': []})
    for restriction in restrictions:
        folder_df = get_folder_results(data_path + '/' + restriction)
        df = df.append(folder_df, ignore_index=True)

    return df


def do_test(data_path):
    return get_results(data_path)


if __name__ == "__main__":
    iterations = 10
    results = []
    save_path = '../data_sets'
    restrictions = ['complete', 'no_small_sample']
    modalities = [('spatial', SPATIAL), ('morphological', MORPHOLOGICAL), ('temporal', TEMPORAL),
                  ('spat_tempo', SPAT_TEMPO)]
    for i in range(iterations):
        for r in restrictions:
            new_path = save_path + f"/{r}_{i}"
            if not os.path.isdir(new_path):
                os.mkdir(new_path)
            for name, places in modalities:
                new_new_path = new_path + f"/{name}/"
                if not os.path.isdir(new_new_path):
                    os.mkdir(new_new_path)
                keep = places
                # TODO make sure that the per_dev=0 is ok
                # TODO in group split it might not be good to just change the seed like this
                ML_util.create_datasets(per_train=0.8, per_dev=0, per_test=0.2, datasets='datas.txt',
                                        should_filter=True, save_path=new_new_path, verbos=False, keep=keep, mode=r,
                                        seed=i)
        results.append(do_test(save_path + f"/{r}_{i}"))
