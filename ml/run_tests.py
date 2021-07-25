import ML_util
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

from gs_rf import grid_search as grid_search_rf
from gs_svm import grid_search as grid_search_svm
from SVM_RF import run as run_model

from constants import SPATIAL, MORPHOLOGICAL, TEMPORAL, SPAT_TEMPO
from utils.hideen_prints import HiddenPrints

chunks = [0, 500, 200]
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

"""min_gamma = -9
max_gamma = -1
num_gamma = 9
min_c = 0
max_c = 6
num_c = 7"""
min_gamma = -1
max_gamma = -1
num_gamma = 1
min_c = 0
max_c = 0
num_c = 1
kernel = 'rbf'

n = 5


def get_num_cells(restriction):
    data_path = f"./data_sets/{restriction}/spatial/0_0.60.20.2/"
    train, dev, test, _, _, _ = ML_util.get_dataset(data_path)
    tot_n = len(train) + len(dev) + len(test)
    test_n = len(test)
    pyr_n = len([cell for cell in test if cell[0][-1] == 1])
    in_n = len([cell for cell in test if cell[0][-1] == 0])
    return tot_n, test_n, pyr_n, in_n


def autolabel(rects, ax):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    This function was taken from https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        height = rect.get_height()
        if height == 0:
            continue
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_results(df, stds, restriction, acc=True):
    tot_n, test_n, pyr_n, in_n = get_num_cells(restriction)

    labels = modalities

    if acc:
        zero = df.xs(0, level="chunk_size").acc
        five_hundred = df.xs(500, level="chunk_size").acc
        two_hundred = df.xs(200, level="chunk_size").acc
        zero_std = stds.xs(0, level="chunk_size").acc
        five_hundred_std = stds.xs(500, level="chunk_size").acc
        two_hundred_std = stds.xs(200, level="chunk_size").acc
    else:
        zero = df.xs(0, level="chunk_size").auc
        five_hundred = df.xs(500, level="chunk_size").auc
        two_hundred = df.xs(200, level="chunk_size").auc
        zero_std = stds.xs(0, level="chunk_size").auc
        five_hundred_std = stds.xs(500, level="chunk_size").auc
        two_hundred_std = stds.xs(200, level="chunk_size").auc

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 12))
    rects1 = ax.bar(x - width, zero, width, label='chunk_size = 0', yerr=zero_std)
    rects2 = ax.bar(x, five_hundred, width, label='chunk_size = 500', yerr=five_hundred_std)
    rects3 = ax.bar(x + width, two_hundred, width, label='chunk_size = 200', yerr=two_hundred_std)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores (percentage)')
    ax.set_title(f"Scores by modality and chunk size; Total={tot_n}, Test={test_n}, PYR={pyr_n}, IN={in_n}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)

    fig.tight_layout()

    plt.show()


def calc_auc(clf, data_path):
    train, dev, test, _, _, _ = ML_util.get_dataset(data_path)
    train = np.concatenate((train, dev))
    train_squeezed = ML_util.squeeze_clusters(train)
    train_features, train_labels = ML_util.split_features(train_squeezed)

    scaler = StandardScaler()
    scaler.fit(train_features)
    _ = scaler.transform(train_features)

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


def get_modality_results(data_path, seed, model):
    accs, pyr_accs, in_accs, aucs = [], [], [], []
    C, gamma, n_estimators, max_depth, min_samples_split, min_samples_leaf = None, None, None, None, None, None

    if model == 'rf':
        clf, acc, pyr_acc, in_acc, n_estimators, max_depth, min_samples_split, min_samples_leaf = grid_search_rf(
            data_path + "/0_0.60.20.2/", False, n_estimators_min, n_estimators_max, n_estimators_num,
            max_depth_min, max_depth_max, max_depth_num, min_samples_splits_min, min_samples_splits_max,
            min_samples_splits_num, min_samples_leafs_min, min_samples_leafs_max, min_samples_leafs_num, n)
        auc = calc_auc(clf, data_path + "/0_0.60.20.2/")

        accs.append(acc)
        pyr_accs.append(pyr_acc)
        in_accs.append(in_acc)
        aucs.append(auc)
    elif model == 'svm':
        _, acc, pyr_acc, in_acc, C, gamma = grid_search_svm(data_path + "/0_0.60.20.2/", False, None, min_gamma,
                                                            max_gamma, num_gamma, min_c, max_c, num_c, kernel, n)
        accs.append(acc)
        pyr_accs.append(pyr_acc)
        in_accs.append(in_acc)
        aucs.append(0)
    else:
        raise Exception(f"model {model} is not suppurted, only svm or rf are supported at the moment")

    restriction, modality = data_path.split('/')[-2:]
    restriction = '_'.join(restriction.split('_')[:-1])

    for chunk_size in chunks[1:]:
        clf, acc, pyr_acc, in_acc = run_model(model, None, None, None, False, None, False, True, False, gamma, C,
                                              kernel,
                                              n_estimators, max_depth, min_samples_split, min_samples_leaf,
                                              data_path + f"/{chunk_size}_0.60.20.2/")
        if model == 'rf':
            auc = calc_auc(clf, data_path + f"/{chunk_size}_0.60.20.2/")
        elif model == 'svm':
            auc = 0

        accs.append(acc)
        pyr_accs.append(pyr_acc)
        in_accs.append(in_acc)
        aucs.append(auc)

    df = pd.DataFrame(
        {'restriction': restriction, 'modality': modality, 'chunk_size': chunks, 'seed': [seed] * len(accs),
         'acc': accs, 'pyr_acc': pyr_accs, 'in_acc': in_accs, 'auc': aucs})

    return df


def get_folder_results(data_path, model):
    df = pd.DataFrame(
        {'restriction': [], 'modality': [], 'chunk_size': [], 'seed': [], 'acc': [], 'pyr_acc': [], 'in_acc': [],
         'auc': []})
    seed = data_path.split('_')[-1]
    for modality in modalities:
        modality_df = get_modality_results(data_path + '/' + modality[0], seed, model)
        df = df.append(modality_df, ignore_index=True)

    return df


def get_results(data_path, model):
    df = pd.DataFrame(
        {'restriction': [], 'modality': [], 'chunk_size': [], 'seed': [], 'acc': [], 'pyr_acc': [], 'in_acc': [],
         'auc': []})
    folder_df = get_folder_results(data_path, model)
    df = df.append(folder_df, ignore_index=True)

    return df


def do_test(data_path, model):
    return get_results(data_path, model)


if __name__ == "__main__":
    model = 'svm'
    iterations = 2
    results = pd.DataFrame(
        {'restriction': [], 'modality': [], 'chunk_size': [], 'seed': [], 'acc': [], 'pyr_acc': [], 'in_acc': [],
         'auc': []})
    save_path = '../data_sets'
    restrictions = ['complete', 'no_small_sample']
    modalities = [('spatial', SPATIAL), ('morphological', MORPHOLOGICAL), ('temporal', TEMPORAL),
                  ('spat_tempo', SPAT_TEMPO)]
    for i in range(iterations):
        print(f"Starting iteration {i}")
        for r in restrictions:
            print(f"    Starting restrictions {r}")
            new_path = save_path + f"/{r}_{i}"
            if not os.path.isdir(new_path):
                os.mkdir(new_path)
            for name, places in modalities:
                print(f"        Starting modality {name}")
                new_new_path = new_path + f"/{name}/"
                if not os.path.isdir(new_new_path):
                    os.mkdir(new_new_path)
                keep = places
                # TODO make sure that the per_dev=0 is ok
                # TODO in group split it might not be good to just change the seed like this
                with HiddenPrints():
                    ML_util.create_datasets(per_train=0.6, per_dev=0.2, per_test=0.2, datasets='datas.txt',
                                            should_filter=True, save_path=new_new_path, verbos=False, keep=keep, mode=r,
                                            seed=i)
            with HiddenPrints():
                results = results.append(do_test(new_path, model), ignore_index=True)

    results.to_csv(f'results_{model}.csv')
