from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
import argparse
import pickle
from constants import INF

import ML_util

N = 10


def evaluate_predictions(model, clusters, scaler, verbos=False):
    total = len(clusters)
    total_pyr = total_in = correct_pyr = correct_in = correct_chunks = correct_clusters = total_chunks = 0
    for cluster in clusters:
        features, labels = ML_util.split_features(cluster)
        features = np.nan_to_num(features)
        features = np.clip(features, -INF, INF)
        # features = np.random.normal(size=features.shape)
        features = scaler.transform(features)
        label = labels[0]  # as they are the same for all the cluster
        total_pyr += 1 if label == 1 else 0
        total_in += 1 if label == 0 else 0
        preds = model.predict_proba(features)
        preds_argmax = preds.argmax(axis=1)  # support probabilistic prediction
        preds_mean = preds.mean(axis=0)
        prediction = preds_mean.argmax()
        total_chunks += preds.shape[0]
        correct_chunks += preds[preds_argmax == label].shape[0]
        correct_clusters += 1 if prediction == label else 0
        correct_pyr += 1 if prediction == label and label == 1 else 0
        correct_in += 1 if prediction == label and label == 0 else 0

    pyr_percent = float('nan') if total_pyr == 0 else 100 * correct_pyr / total_pyr
    in_percent = float('nan') if total_in == 0 else 100 * correct_in / total_in

    if verbos:
        print('Number of correct classified clusters is %d, which is %.4f%s' % (
            correct_clusters, 100 * correct_clusters / total, '%'))
        print('Number of correct classified chunks is %d, which is %.4f%s' % (
            correct_chunks, 100 * correct_chunks / total_chunks, '%'))
        print('Test set consists of %d pyramidal cells and %d interneurons' % (total_pyr, total_in))
        print('%.4f%s of pyrmidal cells classified correctly' % (pyr_percent, '%'))
        print('%.4f%s of interneurons classified correctly' % (in_percent, '%'))
    return correct_clusters, 100 * correct_clusters / total, pyr_percent, in_percent


def grid_search(dataset_path, verbos, n_estimators_min, n_estimators_max, n_estimators_num,
                max_depth_min, max_depth_max, max_depth_num, min_samples_splits_min, min_samples_splits_max,
                min_samples_splits_num, min_samples_leafs_min, min_samples_leafs_max, min_samples_leafs_num, n,
                train=None, dev=None, test=None, seed=0, region_based=False):
    """
    grid search runner function
    see help for parameter explanations
    """
    if train is None or dev is None or test is None:
        train, dev, test, _, _, _ = ML_util.get_dataset(dataset_path)

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
    features = scaler.transform(features)

    n_estimatorss = np.logspace(n_estimators_min, n_estimators_max, n_estimators_num).astype('int')
    max_depths = np.logspace(max_depth_min, max_depth_max, max_depth_num).astype('int')
    min_samples_splits = np.logspace(min_samples_splits_min, min_samples_splits_max, min_samples_splits_num,
                                     base=2).astype('int')
    min_samples_leafs = np.logspace(min_samples_leafs_min, min_samples_leafs_max, min_samples_leafs_num, base=2).astype(
        'int')

    print()
    parameters = {'n_estimators': n_estimatorss, 'max_depth': max_depths, 'min_samples_split': min_samples_splits,
                  'min_samples_leaf': min_samples_leafs}
    model = RandomForestClassifier(random_state=seed, class_weight='balanced')
    clf = GridSearchCV(model, parameters, cv=StratifiedKFold(n_splits=n, shuffle=True, random_state=seed), verbose=0,
                       scoring='roc_auc')
    print('Starting grid search...')
    start = time.time()
    clf.fit(features, labels)
    end = time.time()
    print('Grid search completed in %.2f seconds, best parameters are:' % (end - start))
    print(clf.best_params_)

    n_estimators = clf.best_params_['n_estimators']
    max_depth = clf.best_params_['max_depth']
    min_samples_split = clf.best_params_['min_samples_split']
    min_samples_leaf = clf.best_params_['min_samples_leaf']
    # need to create another one as the other trains on both train and dev
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                        random_state=seed, class_weight='balanced')
    classifier.fit(features, labels)

    print()
    print('Starting evaluation on test set...')

    clust_count, acc, pyr_acc, in_acc = evaluate_predictions(classifier, test, scaler, verbos)
    dev_clust_count, dev_acc, dev_pyr_acc, dev_in_acc = evaluate_predictions(classifier, dev, scaler, verbos)


    """with open('rf_trained_model', 'wb') as fid:  # save the model
        pickle.dump(clf, fid)"""

    return classifier, acc, pyr_acc, in_acc, dev_acc, dev_pyr_acc, dev_in_acc, n_estimators, max_depth, \
           min_samples_split, min_samples_leaf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random forest grid search\n")

    parser.add_argument('--dataset_path', type=str, help='path to the dataset, assume it was created',
                        default='../data_sets/complete_0/spatial/0_0.60.20.2/')
    parser.add_argument('--verbos', type=bool, help='verbosity level (bool)', default=True)
    parser.add_argument('--n_estimators_min', type=int, help='minimal power of n_estimators (base 10)', default=0)
    parser.add_argument('--n_estimators_max', type=int, help='maximal power of n_estimators (base 10)', default=2)
    parser.add_argument('--n_estimators_num', type=int, help='number of n_estimators values', default=3)
    parser.add_argument('--max_depth_min', type=int, help='minimal power of max_depth (base 10)', default=1)
    parser.add_argument('--max_depth_max', type=int, help='maximal power of max_depth (base 10)', default=2)
    parser.add_argument('--max_depth_num', type=int, help='number of max_depth values', default=2)
    parser.add_argument('--min_samples_splits_min', type=int, help='minimal power of min_samples_splits (base 2)',
                        default=1)
    parser.add_argument('--min_samples_splits_max', type=int, help='maximal power of min_samples_splits (base 2)',
                        default=5)
    parser.add_argument('--min_samples_splits_num', type=int, help='number of min_samples_splits values', default=5)
    parser.add_argument('--min_samples_leafs_min', type=int, help='minimal power of min_samples_leafs (base 2)',
                        default=0)
    parser.add_argument('--min_samples_leafs_max', type=int, help='maximal power of min_samples_leafs (base 2)',
                        default=5)
    parser.add_argument('--min_samples_leafs_num', type=int, help='number of min_samples_leafs values', default=6)
    parser.add_argument('--n', type=int, help='number of repetitions', default=N)

    args = parser.parse_args()

    dataset_path = args.dataset_path
    verbos = args.verbos
    n_estimators_min = args.n_estimators_min
    n_estimators_max = args.n_estimators_max
    n_estimators_num = args.n_estimators_num
    max_depth_min = args.max_depth_min
    max_depth_max = args.max_depth_max
    max_depth_num = args.max_depth_num
    min_samples_splits_min = args.min_samples_splits_min
    min_samples_splits_max = args.min_samples_splits_max
    min_samples_splits_num = args.min_samples_splits_num
    min_samples_leafs_min = args.min_samples_leafs_min
    min_samples_leafs_max = args.min_samples_leafs_max
    min_samples_leafs_num = args.min_samples_leafs_num
    n = args.n

    grid_search(dataset_path, verbos, n_estimators_min, n_estimators_max, n_estimators_num,
                max_depth_min, max_depth_max, max_depth_num, min_samples_splits_min, min_samples_splits_max,
                min_samples_splits_num, min_samples_leafs_min, min_samples_leafs_max, min_samples_leafs_num, n)
