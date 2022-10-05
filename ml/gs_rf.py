from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
import argparse
from constants import INF

import ML_util

N = 10
SAMPLE_N = 5_000

def get_inds_unsq(cum_c, inds):
    pairs = []
    for i in inds:
        if i == 0:
            pairs.append((0, cum_c[i]))
        else:
            pairs.append((cum_c[i - 1], cum_c[i]))

    new_inds_raw = [np.arange(pair[0], pair[1]) for pair in pairs]
    new_inds = np.concatenate(new_inds_raw)
    return new_inds

def cv_gen(unsqueezed, n, seed):
    ret = []
    sham_data = np.asarray([elem[0] for elem in unsqueezed])
    sham_features, labels = ML_util.split_features(sham_data)
    cum_c = np.cumsum([len(elem) for elem in unsqueezed])
    basic_split = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    i = 1
    for inds_train, inds_test in basic_split.split(sham_features, labels):
        new_inds_train_full = get_inds_unsq(cum_c, inds_train)
        new_inds_test = get_inds_unsq(cum_c, inds_test)

        np.random.seed(seed * n + i)
        l = len(new_inds_train_full)
        inds = np.random.choice(l, min(SAMPLE_N, l), replace=False)
        new_inds_train_sample = new_inds_train_full[inds]

        ret.append((new_inds_train_sample, new_inds_test))
        i += 1

    return ret

def grid_search(dataset_path, n_estimators_min, n_estimators_max, n_estimators_num,
                max_depth_min, max_depth_max, max_depth_num, min_samples_splits_min, min_samples_splits_max,
                min_samples_splits_num, min_samples_leafs_min, min_samples_leafs_max, min_samples_leafs_num, n,
                train=None, dev=None, test=None, seed=0, region_based=False, shuffle_labels=False):
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
    features = np.nan_to_num(features)
    features = np.clip(features, -INF, INF)

    if shuffle_labels:
        np.random.seed(seed)
        np.random.shuffle(labels)

    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)

    cv = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    if len(features) > SAMPLE_N:
        assert region_based
        cv = cv_gen(train, n, seed)

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
    clf = GridSearchCV(model, parameters, cv=cv, verbose=0,
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

    return classifier, n_estimators, max_depth, min_samples_split, min_samples_leaf


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
