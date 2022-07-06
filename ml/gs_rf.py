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

def grid_search(dataset_path, n_estimators_min, n_estimators_max, n_estimators_num,
                max_depth_min, max_depth_max, max_depth_num, n, train=None, dev=None, test=None, seed=0,
                region_based=False, shuffle_labels=False):
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
        np.random.shuffle(labels)

    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)

    n_estimatorss = np.logspace(n_estimators_min, n_estimators_max, n_estimators_num).astype('int')
    max_depths = np.logspace(max_depth_min, max_depth_max, max_depth_num).astype('int')

    print()
    parameters = {'n_estimators': n_estimatorss, 'max_depth': max_depths}
    model = RandomForestClassifier(random_state=seed, class_weight='balanced',
                                   max_samples=1000 if len(train_squeezed) > 1000 else None)
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
    # need to create another one as the other trains on both train and dev
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                        random_state=seed, class_weight='balanced')
    classifier.fit(features, labels)

    return classifier, n_estimators, max_depth


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
    n = args.n

    grid_search(dataset_path, verbos, n_estimators_min, n_estimators_max, n_estimators_num,
                max_depth_min, max_depth_max, max_depth_num, n)
