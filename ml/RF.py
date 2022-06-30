from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import numpy as np
import time
import argparse

import ML_util
from constants import INF

N = 10

def run(n_estimators, max_depth, min_samples_split, min_samples_leaf,
        dataset_path, seed, region_based=False, shuffle_labels=False):
    """
    runner function for the RF model.
    explanations about the parameters are in the help
   """

    train, dev, test, _, dev_names, test_names = ML_util.get_dataset(dataset_path)

    if (not region_based) and len(dev) > 0:  # for region change to if False
        train = np.concatenate((train, dev))
    train_squeezed = ML_util.squeeze_clusters(train)
    train_features, train_labels = ML_util.split_features(train_squeezed)
    train_features = np.nan_to_num(train_features)
    train_features = np.clip(train_features, -INF, INF)

    if shuffle_labels:
        np.random.shuffle(train_labels)

    print('Scaling data...')
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                 random_state=seed, class_weight='balanced')
    print('Fitting Random Forest model...')
    start = time.time()
    clf.fit(train_features, train_labels)
    end = time.time()
    print('Fitting took %.2f seconds' % (end - start))

    return clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF trainer\n")

    parser.add_argument('--dataset_path', type=str, help='path to the dataset, assume it was created',
                        default='../data_sets/complete_0/spat_tempo/0_0.60.20.2/')
    parser.add_argument('--n_estimators', type=int, help='n_estimators value for RF and GB', default=10)
    parser.add_argument('--max_depth', type=int, help='max_depth value for RF and GB', default=10)
    parser.add_argument('--min_samples_split', type=int, help='min_samples_split value for RF', default=4)
    parser.add_argument('--min_samples_leaf', type=int, help='min_samples_leaf value for RF', default=2)
    parser.add_argument('--seed', type=int, help='seed', default=0)

    args = parser.parse_args()

    dataset_path = args.dataset_path

    n_estimators = args.n_estimators
    max_depth = args.max_depth
    min_samples_split = args.min_samples_split
    min_samples_leaf = args.min_samples_leaf

    seed = args.seed

    run(n_estimators, max_depth, min_samples_split, min_samples_leaf, dataset_path, seed)
