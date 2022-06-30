from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import numpy as np
import time
import argparse

import ML_util
from constants import INF

N = 10

def calc_auc(clf, test_set, scaler):

    preds = []
    targets = []

    for cluster in test_set:
        features, labels = ML_util.split_features(cluster)
        features = np.nan_to_num(features)
        features = np.clip(features, -INF, INF)
        features = scaler.transform(features)
        label = labels[0]  # as they are the same for all the cluster
        prob = clf.predict_proba(features).mean(axis=0)
        pred = prob[1]
        preds.append(pred)
        targets.append(label)

    fpr, tpr, thresholds = roc_curve(targets, preds, drop_intermediate=False)  # calculate fpr and tpr values for different thresholds
    auc_val = auc(fpr, tpr)
    print(f"AUC value is: {auc_val}")
    return auc_val


def evaluate_predictions(model, clusters, names, scaler, verbos=False):
    total = len(clusters)
    if total == 0:
        return 0, 0, 0, 0
    total_pyr = total_in = correct_pyr = correct_in = correct_chunks = total_chunks = correct_clusters = 0
    for cluster, name in zip(clusters, names):
        features, labels = ML_util.split_features(cluster)
        features = np.nan_to_num(features)
        features = np.clip(features, -INF, INF)
        features = scaler.transform(features)
        label = labels[0]  # as they are the same for all the cluster
        total_pyr += 1 if label == 1 else 0
        total_in += 1 if label == 0 else 0
        try:
            preds = model.predict_proba(features)
            preds_argmax = preds.argmax(axis=1)  # support probabilistic prediction
            preds_mean = preds.mean(axis=0)
        except AttributeError:
            preds = preds_argmax = model.predict(features)  # do not support probabilistic prediction
            preds_mean = round(preds_argmax.mean())
        # prediction = round(preds_argmax.mean())
        prediction = preds_mean.argmax()
        total_chunks += preds.shape[0]
        correct_chunks += preds[preds_argmax == label].shape[0]
        correct_clusters += 1 if prediction == label else 0
        correct_pyr += 1 if prediction == label and label == 1 else 0
        correct_in += 1 if prediction == label and label == 0 else 0
        #if prediction != label:
        #    ML_util.show_cluster(name, 'PYR' if prediction == 1 else 'INT', 'PYR' if label == 1 else 'INT')

    pyr_percent = float('nan') if total_pyr == 0 else 100 * correct_pyr / total_pyr
    in_percent = float('nan') if total_in == 0 else 100 * correct_in / total_in

    if verbos:
        print('Number of correct classified clusters is %d, which is %.4f%%' % (
            correct_clusters, 100 * correct_clusters / total))
        print('Number of correct classified chunks is %d, which is %.4f%%' % (
            correct_chunks, 100 * correct_chunks / total_chunks))
        print('Test set consists of %d pyramidal cells and %d interneurons' % (total_pyr, total_in))
        print('%.4f%% of pyramidal cells classified correctly' % pyr_percent)
        print('%.4f%% of interneurons classified correctly' % in_percent)
    print(f"total_chunks is {total_chunks}, total is {total}")

    return 100 * correct_chunks / total_chunks, 100 * correct_clusters / total, pyr_percent, in_percent


def run(n_estimators, max_depth, min_samples_split, min_samples_leaf,
        dataset_path, seed, region_based=False, shuffle_labels=False):
    """
    runner function for the SVM and RF models.
    explanations about the parameters is in the help
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

    print('Evaluating predictions...')
    print('Test Evaluation:')
    chunk_per, clust_per, pyr_per, in_per = evaluate_predictions(clf, test, test_names, scaler, verbos=True)
    print('\nDev Evaluation:')
    dev_chunk_per, dev_clust_per, dev_pyr_per, dev_in_per = evaluate_predictions(clf, dev, dev_names, scaler, verbos=True)

    #calc_auc(clf, test, scaler)
    #calc_auc(clf, dev, scaler)

    return clf, clust_per, pyr_per, in_per, dev_clust_per, dev_pyr_per, dev_in_per


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
