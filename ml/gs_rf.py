from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
import argparse

import ML_util

N = 10

def evaluate_predictions(model, clusters, verbos = False):
    total = len(clusters)
    total_pyr = total_in = correct_pyr = correct_in = correct_chunks = correct_clusters = total_chunks = 0
    for cluster in clusters:
        features, labels = ML_util.split_features(cluster)
        label = labels[0] #as they are the same for all the cluster
        total_pyr += 1 if label == 1 else 0
        total_in += 1 if label == 0 else 0
        preds = model.predict(features)
        prediction = round(preds.mean())
        total_chunks += preds.shape[0]
        correct_chunks += preds[preds == label].shape[0]
        correct_clusters += 1 if prediction == label else 0
        correct_pyr += 1 if prediction == label and label == 1 else 0
        correct_in += 1 if prediction == label and label == 0 else 0

    if verbos:
        print('Number of correct classified clusters is %d, which is %.4f%s' % (correct_clusters, 100 * correct_clusters / total, '%'))
        print('Number of correct classified chunks is %d, which is %.4f%s' % (correct_chunks, 100 * correct_chunks / total_chunks, '%'))
        print('Test set consists of %d pyramidal cells and %d interneurons' % (total_pyr, total_in))
        pyr_percent = float('nan') if total_pyr == 0 else 100 * correct_pyr / total_pyr
        in_percent = float('nan') if total_in == 0 else 100 * correct_in / total_in
        print('%.4f%s of pyrmidal cells classified correctly' % (pyr_percent, '%'))
        print('%.4f%s of interneurons classified correctly' % (in_percent, '%'))
    return correct_clusters, correct_clusters / total

def grid_search(dataset_path, verbos, n_estimators_min, n_estimators_max, n_estimators_num,
                max_depth_min, max_depth_max, max_depth_num, min_samples_splits_min, min_samples_splits_max,
                min_samples_splits_num, min_samples_leafs_min, min_samples_leafs_max, min_samples_leafs_num, n):
    """
    grid search runner function
    see help for parameter explanations
    """
    
    train, dev, test = ML_util.get_dataset(dataset_path)

    train_squeezed = ML_util.squeeze_clusters(train)
    dev_squeezed = ML_util.squeeze_clusters(dev)

    train_features, train_labels = ML_util.split_features(train_squeezed)
    dev_features, dev_labels = ML_util.split_features(dev_squeezed)

    n_estimatorss = np.logspace(n_estimators_min, n_estimators_max, n_estimators_num).astype('int')
    max_depths = np.logspace(max_depth_min, max_depth_max, max_depth_num).astype('int')
    min_samples_splits = np.logspace(min_samples_splits_min, min_samples_splits_max, min_samples_splits_num, base = 2).astype('int')
    min_samples_leafs = np.logspace(min_samples_leafs_min, min_samples_leafs_max, min_samples_leafs_num, base = 2).astype('int')

    # this is an alternative to the following part using sklearn's grid serach method
    # we use the alternative for better control, mainly running each set of parameters several times
    """
    train_dev = np.concatenate((train_squeezed, dev_squeezed))
    train_dev_features, train_dev_labels = ML_util.split_features(train_dev)
    test_inds = np.concatenate((-1 * np.ones((len(train_squeezed))), np.zeros((len(dev_squeezed)))))
    ps = PredefinedSplit(test_inds)
    
    print()
    parameters = {'n_estimators': n_estimatorss, 'max_depth': max_depths, 'min_samples_split' : min_samples_splits, 'min_samples_leaf' : min_samples_leafs}
    model = RandomForestClassifier(class_weight = 'balanced')
    clf = GridSearchCV(model, parameters, cv = ps)
    print('Starting grid search...')
    start = time.time()
    clf.fit(train_dev_features, train_dev_labels)
    end = time.time()
    print('Grid search completed in %.2f seconds, best parameters are:' % (end - start)) 
    print(clf.best_params_)

    bp = clf.best_params_
    n_estimators, max_depth, min_samples_split, min_samples_leaf = bp['n_estimators'], bp['max_depth'], bp['min_samples_split'], bp['min_samples_leaf']
    classifier = RandomForestClassifier(class_weight = 'balanced', n_estimators = n_estimators, max_depth = max_depth)#, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf) 
    classifier.fit(train_features, train_labels)
    """

    scores = []
    best_score = 0
    params = None
    for n_estimators in n_estimatorss:
        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                for min_samples_leaf in min_samples_leafs:
                    temp_score = 0
                    for i in range(n):
                        model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, class_weight = 'balanced')
                        model.fit(train_features, train_labels)
                        score = model.score(dev_features, dev_labels)
                        temp_score += score / n
                    if temp_score > best_score:
                        best_score = temp_score
                        params = (n_estimators, max_depth, min_samples_split, min_samples_leaf)
                        
                    scores.append(temp_score)

    print('Best parameters are: n_estimators %d, max_depth %d, min_samples_split %d, min_samples_leaf %d' % params)

    classifier = model = RandomForestClassifier(class_weight = 'balanced', n_estimators = params[0], max_depth = params[1], min_samples_split = params[2], min_samples_leaf = params[3])
    classifier.fit(train_features, train_labels)
    
    print()
    print('Starting evaluation on test set...')
    return evaluate_predictions(classifier, test, verbos)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Random forest grid search\n")

    parser.add_argument('--dataset_path', type=str, help='path to the dataset, assume it was created', default = '../data_sets/0_0.60.20.2/')
    parser.add_argument('--verbos', type=bool, help='verbosity level (bool)', default = True)
    parser.add_argument('--n_estimators_min', type=int, help='minimal power of n_estimators (base 10)', default = 0)
    parser.add_argument('--n_estimators_max', type=int, help='maximal power of n_estimators (base 10)', default = 3)
    parser.add_argument('--n_estimators_num', type=int, help='number of n_estimators values', default = 4)
    parser.add_argument('--max_depth_min', type=int, help='minimal power of max_depth (base 10)', default = 1)
    parser.add_argument('--max_depth_max', type=int, help='maximal power of max_depth (base 10)', default = 5)
    parser.add_argument('--max_depth_num', type=int, help='number of max_depth values', default = 5)
    parser.add_argument('--min_samples_splits_min', type=int, help='minimal power of min_samples_splits (base 2)', default = 1)
    parser.add_argument('--min_samples_splits_max', type=int, help='maximal power of min_samples_splits (base 2)', default = 5)
    parser.add_argument('--min_samples_splits_num', type=int, help='number of min_samples_splits values', default = 5)
    parser.add_argument('--min_samples_leafs_min', type=int, help='minimal power of min_samples_leafs (base 2)', default = 0)
    parser.add_argument('--min_samples_leafs_max', type=int, help='maximal power of min_samples_leafs (base 2)', default = 5)
    parser.add_argument('--min_samples_leafs_num', type=int, help='number of min_samples_leafs values', default = 6)
    parser.add_argument('--n', type=int, help='number of repetitions', default = N)

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
