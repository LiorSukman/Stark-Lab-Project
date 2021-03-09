from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse

import ML_util
from VIS_heatmap import create_heatmap

def evaluate_predictions(model, clusters, verbos = False):
    total = len(clusters)
    total_pyr = 0
    total_in = 0
    correct_pyr = 0
    correct_in = 0
    correct_chunks = 0
    total_chunks = 0
    correct_clusters = 0
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

    pyr_percent = float('nan') if total_pyr == 0 else 100 * correct_pyr / total_pyr
    in_percent = float('nan') if total_in == 0 else 100 * correct_in / total_in
    
    if verbos:
        print('Number of correct classified clusters is %d, which is %.4f%%' % (correct_clusters, 100 * correct_clusters / total))
        print('Number of correct classified chunks is %d, which is %.4f%%' % (correct_chunks, 100 * correct_chunks / total_chunks))
        print('Test set consists of %d pyramidal cells and %d interneurons' % (total_pyr, total_in))
        print('%.4f%% of pyrmidal cells classified correctly' % (pyr_percent))
        print('%.4f%% of interneurons classified correctly' % (in_percent))
    return correct_clusters, 100 * correct_clusters / total, pyr_percent, in_percent

def grid_search(dataset_path, verbos, saving_path, min_gamma, max_gamma, num_gamma, min_c, max_c, num_c, kernel,
                train = None, dev = None, test = None):
    """
    grid search function for SVM
    see help for explanation about the parameters
    if, train, dev and test are given, ignores dataset_path and uses them instead
    """
    if train is None or dev is None or test is None:
        train, dev, test = ML_util.get_dataset(dataset_path)

    train_squeezed = ML_util.squeeze_clusters(train)
    train_features, train_labels = ML_util.split_features(train_squeezed)
    dev_squeezed = ML_util.squeeze_clusters(dev)

    train_dev = np.concatenate((train_squeezed, dev_squeezed))
    train_dev_features, train_dev_labels = ML_util.split_features(train_dev)
    test_inds = np.concatenate((-1 * np.ones((len(train_squeezed))), np.zeros((len(dev_squeezed)))))
    ps = PredefinedSplit(test_inds)         

    gammas = np.logspace(min_gamma, max_gamma, num_gamma)
    cs = np.logspace(min_c, max_c, num_c)
    
    print()
    parameters = {'C': cs, 'gamma': gammas}
    model = svm.SVC(kernel = 'rbf', class_weight = 'balanced')
    clf = GridSearchCV(model, parameters, cv = ps)
    print('Starting grid search...')
    start = time.time()
    clf.fit(train_dev_features, train_dev_labels)
    end = time.time()
    print('Grid search completed in %.2f seconds, best parameters are:' % (end - start)) 
    print(clf.best_params_)

    C = clf.best_params_['C']
    gamma = clf.best_params_['gamma']
    calssifier = svm.SVC(kernel = kernel, class_weight = 'balanced', C = C, gamma = gamma) # need to create another one as the other trains on both train and dev
    calssifier.fit(train_features, train_labels)

    if verbos:
        scores = clf.cv_results_['mean_test_score']
        cs = [round(v, 3) for v in cs]
        gammas = [round(v, 9) for v in gammas]
        create_heatmap(gammas, cs, 'Gamma', 'C', 'SVM Grid Search', scores.reshape((len(cs), len(gammas))), path = saving_path)

    print()
    print('Starting evaluation on test set...')
    return evaluate_predictions(calssifier, test, verbos)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM grid search\n")

    parser.add_argument('--dataset_path', type=str, help='path to the dataset, assume it was created', default = '../data_sets/0_0.60.20.2/')
    parser.add_argument('--verbos', type=bool, help='verbosity level (bool)', default = True)
    parser.add_argument('--saving_path', type=str, help='path to save graphs, assumed to be created', default = '../graphs/')
    parser.add_argument('--min_gamma', type=int, help='minimal power of gamma (base 10)', default = -9)
    parser.add_argument('--max_gamma', type=int, help='maximal power of gamma (base 10)', default = -1)
    parser.add_argument('--num_gamma', type=int, help='number of gamma values', default = 36)
    parser.add_argument('--min_c', type=int, help='minimal power of C (base 10)', default = 0)
    parser.add_argument('--max_c', type=int, help='maximal power of C (base 10)', default = 10)
    parser.add_argument('--num_c', type=int, help='number of C values', default = 44)
    parser.add_argument('--kernel', type=str, help='kernael for SVM (notice that different kernels than rbf might require more parameters)', default = 'rbf')


    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    verbos = args.verbos
    saving_path = args.saving_path
    min_gamma = args.min_gamma
    max_gamma = args.max_gamma
    num_gamma = args.num_gamma
    min_c = args.min_c
    max_c = args.max_c
    num_c = args.num_c
    saving_path = args.saving_path
    kernel = args.kernel
    
    grid_search(dataset_path, verbos, saving_path, min_gamma, max_gamma, num_gamma, min_c, max_c, num_c, saving_path, kernel)
