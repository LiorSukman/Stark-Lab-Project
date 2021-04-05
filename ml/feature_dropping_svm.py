import numpy as np
from gs_svm import grid_search
import time
import ML_util
from itertools import chain, combinations

#indices of the features in the data
indices = [[0, 1, 2, 3], [4, 5], [6, 7], [8, 9, 10], [11], [12, 13, 14]]
#name of each feature, corresponding to the indices list
names = ['time lag', 'spatial dispersion', 'direction agreeableness', 'graph speeds', 'channels contrast', 'geometrical']

def powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

def remove_features(keep, data):
    clusters = []
    for cluster in data:
        clusters.append(cluster[:, keep])
    return np.asarray(clusters)

def print_results(results):
    """
    The function receives results list and prints it 
    """
    num_features = 0
    for comb, acc, pyr_acc, in_acc in results:
        if len(comb) != num_features:
            print('----------------')
            num_features = len(comb)
            print('Results with %d feature(s)' % num_features)
        message = 'Using the features: '
        comb_fets = []
        for ind in comb:
            comb_fets.append(names[ind])
        message += str(comb_fets)
        message += ' general accuracy is: %.3f ;' % acc
        message += 'accuracy on pyr is %.3f ;' % pyr_acc
        message += 'accuracy on in is %.3f' % in_acc
        print(message)
            

def feature_dropping(dataset_path, verbos, saving_path, min_gamma, max_gamma, num_gamma, min_c, max_c, num_c, kernel):
    
    train, dev, test = ML_util.get_dataset(dataset_path)

    combinations = powerset(list(range(len(names))))

    accs = []

    for comb in combinations: # go over combinations
        inds = []
        comb_fets = []
        message = 'Used features are '
        for i in comb:  # create message and data
            inds += indices[i]
            comb_fets.append(names[i])
        message += str(comb_fets)
        print(message)
        inds.append(-1)
        train_up = remove_features(inds, train)
        dev_up = remove_features(inds, dev)
        test_up = remove_features(inds, test)
        # run grid search
        _, acc, pyr_acc, in_acc = grid_search(dataset_path, verbos, saving_path, min_gamma, max_gamma, num_gamma, min_c, max_c, num_c, saving_path, kernel, train = train_up, dev = dev_up, test = test_up)
        accs.append((comb, acc, pyr_acc, in_acc))

    print_results(accs)                        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SVM feature dropping\n")

    parser.add_argument('--dataset_path', type=str, help='path to the dataset, assume it was created', default = '../data_sets/0_0.60.20.2/')
    parser.add_argument('--verbos', type=bool, help='verbosity level (bool)', default = True)
    parser.add_argument('--saving_path', type=str, help='path to save graphs, assumed to be created', default = '../graphs/')
    parser.add_argument('--min_gamma', type=int, help='minimal power of gamma (base 10)', default = -9)
    parser.add_argument('--max_gamma', type=int, help='maximal power of gamma (base 10)', default = -1)
    parser.add_argument('--num_gamma', type=int, help='number of gamma values', default = 9)
    parser.add_argument('--min_c', type=int, help='minimal power of C (base 10)', default = 0)
    parser.add_argument('--max_c', type=int, help='maximal power of C (base 10)', default = 10)
    parser.add_argument('--num_c', type=int, help='number of C values', default = 11)
    parser.add_argument('--kernel', type=int, help='kernael for SVM (notice that different kernels than rbd might require more parameters)', default = 'rbf')


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

    feature_dropping(dataset_path, verbos, saving_path, min_gamma, max_gamma, num_gamma, min_c, max_c, num_c, saving_path, kernel)
    
