import numpy as np
from gs_svm import grid_search
import argparse
import ML_util
from itertools import chain, combinations

# indices of the features in the data
indices = [[0, 1, 2, 3], [4, 5], [6, 7], [8, 9, 10], [11], [12, 13, 14]]
# name of each feature, corresponding to the indices list
names = ['time lag', 'spatial dispersion', 'direction agreeableness', 'graph speeds', 'channels contrast',
         'geometrical']


def powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def remove_features(keep, data):
    clusters = []
    for cluster in data:
        clusters.append(cluster[:, keep])
    return np.asarray(clusters)


def get_ref_score(ref_comb, results):
    for comb, acc, pyr_acc, in_acc in results:
        if list(comb) == ref_comb:
            return acc, pyr_acc, in_acc
    raise AssertionError


def calc_score(results):
    # TODO: consider taking into account PYR/IN ratio for general score
    chance = 50
    print()
    for fet in range(len(names)):
        gen_scores = []
        pyr_scores = []
        in_scores = []
        for comb, acc, pyr_acc, in_acc in results:
            if fet not in comb:
                continue
            if len(comb) == 1:
                gen_scores.append(acc - chance)
                pyr_scores.append(pyr_acc - chance)
                in_scores.append(in_acc - chance)
            else:
                ref_comb = list(comb)
                ref_comb.remove(fet)
                gen_ref, pyr_ref, in_ref = get_ref_score(ref_comb, results)
                gen_scores.append(acc - gen_ref)
                pyr_scores.append(pyr_acc - pyr_ref)
                in_scores.append(in_acc - in_ref)
        print('----------------')
        print(f"The score of feature {names[fet]} is: General={sum(gen_scores) / len(gen_scores)}; "
              f"PYR={sum(pyr_scores) / len(pyr_scores)};"
              f" IN={sum(in_scores) / len(in_scores)}")


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
        message += 'accuracy on PYR is %.3f ;' % pyr_acc
        message += 'accuracy on IN is %.3f' % in_acc
        print(message)


def feature_dropping(dataset_path, verbos, saving_path, min_gamma, max_gamma, num_gamma, min_c, max_c, num_c, kernel):
    train, dev, test, _, _, _ = ML_util.get_dataset(dataset_path)

    combs = powerset(list(range(len(names))))

    accs = []

    for comb in combs:  # go over combinations
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
        _, acc, pyr_acc, in_acc, _, _ = grid_search(dataset_path, verbos, saving_path, min_gamma, max_gamma, num_gamma,
                                                    min_c, max_c, num_c, kernel, 5, train=train_up, dev=dev_up,
                                                    test=test_up)
        accs.append((comb, acc, pyr_acc, in_acc))

    print_results(accs)
    calc_score(accs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM feature dropping\n")

    parser.add_argument('--dataset_path', type=str, help='path to the dataset, assume it was created',
                        default='../data_sets/0_0.60.20.2/')
    parser.add_argument('--verbos', type=bool, help='verbosity level (bool)', default=True)
    parser.add_argument('--saving_path', type=str, help='path to save graphs, assumed to be created',
                        default='../graphs/')
    parser.add_argument('--min_gamma', type=int, help='minimal power of gamma (base 10)', default=-7)
    parser.add_argument('--max_gamma', type=int, help='maximal power of gamma (base 10)', default=-1)
    parser.add_argument('--num_gamma', type=int, help='number of gamma values', default=7)
    parser.add_argument('--min_c', type=int, help='minimal power of C (base 10)', default=0)
    parser.add_argument('--max_c', type=int, help='maximal power of C (base 10)', default=6)
    parser.add_argument('--num_c', type=int, help='number of C values', default=7)
    parser.add_argument('--kernel', type=str,
                        help='kernel for SVM (notice that different kernels than rbf might require more parameters)',
                        default='rbf')

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

    feature_dropping(dataset_path, verbos, saving_path, min_gamma, max_gamma, num_gamma, min_c, max_c, num_c, kernel)
