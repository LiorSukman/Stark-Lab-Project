import argparse
import ML_util
import os

from constants import SPATIAL, MORPHOLOGICAL, TEMPORAL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dataset_creator\n")

    parser.add_argument('--per_train', type=float, help='percentage of the data to be in train set', default=0.6)
    parser.add_argument('--per_dev', type=float, help='percentage of the data to be in development set', default=0.2)
    parser.add_argument('--per_test', type=float, help='percentage of the data to be in test set', default=0.2)
    parser.add_argument('--datasets', type=str, help='path to data dirs', default='datas.txt')
    parser.add_argument('--should_filter', type=bool, help='filter unlabeled units out', default=True)
    parser.add_argument('--verbos', type=bool, help='print information about datasets', default=True)
    parser.add_argument('--save_path', type=str, help='path to save datasets, make sure path exists',
                        default='../data_sets', )
    parser.add_argument('--keep', type=int, help='indices to keep, make sure to put -1 in there for the label',
                        default=[])

    args = parser.parse_args()

    per_train = args.per_train
    per_dev = args.per_dev
    per_test = args.per_test
    datasets = args.datasets
    should_filter = args.should_filter
    save_path = args.save_path
    verbos = args.verbos
    keep = MORPHOLOGICAL#args.keep

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    ML_util.create_datasets(per_train=per_train, per_dev=per_dev, per_test=per_test,
                            datasets=datasets, should_filter=should_filter,
                            save_path=save_path, verbos=verbos, keep=keep)
