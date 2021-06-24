import argparse
import ML_util
import os

from constants import SPATIAL, MORPHOLOGICAL, TEMPORAL, SPAT_TEMPO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dataset_creator\n")

    parser.add_argument('--per_train', type=float, help='percentage of the data to be in train set', default=0.6)
    parser.add_argument('--per_dev', type=float, help='percentage of the data to be in development set', default=0.2)
    parser.add_argument('--per_test', type=float, help='percentage of the data to be in test set', default=0.2)
    parser.add_argument('--datasets', type=str, help='path to data dirs', default='datas.txt')
    parser.add_argument('--should_filter', type=bool, help='filter unlabeled units out', default=True)
    parser.add_argument('--verbose', type=bool, help='print information about datasets', default=True)
    parser.add_argument('--group_split', type=bool, help='whether to split based on animals or on units', default=False)
    parser.add_argument('--save_path', type=str, help='path to save datasets, make sure path exists',
                        default='../data_sets')
    parser.add_argument('--keep', type=int, help='indices to keep, make sure to put -1 in there for the label',
                        default=[])
    parser.add_argument('--all', type=bool, help='whether to create all the possible datasets, will be saved under '
                                                 '../data_sets', default=True)

    args = parser.parse_args()

    per_train = args.per_train
    per_dev = args.per_dev
    per_test = args.per_test
    datasets = args.datasets
    should_filter = args.should_filter
    save_path = args.save_path
    verbose = args.verbose
    keep = args.keep
    group_split = args.group_split

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    if not args.all:
        ML_util.create_datasets(per_train=per_train, per_dev=per_dev, per_test=per_test,
                                datasets=datasets, should_filter=should_filter,
                                save_path=save_path, verbos=verbose, keep=keep)
    else:
        restrictions = ['complete', 'no_small_sample']
        modalities = [('spatial', SPATIAL), ('morphological', MORPHOLOGICAL), ('temporal', TEMPORAL),
                      ('spat_tempo', SPAT_TEMPO)]
        for r in restrictions:
            new_path = save_path + f"/{r}"
            if not os.path.isdir(new_path):
                os.mkdir(new_path)
            for name, places in modalities:
                new_new_path = new_path + f"/{name}/"
                if not os.path.isdir(new_new_path):
                    os.mkdir(new_new_path)
                keep = places
                ML_util.create_datasets(per_train=per_train, per_dev=per_dev, per_test=per_test,
                                        datasets=datasets, should_filter=should_filter, group_split=group_split,
                                        save_path=new_new_path, verbos=verbose, keep=keep, mode=r)


