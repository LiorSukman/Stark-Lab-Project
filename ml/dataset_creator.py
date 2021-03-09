import argparse
import ML_util
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="dataset_creator\n")
    
    parser.add_argument('--per_train', type=float, help='percentage of the data to be in train set', default = 0.6)
    parser.add_argument('--per_dev', type=float, help='percentage of the data to be in developemnt set', default = 0.2)
    parser.add_argument('--per_test', type=float, help='percentage of the data to be in test set', default = 0.2)
    parser.add_argument('--datasets', type=str, help='path to data dirs', default='datas.txt')
    parser.add_argument('--should_filter', type=bool, help='filter unlabeled units out', default = True)
    parser.add_argument('--verbos', type=bool, help='print information about datasets', default = False)
    parser.add_argument('--save_path', type=str, help='path to save datasets, make sure path exsists', default='../data_sets',)
    parser.add_argument('--keep', type=int, help='indeces to keep, make sure to put -1 in there for the label', default=[])

    args = parser.parse_args()
    
    per_train = args.per_train
    per_dev = args.per_dev
    per_test = args.per_test
    datasets = args.datasets
    should_filter = args.should_filter
    save_path = args.save_path
    verbos = args.verbos
    keep = args.keep
    ML_util.create_datasets(per_train = per_train, per_dev = per_dev, per_test = per_test,
                            datasets = datasets, should_filter = should_filter,
                            save_path = save_path, verbos = verbos, keep = keep)
