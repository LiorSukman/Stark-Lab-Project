import torch
import torch.nn as nn
import torch.optim as optim
import ML_util
from NN_model import Net
from NN_trainer import SupervisedTrainer
import numpy as np
import argparse

N = 50

EPOCHS = 100 # using patience we will rarely go over 50 and commonly stay in the 10-20 area 
PATIENCE = 10
BATCH_SIZE = 32
LR = 0.001
CLASSES = 2
FEATURES = 15

def evaluate_predictions(model, data):
    total = len(data)
    total_pyr = total_in = correct_pyr = correct_in = 0
    correct_prob = incorrect_prob = pyr_prob = in_prob = 0
    correct_chunks = total_chunks = correct_clusters = 0
    for cluster in data:
        input, label = ML_util.parse_test(cluster)
        total_pyr += 1 if label == 1 else 0
        total_in += 1 if label == 0 else 0
        prediction, prob, raw = model.predict(input)
        all_predictions = raw.argmax(dim = 1)
        total_chunks += len(all_predictions)
        correct_chunks += len(all_predictions[all_predictions == label])
        correct_clusters += 1 if prediction == label else 0
        correct_pyr += 1 if prediction == label and label == 1 else 0
        correct_in += 1 if prediction == label and label == 0 else 0

        if prediction == label:
            correct_prob += prob
        else:
            incorrect_prob += prob
        if label == 1:
            pyr_prob += prob
        else:
            in_prob += prob
    correct_prob = correct_prob / correct_clusters
    incorrect_prob = incorrect_prob / (total - correct_clusters)
    pyr_prob = pyr_prob / total_pyr
    in_prob = in_prob / total_in
        
    pyr_percent = float('nan') if total_pyr == 0 else 100 * correct_pyr / total_pyr
    in_percent = float('nan') if total_in == 0 else 100 * correct_in / total_in
    return 100 * correct_clusters / total, 100 * correct_chunks / total_chunks, pyr_percent, in_percent
        

def run(n1, n2, f1, f2, train, dev, test, lr, opt, epochs, patience, batch_size, classes, features):
    """
    runner function for the NN model,
    creates a model using the given parameters (n1, n2, f1, f2, classes, features),
    trains it using the given parameters (lr, opt, epochs, patience, batch_size)
    """

    train_squeezed = ML_util.squeeze_clusters(train)
    dev_squeezed = ML_util.squeeze_clusters(dev)

    one_label = train_squeezed[train_squeezed[:,-1] == 1]
    #ratio of pyramidal waveforms
    #note that it is over all waveforms and not specifically over clusters (each cluster can have different number of waveforms)
    ratio = one_label.shape[0] / train_squeezed.shape[0] 
            
    class_weights = torch.tensor([ratio / (1 - ratio), 1.0]) #crucial as there is overrepresentation of pyramidal neurons in the data
    criterion = nn.CrossEntropyLoss(weight = class_weights)

    one_label = dev_squeezed[dev_squeezed[:,-1] == 1]
    ratio = one_label.shape[0] / dev_squeezed.shape[0]
    class_weights = torch.tensor([ratio / (1 - ratio), 1.0])
    eval_criterion = nn.CrossEntropyLoss(weight = class_weights)

    trainer = SupervisedTrainer(criterion = criterion, batch_size = batch_size, patience = patience, eval_criterion = eval_criterion)

    model = Net(n1, n2, f1, f2, features, classes)

    print('Starting training...')
    best_epoch = trainer.train(model, train_squeezed, num_epochs = epochs, dev_data = dev_squeezed, learning_rate = lr, optimizer = opt)

    print('best epoch was %d' % (best_epoch))
    trainer.load_model(model, epoch = best_epoch)

    return evaluate_predictions(model, test)

def param_search(dataset_path, verbos, ns_min, ns_max, ns_num, lrs_min, lrs_max, lrs_num, n, default_lr, default_opt,
                 epochs, patience, batch_size, classes, features):
    """
    hyperparamater search function for the NN
    see help for explanation about the features
    """
    
    train, dev, test = ML_util.get_dataset(dataset_path)

    new_data = np.concatenate((train, dev))
    new_train, new_dev, new_test = ML_util.split_data(new_data, data_name = 'NN', verbos = verbos)
    
    accuracy, chunck_accuracy, pyr_accuracy, in_accuracy = 0,0,0,0
    nonlinearity = ['relu', 'sigmoid', 'tanh', 'none']
    ns = np.logspace(ns_min, ns_max, ns_num, base = 2).astype('int')
    lrs = np.logspace(lrs_min, lrs_max, lrs_num)
    best_val = (0,0,0,0)
    opts = ['sgd', 'adam', 'adadelta']
    accs = []
    best_accuracy = 0
    # phase 1
    for n1 in ns:
        for n2 in ns:
            for f1 in nonlinearity:
                for f2 in nonlinearity:
                    accuracy = 0
                    for i in range(n):
                        a, b, c, d = run(n1, n2, f1, f2, new_train, new_dev, new_test, default_lr, default_opt, epochs, patience, batch_size, classes, features)
                        accuracy += a / n
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_val = (n1, n2, f1, f2)
                    accs.append(accuracy)
                    
    if verbos:
        print('results for phase 1:')
        print('n1: %d, n2: %d, f1: %s, f2: %s' % best_val)
        print('best accuracy:', best_accuracy)

    best_n1, best_n2, best_f1, best_f2 = best_val
    accs = []
    best_accuracy = 0

    # phase 2
    for opt in opts:
        if opt == 'adadelta':
            accuracy = 0
            for i in range(n):
                a, b, c, d = run(best_n1, best_n2, best_f1, best_f2, new_train, new_dev, new_test, lr, opt, epochs, patience, batch_size, classes, features)
                accuracy += a / n
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_val = (lr, opt)
            accs.append(accuracy)
            continue
        else:
            for lr in lrs:  
                accuracy = 0
                for i in range(n):
                    a, b, c, d = run(best_n1, best_n2, best_f1, best_f2, new_train, new_dev, new_test, lr, opt, epochs, patience, batch_size, classes, features)
                    accuracy += a / n
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_val = (lr, opt)
                accs.append(accuracy)

    if verbos:
        print('results for phase 2:')
        print('lr: %.6f, optimizer: %s' % best_val)
        print('best accuracy:', best_accuracy)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NN hyperparameter search\n")

    parser.add_argument('--dataset_path', type=str, help='path to the dataset, assume it was created', default = '../data_sets/0_0.60.20.2/')
    parser.add_argument('--verbos', type=bool, help='verbosity level (bool)', default = True)
    parser.add_argument('--ns_min', type=int, help='minimal power for hidden layers size (base 2)', default = 1)
    parser.add_argument('--ns_max', type=int, help='maximal power for hidden layers size (base 2)', default = 8)
    parser.add_argument('--ns_num', type=int, help='number of hidden layer sizes to check', default = 8)
    parser.add_argument('--default_lr', type=float, help='default learning rate for phase 1', default = 0.001)
    parser.add_argument('--default_opt', type=str, help='default optimizer for phase 1', default = 'sgd')
    parser.add_argument('--lrs_min', type=int, help='minimal power for learning rate (base 10)', default = -5)
    parser.add_argument('--lrs_max', type=int, help='maximal power for learning rate (base 10)', default = -1)
    parser.add_argument('--lrs_num', type=int, help='number of learning rates to check', default = 16)
    parser.add_argument('--n', type=int, help='number of repetitions', default = 1)
    parser.add_argument('--epochs', type=int, help='number of epochs (times to go over the data)', default = EPOCHS)
    parser.add_argument('--patience', type=int, help='number of epochs to tolerate with no improvement on the dev set', default = PATIENCE)
    parser.add_argument('--batch_size', type=int, help='number of examples in a batch', default = BATCH_SIZE)
    parser.add_argument('--classes', type=int, help='size of the output layer (number of classes)', default = CLASSES)
    parser.add_argument('--features', type=int, help='size of the input layer (number of features)', default = FEATURES)

    args = parser.parse_args()
 
    dataset_path = args.dataset_path
    verbos = args.verbos
    ns_min = args.ns_min
    ns_max = args.ns_max
    ns_num = args.ns_num
    default_lr = args.default_lr
    default_opt = args.default_opt
    lrs_min = args.lrs_min
    lrs_max = args.lrs_max
    lrs_num = args.lrs_num
    n = args.n
    epochs = args.epochs
    patience = args.patience
    batch_size = args.batch_size
    classes = args.classes
    features = args.features
    
    param_search(dataset_path, verbos, ns_min, ns_max, ns_num, lrs_min, lrs_max, lrs_num, n, default_lr, default_opt,
                 epochs, patience, batch_size, classes, features)

