import torch
import torch.nn as nn
import ML_util
from NN_model import Net
from NN_trainer import SupervisedTrainer
import numpy as np
import argparse

EPOCHS = 100  # using patience we will rarely go over 50 and commonly stay in the 10-20 area
PATIENCE = 10
BATCH_SIZE = 32
LR = 0.01584893192461114
opt = 'adam'
N1 = 32
N2 = 256
F1 = F2 = 'sigmoid'
CLASSES = 2
FEATURES = 15


def repredict(probs, thr=0.75):
    """
    Repredict using only the predictions with confidence higher than thr
    probs (array of size (n, CLASSES)): Indicates the confidence for each class on a specific feature chunck
    thr (float, optional): The threshold for removing predictions
    """
    probs_up = probs[probs.max(dim=1)[0] >= thr]
    probs_up = probs[probs.max(dim=1)[0] <= thr + 0.1]
    if len(probs) == 0:  # return regular prediction
        prob = torch.mean(probs, dim=0)
        arg_max = torch.argmax(prob)
        return arg_max
    predictions = probs_up.argmax(dim=1)
    return torch.argmax(predictions)


def evaluate_predictions(model, data, verbos=True):
    total = len(data)
    total_pyr = 0
    total_in = 0
    correct_pyr = 0
    correct_in = 0
    correct_prob = 0
    incorrect_prob = 0
    pyr_prob = 0
    in_prob = 0
    correct_chunks = 0
    total_chunks = 0
    correct_clusters = 0
    for cluster in data:
        input, label = ML_util.parse_test(cluster)
        total_pyr += 1 if label == 1 else 0
        total_in += 1 if label == 0 else 0
        prediction, prob, raw = model.predict(input)
        # if prob < 0.7: # this doesn't seem to improve so it is commented out
        #    prediction = repredict(raw)
        all_predictions = raw.argmax(dim=1)
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

    if verbos:
        print('Number of correct classified clusters is %d, which is %.4f%%' % (
            correct_clusters, 100 * correct_clusters / total))
        print('Number of correct classified feature chunks is %d, which is %.4f%%' % (
            correct_chunks, 100 * correct_chunks / total_chunks))
        print('Test set consists of %d pyramidal cells and %d interneurons' % (total_pyr, total_in))
        print('%.4f%% of pyramidal cells classified correctly' % (pyr_percent))
        print('%.4f%% of interneurons classified correctly' % (in_percent))
        print('mean confidence in correct clusters is %.4f and in incorrect clusters is %.4f' % (
            correct_prob, incorrect_prob))
        print('mean confidence for pyramidal is %.4f and for interneurons is %.4f' % (pyr_prob, in_prob))

    return 100 * correct_clusters / total, 100 * correct_chunks / total_chunks, 100 * correct_pyr / total_pyr, \
           100 * correct_in / total_in, 100 * correct_prob, 100 * incorrect_prob, 100 * pyr_prob, 100 * in_prob


def run(epochs, patience, batch_size, learning_rate, optimizer, n1, n2, f1, f2, classes, features, dataset_path,
        loading_path, saving_path):
    """
    runner function of the neural network module, see help for explanations regarding parameters
    """
    train, dev, test, _, _, _ = ML_util.get_dataset(dataset_path)
    train_squeezed = ML_util.squeeze_clusters(train)
    dev_squeezed = ML_util.squeeze_clusters(dev)

    # create a balanced loss function for the train set
    one_label = train_squeezed[train_squeezed[:, -1] == 1]
    # ratio of pyramidal waveforms note that it is over all waveforms and not specifically over clusters (each
    # cluster can have different number of waveforms)
    ratio = one_label.shape[0] / train_squeezed.shape[0]

    class_weights = torch.tensor(
        [ratio / (1 - ratio), 1.0])  # crucial as there is overrepresentation of pyramidal neurons in the data
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # create a balanced loss function for the dev set
    one_label = dev_squeezed[dev_squeezed[:, -1] == 1]
    ratio = one_label.shape[0] / dev_squeezed.shape[0]
    class_weights = torch.tensor([ratio / (1 - ratio), 1.0])
    eval_criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainer = SupervisedTrainer(criterion=criterion, batch_size=batch_size, patience=patience,
                                eval_criterion=eval_criterion, path=saving_path)

    model = Net(n1, n2, f1, f2, features, classes)
    if loading_path is None:
        print('Starting training...')
        best_epoch = trainer.train(model, train_squeezed, num_epochs=epochs, dev_data=dev_squeezed,
                                   learning_rate=learning_rate, optimizer=optimizer)

        print('best epoch was %d' % best_epoch)
        trainer.load_model(model, epoch=best_epoch)
    else:
        print('Loading model...')
        trainer.load_model(model, path=loading_path)

    return evaluate_predictions(model, test, verbos=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Network\n")

    parser.add_argument('--epochs', type=int, help='number of epochs (times to go over the data)', default=EPOCHS)
    parser.add_argument('--patience', type=int, help='number of epochs to tolerate with no improvement on the dev set',
                        default=PATIENCE)
    parser.add_argument('--batch_size', type=int, help='number of examples in a batch', default=BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, help='learning rate for the optimizer', default=LR)
    parser.add_argument('--optimizer', type=str, help='optimizer to use, can be sgd, adam or adadelta', default=opt)
    parser.add_argument('--n1', type=int, help='size of the first hidden layer', default=N1)
    parser.add_argument('--n2', type=int, help='size of the second hidden layer', default=N2)
    parser.add_argument('--f1', type=str, help='activation function before first hidden layer', default=F1)
    parser.add_argument('--f2', type=str, help='oactivation function before second hidden layer', default=F2)
    parser.add_argument('--classes', type=int, help='size of the output layer (number of classes)', default=CLASSES)
    parser.add_argument('--features', type=int, help='size of the input layer (number of features)', default=FEATURES)
    parser.add_argument('--dataset_path', type=str, help='path to the dataset, assume it was created',
                        default='../data_sets/complete_0/spatial/200_0.60.20.2/')
    parser.add_argument('--loading_path', type=str, help='path to a trained model to evaluate', default=None)
    parser.add_argument('--saving_path', type=str, help='path to save models while training, assumes path exsists',
                        default='../saved_models/')

    args = parser.parse_args()

    epochs = args.epochs
    patience = args.patience
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    optimizer = args.optimizer
    n1 = args.n1
    n2 = args.n2
    f1 = args.f1
    f2 = args.f2
    classes = args.classes
    features = args.features
    dataset_path = args.dataset_path
    loading_path = args.loading_path
    saving_path = args.saving_path

    run(epochs, patience, batch_size, learning_rate, optimizer, n1, n2, f1, f2, classes, features, dataset_path,
        loading_path, saving_path)
