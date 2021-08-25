import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from ML_util import create_batches
from NN_evaluator import Evaluator

import logging
import os
import random
import time
from tqdm import tqdm


class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
   supervised setting.

   Args:
      criterion (optional): loss for training, (default: CrossEntropyLoss)
      batch_size (int, optional): batch size for experiment, (default: 32)
      print_every (int, optional): number of batches to print after, (default: 100)
   """

    def __init__(self, criterion=nn.CrossEntropyLoss(), batch_size=32, random_seed=None, print_every=100,
                 eval_criterion=nn.CrossEntropyLoss(), path='../saved_models/', patience=None):
        self._trainer = "Simple Trainer"
        self.path = path
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.criterion = criterion
        self.evaluator = Evaluator(criterion=eval_criterion, batch_size=batch_size)
        self.optimizer = None
        self.patience = patience
        self.print_every = print_every

        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

    def visualize_learning(self, dev_loss, dev_acc, train_loss):
        if len(dev_loss) != 0:
            plt.plot(np.arange(1, len(dev_loss) + 1), dev_loss, 'b')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title('dev loss over epoch')
            plt.savefig(self.path + 'graph_dev_loss.png')
            plt.clf()
            plt.plot(np.arange(1, len(dev_acc) + 1), dev_acc, 'b')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.title('dev accuracy over epoch')
            plt.savefig(self.path + 'graph_dev_acc.png')
            plt.clf()
        plt.plot(np.arange(1, len(train_loss) + 1), train_loss, 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('train loss over epoch')
        plt.savefig(self.path + 'graph_train_loss.png')
        plt.clf()

    def load_model(self, model, epoch=None, path=None):
        if epoch is not None and path is not None:
            raise Exception('load_model with both epoch and path is ambiguous')
        elif epoch is not None:
            model.load_state_dict(torch.load(self.path + 'epoch' + str(epoch)))
        elif path is not None:
            model.load_state_dict(torch.load(path))
        else:
            raise Exception('load_model was called without path and without epoch')

    def _train_batch(self, input_var, labels, model):
        # Get loss
        criterion = self.criterion
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(input_var)
        loss = criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step, dev_data=None):
        log = self.logger

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        device = torch.device('cuda') if torch.cuda.is_available() else -1

        steps_per_epoch = len(data) // self.batch_size + (1 if len(data) % self.batch_size != 0 else 0)
        total_steps = steps_per_epoch * n_epochs

        best_dev_loss = float('inf')
        # best_accuracy = 0 if we would like to choose by accuracy instead of loss
        best_epoch = 0

        dev_loss_lst = []
        dev_acc_lst = []
        train_loss_lst = []

        step = start_step
        for epoch in tqdm(range(start_epoch, n_epochs + 1)):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            # create batches
            batches = create_batches(data, self.batch_size)

            for batch in batches:
                step += 1

                input_var, labels = batch  # need to make sure it is indeed var

                loss = self._train_batch(input_var, labels, model)

                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if step == 0: continue

                if step % self.print_every == 0:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = 'Progress: %d%%, Train: %.4f' % (step / total_steps * 100, print_loss_avg)
                    log.info(log_msg)

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = "Finished epoch %d: Train: %.4f" % (epoch, epoch_loss_avg)
            train_loss_lst.append(epoch_loss_avg)
            if dev_data is not None:
                dev_loss, accuracy = self.evaluator.evaluate(model, dev_data)
                # here we should update lr if we will ahve scheduler or just according to the epoch loss if we won't have dev
                log_msg += ", Dev: %.4f, Accuracy: %.4f" % (dev_loss, accuracy)
                dev_loss_lst.append(dev_loss)
                dev_acc_lst.append(accuracy / 100)
                model.train(mode=True)
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    best_epoch = epoch
                if self.patience is not None:
                    if epoch - best_epoch >= self.patience:
                        print(log_msg)
                        log.info(log_msg)
                        torch.save(model.state_dict(), self.path + 'epoch' + str(epoch))
                        print(
                            'Breaking trainer after %d epochs because no improvement happened in the last %d epochs' % (
                                epoch, self.patience))
                        break

            print(log_msg)
            log.info(log_msg)
            torch.save(model.state_dict(), self.path + 'epoch' + str(epoch))

        self.visualize_learning(dev_loss_lst, dev_acc_lst, train_loss_lst)

        return best_epoch

    def train(self, model, data, num_epochs=10, dev_data=None, optimizer=None, learning_rate=0.001):
        """ Run training for a given model.

        Args:
            model: model to run training on
            data: dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 10)
            dev_data (optional): dev Dataset (default None)
            optimizer (optional): optimizer for training (default: Optimizer(pytorch.optim.SGD))
            learning_rate (optional): learning rate of the model
        Returns:
            model: trained model.
        """

        start_epoch = 1
        step = 0
        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer == 'adadelta':
            optimizer = optim.Adadelta(model.parameters())
        self.optimizer = optimizer

        # think of adding a scheduler

        self.logger.info("Optimizer: %s" % self.optimizer)

        best_epoch = self._train_epoches(data, model, num_epochs, start_epoch, step, dev_data=dev_data)
        return best_epoch
