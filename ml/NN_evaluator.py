import torch
import torch.optim as optim
import torch.nn as nn
from ML_util import create_batches

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        criterion (optional): loss for evaluator
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, criterion = nn.CrossEntropyLoss(), batch_size = 64):
        self.criterion = criterion
        self.batch_size = batch_size

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model: model to evaluate
            data: dataset to evaluate against

        Returns:
            loss (float): average loss of the given model on the given dataset
            accuracy (float): accuracy of the given model on the given dataset
        """
        model.eval()

        criterion = self.criterion
        correct = 0
        total = 0
        total_batch = 0
        loss = 0

        device = torch.device('cuda') if torch.cuda.is_available() else -1

        with torch.no_grad():
            batches = create_batches(data, self.batch_size) 
            for batch in batches:
                input_vars, labels = batch

                outputs = model(input_vars)

                loss += criterion(outputs, labels) 

                # Evaluation
                for step, step_output in enumerate(outputs):
                    label = labels[step]
                    
                    is_correct = 1 if label == torch.argmax(step_output) else 0
                    correct += is_correct
                    total += 1
                total_batch += 1

        if total == 0:
            accuracy = float('nan')
            avg_loss = float('nan')
        else:
            accuracy = 100 * correct / total
            avg_loss = loss / total_batch

        return avg_loss, accuracy
