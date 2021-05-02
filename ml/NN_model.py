import torch
import torch.nn as nn
import torch.nn.functional as F

vote_options = ['single', 'separated']


# simple neural classifier
class Net(nn.Module):
    def __init__(self, n1, n2, f1, f2, num_features, num_classes, vote='single'):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(num_features, n1)

        if f1 == 'relu':
            self.nl1 = nn.ReLU()
        elif f1 == 'sigmoid':
            self.nl1 = nn.Sigmoid()
        elif f1 == 'tanh':
            self.nl1 = nn.Tanh()
        else:
            self.nl1 = lambda a: a

        self.layer2 = nn.Linear(n1, n2)

        if f2 == 'relu':
            self.nl2 = nn.ReLU()
        elif f2 == 'sigmoid':
            self.nl2 = nn.Sigmoid()
        elif f2 == 'tanh':
            self.nl2 = nn.Tanh()
        else:
            self.nl2 = lambda a: a

        self.layer3 = nn.Linear(n2, num_classes)

        self.vote = vote

        if vote not in vote_options:
            raise Exception('Vote paramater must be in' + str(vote_options))

    def change_vote(self, vote):
        if vote not in vote_options:
            raise Exception('Vote paramater must be in' + str(vote_options))
        if vote == self.vote:
            raise Warning('Vote parameter did not change as current and new value are the same')
        self.vote = vote

    def forward(self, x):
        """
         forward pass
       """
        x = self.layer1(x)
        x = self.nl1(x)
        x = self.layer2(x)
        x = self.nl2(x)
        x = self.layer3(x)
        return x

    def predict(self, x):
        """
      predict a unit using the models voting method 
      """
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
            x = F.softmax(x, dim=1)
            if self.vote == 'single':
                probs = torch.mean(x, dim=0)
                pred = torch.argmax(probs)
                prob = probs[pred]
            elif self.vote == 'separated':
                votes = torch.argmax(x, dim=1).float()
                prob = torch.mean(votes)
                pred = round(prob.item())
                prob = max(prob, 1 - prob)
        return (pred, prob, x)
