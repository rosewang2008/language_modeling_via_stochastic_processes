import math
import torch
from torch import nn
import torch.nn.functional as F


class PCLLoss(object):

    def __init__(self, logits, labels):
        super().__init__()
        self.logits = logits
        self.labels = labels.float()
        self.sigmoid = nn.Sigmoid()

    @property
    def acc(self):
        return self._accuracy

    def get_loss(self):
        loss_f = nn.BCELoss()
        ps = self.sigmoid(self.logits)[:,0]
        loss = loss_f(ps, self.labels) # input, target
        # Calculate accuracy
        predicted = ps >= 0.5
        correct_prediction = (predicted == self.labels)
        self._accuracy = correct_prediction.sum()/float(len(correct_prediction))
        return loss
