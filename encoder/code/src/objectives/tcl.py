import math
import torch
from torch import nn
import torch.nn.functional as F


class TCLLoss(object):

    def __init__(self, logits, labels):
        super().__init__()
        self.logits = logits
        self.labels = labels

    @property
    def acc(self):
        return self._accuracy

    def get_loss(self):
        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(self.logits, self.labels) # input, target

        # Calculate accuracy
        correct_prediction = (torch.argmax(self.logits, 1) == self.labels)
        self._accuracy = correct_prediction.sum()/float(len(correct_prediction))
        return loss
