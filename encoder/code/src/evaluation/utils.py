""" Fuctions for evaluation
    This software includes the work that is distributed in the Apache License 2.0

    from https://github.com/ilkhem/icebeem/blob/3a7b1bfe7b62fdcdd753862773dd5e607e3fe4f9/models/tcl/tcl_eval.py
"""

import sys

import numpy as np
# import tensorflow as tf
from sklearn.metrics import confusion_matrix
import torch


def _squared(source):
    return source ** 2

def _abs(source):
    return torch.abs(source)

def calc_confusion_matrix(pred, label, normalize_confmat=True):
    """ Calculate confusion matrix
    Args:
        pred: [Ndata x Nlabel]
        label: [Ndata x Nlabel]
    Returns:
        conf: confusion matrix
    """
    # Confusion matrix ----------------------------------------
    conf = confusion_matrix(label[:], pred[:]).astype(np.float32)
    # Normalization
    if normalize_confmat:
        for i in range(conf.shape[0]):
            conf[i, :] = conf[i, :] / np.sum(conf[i, :])
    return conf
