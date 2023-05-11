import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.models.utils import weights_init

# def maxout(y, k):
#     #   y: data tensor
#     #   k: number of affine feature maps
#     input_shape = y.get_shape().as_list()
#     ndim = len(input_shape)
#     ch = input_shape[-1]
#     assert ndim == 4 or ndim == 2
#     assert ch is not None and ch % k == 0
#     if ndim == 4:
#         y = tf.reshape(y, [-1, input_shape[1], input_shape[2], ch / k, k])
#     else:
#         y = tf.reshape(y, [-1, int(ch / k), k])
#     y = tf.reduce_max(y, ndim)
#     return y
class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert x.shape[1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[1], self._pool_size)
        m, i = x.view(*x.shape[:1], x.shape[1] // self._pool_size,
                      self._pool_size, *x.shape[2:]).max(2)
        return m

class Absolute(nn.Module):
    def __init__(self):
        super(Absolute, self).__init__()

    def forward(self, input):
        return torch.abs(input)


class TCLModel(nn.Module):

    def __init__(self, data_dim, num_classes, num_components, num_layers,
                 wd=1e-4, maxout_k=2, MLP_trainable=True, feature_nonlinearity='abs'):
        """
        Args:
            x: data holder.
            list_hidden_nodes: number of nodes for each layer. 1D array [num_layer]
            num_class: number of classes of MLR
            wd: (option) parameter of weight decay (not for bias)
            maxout_k: (option) number of affine feature maps
            MLP_trainable: (option) If false, fix MLP4 layers
            feature_nonlinearity: (option) Nonlinearity of the last hidden layer (feature value)
        Returns:
            logits: logits tensor:
            feat: feature tensor
        """
        super(TCLModel, self).__init__()

        self.data_dim = data_dim
        self.list_hidden_nodes = [num_components * 2] * (num_layers - 1) + [num_components]
        self.num_layers = len(self.list_hidden_nodes)
        self.num_classes = num_classes

        self.wd = wd
        self.maxout_k = maxout_k
        self.MLP_trainable = MLP_trainable
        self.feature_nonlinearity = feature_nonlinearity

        # Construct model
        self.feature_extractor = self.create_feature_extractor()
        self.predictor = self.create_prediction_head()

        # Turn off grad if needed
        if not self.MLP_trainable:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Set biases to 0
        self.feature_extractor.apply(weights_init)
        self.predictor.apply(weights_init)


    def set_to_train(self):
        for param in self.parameters():
            param.requires_grad = True

    def create_feature_extractor(self):
        modules = []

        for ln in range(self.num_layers):
            in_dim = self.list_hidden_nodes[ln - 1] if ln > 0 else self.data_dim
            out_dim = self.list_hidden_nodes[ln]

            if ln < self.num_layers - 1:  # Increase number of nodes for maxout
                out_dim = self.maxout_k * out_dim

            layer = torch.nn.Linear(in_dim, out_dim)
            modules.append(layer)

            # Nonlinearity
            if ln < self.num_layers - 1:
                modules.append(Maxout(self.maxout_k))
                # x = maxout(x, maxout_k)
            else:  # The last layer (feature value)
                if self.feature_nonlinearity == 'abs':
                    modules.append(Absolute())
                    # x = tf.abs(x)
                else:
                    raise ValueError
        return nn.Sequential(*modules)

    def create_prediction_head(self):
        return nn.Linear(self.list_hidden_nodes[-1], self.num_classes)

    def forward(self, inputs):
        feats = self.feature_extractor(inputs) # batch_size, feat_dim
        logits = self.predictor(feats) # batch_size, classes
        return logits, feats
