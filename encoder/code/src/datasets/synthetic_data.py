import os
import sys
import json
import torch
import random
import string
import pickle
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
from sklearn.preprocessing import scale
from torchvision import transforms
from src.datasets.utils import leaky_ReLU, sigmoidAct, generateUniformMat, pca


class SyntheticData(data.Dataset):
    """TCL Synthetic data"""

    def __init__(self, train, data_dim, n_segments, n_obs_seg,
            n_layers, seed, one_hot_labels):
        super().__init__()
        self.train = train
        self.data_dim = data_dim
        self.n_segments = n_segments
        self.n_obs_seg = n_obs_seg
        self.n_layers = n_layers
        self.seed = seed
        self.one_hot_labels = one_hot_labels
        self.observation_data, self.labels, self.source_data = self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Copied from
        https://github.com/ilkhem/icebeem/blob/3a7b1bfe7b62fdcdd753862773dd5e607e3fe4f9/data/imca.py#L399

        Returns:
            x: Observation data
            y: Labels (time segments)
            s: Source data
        """
        np.random.seed(self.seed)
        dat_all = self.gen_nonstationary_data()

        x = dat_all['obs'].T
        x, pca_params = pca(x, num_comp=x.shape[0])
        x = x.T

        if self.one_hot_labels:
            y = to_one_hot(dat_all['labels'])[0]
        else:
            y = dat_all['labels']
        s = dat_all['source']
        return x, y, s

    def gen_nonstationary_data(self, source='Laplace', NonLin='leaky',
                               negSlope=.2, Niter4condThresh=1e4):
        """
        generate multivariate data based on the non-stationary non-linear ICA model of Hyvarinen & Morioka (2016)
        INPUT
            - Ncomp: number of components (i.e., dimensionality of the data)
            - Nlayer: number of non-linear layers!
            - Nsegment: number of data segments to generate
            - NsegmentObs: number of observations per segment
            - source: either Laplace or Gaussian, denoting distribution for latent sources
            - NonLin: linearity employed in non-linear mixing. Can be one of "leaky" = leakyReLU or "sigmoid"=sigmoid
              Specifically for leaky activation we also have:
                  - negSlope: slope for x < 0 in leaky ReLU
                  - Niter4condThresh: number of random matricies to generate to ensure well conditioned
        OUTPUT:
          - output is a dictionary with the following values:
              - sources: original non-stationary source
              - obs: mixed sources
              - labels: segment labels (indicating the non stationarity in the data)
        """
        Ncomp = self.data_dim
        Nlayer = self.n_layers
        Nsegment = self.n_segments
        NsegmentObs = self.n_obs_seg

        np.random.seed(self.seed)
        # check input is correct
        assert NonLin in ['leaky', 'sigmoid']

        # generate non-stationary data:
        Nobs = NsegmentObs * Nsegment  # total number of observations
        labels = np.array([0] * Nobs)  # labels for each observation (populate below)

        # generate data, which we will then modulate in a non-stationary manner:
        if source == 'Laplace':
            dat = np.random.laplace(0, 1, (Nobs, Ncomp))
            dat = scale(dat)  # set to zero mean and unit variance
        elif source == 'Gaussian':
            dat = np.random.normal(0, 1, (Nobs, Ncomp))
            dat = scale(dat)
        else:
            raise Exception("wrong source distribution")

        # get modulation parameters
        modMat = np.random.uniform(0, 1, (Ncomp, Nsegment))

        # now we adjust the variance within each segment in a non-stationary manner
        for seg in range(Nsegment):
            segID = range(NsegmentObs * seg, NsegmentObs * (seg + 1))
            dat[segID, :] = np.multiply(dat[segID, :], modMat[:, seg])
            labels[segID] = seg

        # now we are ready to apply the non-linear mixtures:
        mixedDat = np.copy(dat)

        # generate mixing matrices:
        # will generate random uniform matrices and check their condition number based on following simulations:
        condList = []
        for i in range(int(Niter4condThresh)):
            # A = np.random.uniform(0,1, (Ncomp, Ncomp))
            A = np.random.uniform(1, 2, (Ncomp, Ncomp))  # - 1
            for i in range(Ncomp):
                A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
            condList.append(np.linalg.cond(A))

        condThresh = np.percentile(condList, 15)  # only accept those below 25% percentile

        # now we apply layers of non-linearity (just one for now!). Note the order will depend on natural of nonlinearity!
        # (either additive or more general!)
        mixingList = []
        for l in range(Nlayer - 1):
            # generate causal matrix first:
            A = generateUniformMat(Ncomp, condThresh)
            mixingList.append(A)

            # we first apply non-linear function, then causal matrix!
            if NonLin == 'leaky':
                mixedDat = leaky_ReLU(mixedDat, negSlope)
            elif NonLin == 'sigmoid':
                mixedDat = sigmoidAct(mixedDat)
            # apply mixing:
            mixedDat = np.dot(mixedDat, A)

        return {'source': dat, 'obs': mixedDat, 'labels': labels, 'mixing': mixingList, 'var': modMat}


    def __getitem__(self, index):
        result = {
            'observation': self.observation_data[index, :],
            'source': self.source_data[index, :],
            'label': self.labels[index]
        }
        return result

    def __len__(self):
        return len(self.labels)
