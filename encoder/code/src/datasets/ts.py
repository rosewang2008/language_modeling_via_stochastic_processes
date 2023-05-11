import os
import sys
import json
import torch
import random
import string
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.utils.data as data
from sklearn.preprocessing import scale

def l2normalize(Amat, axis=0):
    # axis: 0=column-normalization, 1=row-normalization
    l2norm = np.sqrt(np.sum(Amat*Amat,axis))
    Amat = Amat / l2norm
    return Amat


class TimeSeriesData(data.Dataset):
    """
    M4 competition data
    """

    def __init__(self, train, seed, data_dim, sequence_length, one_hot_labels, filepath):
        super().__init__()
        self.train = train
        self.seed = seed
        self.data_dim = data_dim
        self.sequence_length = sequence_length
        self.one_hot_labels = one_hot_labels
        self.filepath = filepath
        np.random.seed(self.seed)

        self.data = pd.read_csv(self.filepath, delimiter=',')
        self.data = self.data.iloc[:, :self.data_dim+1]
        self._normalize_data()

    def _z_score(self):
        # compute z score for normalization
        for idx in range(1, self.data.shape[1]):
            mu = self.data.iloc[:, idx].mean()
            std = self.data.iloc[:, idx].std()
            self.data.iloc[:, idx] = self.data.iloc[:, idx].apply(lambda x: (x-mu)/std)

    def _log(self):
        # taking the log of data
        for idx in range(1, self.data.shape[1]):
            self.data.iloc[:, idx] = self.data.iloc[:, idx].apply(lambda x: np.log(x))

    def _normalize_data(self):
        self._z_score()

        # self._log()


    def __getitem__(self, index):
        result = {
            'y_t': self.data.iloc[
                index:index+self.sequence_length, 1:].to_numpy().astype(float), # 0th column is day ID
            'y_tp1': self.data.iloc[
                index+1:index+1+self.sequence_length, 1:].to_numpy().astype(float),
        }
        return result

    def __len__(self):
        return self.data.shape[0] - 1 - self.sequence_length


class StocksData(data.Dataset):
    """
    Stocks from Yahoo
    """

    def __init__(self, train, seed, data_dim, sequence_length, one_hot_labels, directory):
        super().__init__()
        self.train = train
        self.seed = seed
        self.data_dim = data_dim
        self.sequence_length = sequence_length
        self.one_hot_labels = one_hot_labels
        self.directory = directory
        np.random.seed(self.seed)

        self._data = []
        for fname in os.listdir(directory):
            if fname.endswith(".csv") and len(self._data) != self.data_dim:
                print("loading {}".format(fname))
                fpath = os.path.join(directory, fname)
                df = pd.read_csv(fpath, delimiter=',')
                self._data.append(df)
                # all data shape is (14871, 8)

        self._normalize_data()
        self._shift_end_to_zero()

    def _shift_end_to_zero(self):
        offset = np.tile(0 - self._data[:, -1], (self._data.shape[-1], 1)).T
        self._data = offset + self._data

    def _log(self):
        """apply log onto data"""
        for df in self._data:
            df['log_close'] =np.log(df['Close'])
        self._data = np.concatenate([[df['log_close']] for df in self._data])

    def _z_score(self):
        """apply z score onto data"""
        for df in self._data:
            mu = df['Close'].mean()
            std = df['Close'].std()
            df['z_close'] = (df['Close']-mu)/std
        self._data = np.concatenate([[df['z_close']] for df in self._data])

    def _normalize_data(self):
        self._z_score()
        # self._log()

    def __getitem__(self, index):

        t = index / self._data.shape[-1]

        result = {
            'y_t': self._data[:, index:index+self.sequence_length].T,
            'y_tp1': self._data[:, index+1:index+1+self.sequence_length].T,
            't': t
        }
        return result

    def __len__(self):
        return self._data.shape[1] - 1 - self.sequence_length


class IntervalStocksData(StocksData):
    """
    Stocks from Yahoo
    """

    def __init__(self, train, seed, data_dim, sequence_length,
                 one_hot_labels, directory, sampling_interval):
        super().__init__(
            train=train,
            seed=seed,
            data_dim=data_dim,
            sequence_length=sequence_length,
            one_hot_labels=one_hot_labels,
            directory=directory,)
        self.sampling_interval = sampling_interval

        # Splice the data correctly during loading
        # self._data: (data_dim, 14871)
        num_obs = self._data.shape[1] // self.sampling_interval
        idxs = np.arange(num_obs) * self.sampling_interval
        self._data = self._data[:, idxs]

    def __getitem__(self, index):
        y_t_idxs = np.array([index + _ for _ in range(self.sequence_length)])
        y_tp1_idxs = y_t_idxs + 1

        result = {
            'y_t': self._data[:, y_t_idxs].T,
            'y_tp1': self._data[:, y_tp1_idxs].T,
        }
        return result

    def __len__(self):
        return self._data.shape[1] - self.sequence_length - 1
