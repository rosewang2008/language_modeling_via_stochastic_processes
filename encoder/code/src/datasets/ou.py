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

def l2normalize(Amat, axis=0):
    # axis: 0=column-normalization, 1=row-normalization
    l2norm = np.sqrt(np.sum(Amat*Amat,axis))
    Amat = Amat / l2norm
    return Amat


class OUData(data.Dataset):
    """
    Data generated from the Ornstein-Ohlenbeck process
    """

    def __init__(self,
                 train, data_dim, n_samples, seed, one_hot_labels,
                 samples_per_seq, dt, mu, sigma,
                 x_range=None):
        super().__init__()
        self.train = train
        self.n_samples = n_samples
        self.seed = seed
        self.data_dim = data_dim
        self.one_hot_labels = one_hot_labels
        self.samples_per_seq = samples_per_seq
        self.dt = dt
        self.mu = mu
        self.sigma = sigma
        if x_range is None:
            self.x_range = [-5, 5]
        else:
            self.x_range = x_range
        np.random.seed(self.seed)

        self._generate_synthetic_data()

    def _generate_unnormalized_data(self):
        # Generate X's
        self._xs = []
        raise ValueError()

        for dim in range(int(self.data_dim)): # number of dimensions; rn: 1D, 2D
            _x_dim = []
            for sample_idx in range(int(self.n_samples)):
                if sample_idx % self.samples_per_seq == 0: # doing n_samples//samples_per_seq sequences
                    # Start a new mean
                    x_t = np.random.randint(-10, 10)
                    _x_dim.append(x_t)
                else:
                    # First order approximation
                    x_tp1 = (x_t - x_t * self.dt
                             + np.sqrt(self.dt)*self.sigma*np.random.normal(self.mu, 1.0, 1))[0]
                    _x_dim.append(x_tp1)
                    x_t = x_tp1
            self._xs.append(_x_dim)

        assert len(self._xs) == int(self.data_dim)

    def _generate_synthetic_data(self):
        self._generate_unnormalized_data()
        self.xs = np.array(self._xs)
        self.ys = np.array(self._ys)
        # # # Normalize X
        # self.ys = [(x - np.mean(x))/(np.max(x) - np.min(x)) for x in self._xs]
        # # self.ys = self._xs
        # self.ys = np.array(self.ys)

    def __getitem__(self, index):
        # Checking for new sequence starting at index + 1
        if (index + 1) % (self.samples_per_seq) == 0:
            # NOTE index should be multiples of 999, because at 1000 is start of new seq
            index -= 1

        step = (index + 1) % self.samples_per_seq
        t = step / self.samples_per_seq

        result = {
            'y_t': self.ys[index, :],
            'y_tp1': self.ys[index+1, :],
            'x_t': self.xs[index, :],
            'x_tp1': self.xs[index+1, :],
            't': t,
        }
        return result

    def __len__(self):
        return self.n_samples - 1


class OUData_LT(OUData):

    def __init__(self,
                 train, data_dim, n_samples, seed, one_hot_labels,
                 samples_per_seq, dt, mu, sigma,
                 A=None,
                 A_range=None,
                 iter4condthresh=1000,
                 cond_thresh_ratio=0.25):

        # Parameters for initializing LT matrix
        if A_range is None:
            self.A_range = [-1, 1]
        else:
            self.A_range = A_range

        self.A = A

        self.iter4condthresh = iter4condthresh
        self.cond_thresh_ratio = cond_thresh_ratio

        super().__init__(
            train=train, data_dim=data_dim, n_samples=n_samples, seed=seed,
            one_hot_labels=one_hot_labels, samples_per_seq=samples_per_seq,
            dt=dt, mu=mu, sigma=sigma)

    def g(self, x):
        return np.dot(self.A, x)

    def _init_LT(self):
        """initialize mixing matrix A with reasonable conditioning number"""
        # Determining conditioning number
        if self.A is not None:
            condA = np.linalg.cond(self.A)
        else:
            condList = np.zeros([self.iter4condthresh])
            for i in range(self.iter4condthresh):
                A = np.random.uniform(self.A_range[0], self.A_range[1], [self.data_dim, self.data_dim])
                A = l2normalize(A, axis=0)
                condList[i] = np.linalg.cond(A)
            condList.sort() # Ascending order
            condThresh = condList[int(self.iter4condthresh * self.cond_thresh_ratio)]
            print("Conditioning thresh: {}".format(condThresh))

            # Generating mixing matrix
            condA = condThresh + 1
            while condA > condThresh:
                A = np.random.uniform(self.A_range[0], self.A_range[1], [self.data_dim, self.data_dim])
                A = l2normalize(A)  # Normalize (column)
                condA = np.linalg.cond(A)

            self.A = A

        print("A: {}".format(self.A))
        print("inv(A): {}".format(np.linalg.inv(self.A)))
        print("\kappa(A): {}".format(condA))


    def _generate_unnormalized_data(self):
        self._xs = []
        self._ys = []
        self._init_LT()

        for sample_idx in range(int(self.n_samples)):
            if sample_idx % self.samples_per_seq == 0: # doing n_samples//samples_per_seq sequences
                # Start a new x_0
                x_t = np.random.uniform(self.x_range[0], self.x_range[1], self.data_dim)
                y_t = self.g(x_t)
                self._xs.append(x_t)
                self._ys.append(y_t)
            else:
                # First order approximation
                noise = np.sqrt(self.dt)*self.sigma*np.random.normal(self.mu, 1.0, self.data_dim)
                # noise = 0.0
                x_tp1 = (x_t - x_t * self.dt + noise)
                y_tp1 = self.g(x_tp1)
                self._xs.append(x_tp1)
                self._ys.append(y_tp1)
                x_t = x_tp1

class OUBridgeData_LT(OUData_LT):
    def __init__(self,
                 train, data_dim, n_samples, seed, one_hot_labels,
                 samples_per_seq, dt, mu, sigma,
                 A=None,
                 A_range=None):

        self.B_0 = 0
        self.B_T = 1
        super().__init__(
            train=train, data_dim=data_dim, n_samples=n_samples, seed=seed,
            one_hot_labels=one_hot_labels, samples_per_seq=samples_per_seq,
            dt=dt, mu=mu, sigma=sigma, A=A, A_range=A_range)
        assert self.dt == 1./samples_per_seq

    def _generate_unnormalized_data(self):
        self._xs = []
        self._ys = []
        self._init_LT()

        for sample_idx in range(int(self.n_samples)):
            if sample_idx % self.samples_per_seq == 0: # doing n_samples//samples_per_seq sequences
                # Start a new x_0
                x_t = np.zeros(self.data_dim)
                y_t = self.g(x_t)
                self._xs.append(x_t)
                self._ys.append(y_t)
            elif (sample_idx + 1) % self.samples_per_seq == 0: # last idx
                x_t = np.ones(self.data_dim)
                y_t = self.g(x_t)
                self._xs.append(x_t)
                self._ys.append(y_t)
            else:
                # First order approximation
                noise = np.sqrt(self.dt)*self.sigma*np.random.normal(self.mu, 1.0, self.data_dim)
                # noise /= self.data_dim
                step = sample_idx % self.samples_per_seq
                t = step/self.samples_per_seq
                # noise = 0.0
                x_tp1 = x_t * (1- self.dt/(1. - t)) + (self.dt/(1.-t))*self.B_T + noise
                y_tp1 = self.g(x_tp1)
                self._xs.append(x_tp1)
                self._ys.append(y_tp1)
                x_t = x_tp1


class BrownianBridgeData_LT(OUBridgeData_LT):
    """For brownian bridge everything setup"""
    def __init__(self,
                 train, data_dim, n_samples, seed, one_hot_labels,
                 samples_per_seq, dt, mu, sigma,
                 A=None,
                 A_range=None):
        super().__init__(
            train=train, data_dim=data_dim, n_samples=n_samples, seed=seed,
            one_hot_labels=one_hot_labels, samples_per_seq=samples_per_seq,
            dt=dt, mu=mu, sigma=sigma, A=A, A_range=A_range)

    def __getitem__(self, index):
        # Check if index is start of a seq. If so -> +1
        if index % self.samples_per_seq == 0:
            old_idx = index
            index += 1
        # If index is end of seq
        elif index % (self.samples_per_seq ) == (self.samples_per_seq - 1):
            old_idx = index
            index -= 1

        t = index % self.samples_per_seq
        y_0 = self.ys[index - t]
        y_t = self.ys[index]
        y_T = self.ys[index + (self.samples_per_seq - 1 - t)]

        progress_t = t/self.samples_per_seq
        alpha = self.dt/(1-progress_t)
        var = alpha * (1-alpha) * (1-progress_t)

        result = {
            'y_0': y_0,
            'y_t': y_t,
            'y_T': y_T,
            't': t,
            'T': self.samples_per_seq,
            'total_t': self.samples_per_seq,
            'alpha': alpha,
            'var': var,
        }
        return result

class BrownianBridgeRandomT_LT(OUBridgeData_LT):
    """For brownian bridge everything setup"""
    def __init__(self,
                 train, data_dim, n_samples, seed, one_hot_labels,
                 samples_per_seq, dt, mu, sigma,
                 A=None,
                 A_range=None):
        super().__init__(
            train=train, data_dim=data_dim, n_samples=n_samples, seed=seed,
            one_hot_labels=one_hot_labels, samples_per_seq=samples_per_seq,
            dt=dt, mu=mu, sigma=sigma, A=A, A_range=A_range)

    def __getitem__(self, index):
        # Check if index is start of a seq. If so -> +2
        if index % self.samples_per_seq == 0:
            index += 2
        if index % self.samples_per_seq == 1:
            index += 1

        T = index % self.samples_per_seq
        # t is a random point in between
        t = np.random.randint(1, T)
        y_0 = self.ys[index - T]
        y_t = self.ys[index - T + t]
        y_T = self.ys[index]

        progress_t = t/T# self.samples_per_seq
        alpha = self.dt/(1-progress_t)
        var = alpha * (1-alpha) * (1-progress_t)
        try:
            assert t > 0 and t < self.samples_per_seq
        except:
            import pdb; pdb.set_trace()

        result = {
            'y_0': y_0,
            'y_t': y_t,
            'y_T': y_T,
            't_': 0,
            't': t,
            'T': T, # self.samples_per_seq,
            'total_t': self.samples_per_seq,
            'alpha': alpha,
            'var': var,
        }
        return result

class BrownianBridgeRandom_LT(OUBridgeData_LT):
    """For brownian bridge everything setup"""
    def __init__(self,
                 train, data_dim, n_samples, seed, one_hot_labels,
                 samples_per_seq, dt, mu, sigma,
                 A=None,
                 A_range=None):
        super().__init__(
            train=train, data_dim=data_dim, n_samples=n_samples, seed=seed,
            one_hot_labels=one_hot_labels, samples_per_seq=samples_per_seq,
            dt=dt, mu=mu, sigma=sigma, A=A, A_range=A_range)

    def __getitem__(self, index):
        # Check if index is start of a seq. If so -> +2
        if index % self.samples_per_seq == 0:
            index += 2
        if index % self.samples_per_seq == 1:
            index += 1

        T = index % self.samples_per_seq
        # t is a random point in between
        nums = list(range(T))
        t1 = random.choice(nums)
        nums.remove(t1)
        t2 = random.choice(nums)
        if t2 < t1:
            t = t2
            t2 = t1
            t1 = t

        assert t1 < t2 and t2 < T
        y_0 = self.ys[index - T + t1]
        y_t = self.ys[index - T + t2]
        y_T = self.ys[index]

        result = {
            'y_0': y_0,
            'y_t': y_t,
            'y_T': y_T,
            't_': t1,
            't': t2,
            'T': T, # self.samples_per_seq,
            'total_t': self.samples_per_seq,
            'alpha': 0,
            'var': 0,
        }
        return result

class OUData_Classification(OUData_LT):

    def __init__(self,
                 train, data_dim, n_samples, seed, one_hot_labels,
                 samples_per_seq, dt, mu, sigma,
                 A=None,
                 A_range=None):
        self.pos_p = 0.5 # probability of positive sample
        super().__init__(
            train=train, data_dim=data_dim, n_samples=n_samples, seed=seed,
            one_hot_labels=one_hot_labels, samples_per_seq=samples_per_seq,
            dt=dt, mu=mu, sigma=sigma, A=A, A_range=A_range)

    def __getitem__(self, index):
        index_t = index
        # Checking for new sequence starting at index + 1
        if (index + 1) % (self.samples_per_seq) == 0:
            # NOTE index should be multiples of 999, because at 1000 is start of new seq
            index_t -= 1

        coin_flip = np.random.binomial(1, self.pos_p)
        if coin_flip: # heads -> positive
            index_tp1 = index_t + 1
        else:
            traj_idx = index_t // self.samples_per_seq
            start_idx = traj_idx * self.samples_per_seq
            end_idx = (traj_idx+1) * self.samples_per_seq
            index_tp1 = np.random.randint(start_idx, end_idx)

        result = {
            'y_t': self.ys[index_t, :],
            'y_tp1': self.ys[index_tp1, :],
            'label': coin_flip,
            'x_t': self.xs[index_t, :],
            'x_tp1': self.xs[index_tp1, :],
        }
        return result

class Noisy_OUData_Classification(OUData_Classification):

    def __init__(self,
                 train, data_dim, n_samples, seed, one_hot_labels,
                 samples_per_seq, dt, mu, sigma, noisy_sigma,
                 A=None, A_range=None):
        self.noisy_sigma = noisy_sigma
        super().__init__(
            train=train, data_dim=data_dim, n_samples=n_samples, seed=seed,
            one_hot_labels=one_hot_labels, samples_per_seq=samples_per_seq,
            dt=dt, mu=mu, sigma=sigma, A=A, A_range=A_range)

    def g(self, x):
        noise = np.random.normal(0, self.noisy_sigma, self.data_dim)
        return np.dot(self.A, x) + noise

class Noisy_OUData_LT(OUData_LT):

    def __init__(self,
                 train, data_dim, n_samples, seed, one_hot_labels,
                 samples_per_seq, dt, mu, sigma, noisy_sigma,
                 seq_len,
                 A=None, A_range=None, iter4condthresh=1000,
                 cond_thresh_ratio=0.25):
        self.noisy_sigma = noisy_sigma
        self.seq_len = seq_len
        super().__init__(
            train=train, data_dim=data_dim, n_samples=n_samples, seed=seed,
            one_hot_labels=one_hot_labels, samples_per_seq=samples_per_seq,
            dt=dt, mu=mu, sigma=sigma, A=A, A_range=A_range, iter4condthresh=iter4condthresh,
            cond_thresh_ratio=cond_thresh_ratio
        )

    def g(self, x):
        noise = np.random.normal(0, self.noisy_sigma, self.data_dim)
        return np.dot(self.A, x) + noise

    def __getitem__(self, index):
        # Checking for new sequence starting at index + 1
        for offset in range(1, 1+self.seq_len):
            if (index + offset) % (self.samples_per_seq) == 0:
                # NOTE index should be multiples of 999, because at 1000 is start of new seq
                index -= offset

        result = {
            'y_t': self.ys[index:index+self.seq_len, :],
            'y_tp1': self.ys[index+1:index+1+self.seq_len, :],
            'x_t': self.xs[index:index+self.seq_len, :],
            'x_tp1': self.xs[index+1:index+1+self.seq_len, :],
        }
        return result

    def __len__(self):
        return self.n_samples - 1 - self.seq_len

class OUData_Bridge(OUData):

    def __init__(self,
                 train, data_dim, n_samples, seed, one_hot_labels,
                 samples_per_seq, dt, mu, sigma):
        self.W0 = 0
        self.WT = 0

        super().__init__(
            train=train, data_dim=data_dim, n_samples=n_samples, seed=seed,
            one_hot_labels=one_hot_labels, samples_per_seq=samples_per_seq,
            dt=dt, mu=mu, sigma=sigma)

    def _generate_unnormalized_data(self):
        self._xs = []
        self._ts = []
        raise ValueError()

        for dim in range(int(self.data_dim)): # number of dimensions; rn: 1D, 2D
            X = np.zeros((self.n_samples, self.samples_per_seq), dtype=np.float32)
            X[:, 0] = self.W0
            X[:, -1] = self.WT
            ts = [0.0]
            dt = 1./self.samples_per_seq
            std = np.sqrt(self.dt)
            for step in range(1, self.samples_per_seq - 1):
                t = step/self.samples_per_seq
                W_t = np.random.normal(0, 1, self.n_samples) * std
                X[:, step] = X[:, step-1] - dt*(X[:, step-1]/(1-t)) + W_t
                ts.append(t)
            ts.append(1.)

            _x_dim = (X.flatten()-np.mean(X))/(2) # 2 = 1-(-1)
            self._xs.append(_x_dim)
            self._ts.append(ts)

        assert len(self._xs) == int(self.data_dim)

        dt = 1./self.samples_per_seq
        times = np.arange(0.0, self.samples_per_seq)/self.samples_per_seq
        times = 1 - dt/(1-times)
        self.ts = np.zeros((self.n_samples, self.samples_per_seq))
        self.ts[:] = times
        self.ts = self.ts.flatten()

    def __getitem__(self, index):
        result = {
            'y_t': self.ys[:, index],
            'y_tp1': self.ys[:, index+1],
            't': self.ts[index]
        }
        return result
