"""Copied from https://github.com/ilkhem/icebeem/blob/3a7b1bfe7b62fdcdd753862773dd5e607e3fe4f9/metrics/mcc.py#L338"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


def auction_linear_assignment(x, eps=None, reduce='sum'):
    """
    Solve the linear sum assignment problem using the auction algorithm.
    Implementation in pytorch, GPU compatible.
    x_ij is the affinity between row (person) i and column (object) j, the
    algorithm aims to assign to each row i a column j_i such that the total benefit
    \sum_i x_{ij_i} is maximized.
    pytorch implementation, supports GPU.
    Algorithm adapted from http://web.mit.edu/dimitrib/www/Auction_Survey.pdf
    :param x: torch.Tensor
            The affinity (or benefit) matrix of size (n, n)
    :param eps: float, optional
            Bid size. Smaller values yield higher accuracy at the price of
            longer runtime.
    :param reduce: str, optional
            The reduction method to be applied to the score.
            If `sum`, sum the entries of cost matrix after assignment.
            If `mean`, compute the mean of the cost matrix after assignment.
            If `none`, return the vector (n,) of assigned column entry per row.
    :return: (torch.Tensor, torch.Tensor, int)
            Tuple of (score after application of reduction method, assignment,
            number of steps in the auction algorithm).
    """
    eps = 1 / x.size(0) if eps is None else eps

    price = torch.zeros((1, x.size(1))).to(x.device)
    assignment = torch.zeros(x.size(0)).long().to(x.device) - 1
    bids = torch.zeros_like(x).to(x.device)

    n_iter = 0
    while (assignment == -1).any():
        n_iter += 1

        # -- Bidding --
        # set I of unassigned rows (persons)
        # a person is unassigned if it is assigned to -1
        I = (assignment == -1).nonzero().squeeze(dim=1)
        # value matrix = affinity - price
        value_I = x[I, :] - price
        # find j_i, the best value v_i and second best value w_i for each i \in I
        top_value, top_idx = value_I.topk(2, dim=1)
        jI = top_idx[:, 0]
        vI, wI = top_value[:, 0], top_value[:, 1]
        # compute bid increments \gamma
        gamma_I = vI - wI + eps
        # fill entry (i, j_i) with \gamma_i for each i \in I
        # every unassigned row i makes a bid at one j_i with value \gamma_i
        bids_ = bids[I, :]
        bids_.zero_()
        bids_.scatter_(dim=1, index=jI.contiguous().view(-1, 1), src=gamma_I.view(-1, 1))

        # -- Assignment --
        # set J of columns (objects) that have at least a bidder
        # if a column j in bids_ is empty, then no bid was made to object j
        J = (bids_ > 0).sum(dim=0).nonzero().squeeze(dim=1)
        # determine the highest bidder i_j and corresponding highest bid \gamma_{i_j}
        # for each object j \in J
        gamma_iJ, iJ = bids_[:, J].max(dim=0)
        # since iJ is the index of highest bidder in the "smaller" array bids_,
        # find its actual index among the unassigned rows I
        # now iJ is a subset of I
        iJ = I[iJ]
        # raise the price of column j by \gamma_{i_j} for each j \in J
        price[:, J] += gamma_iJ
        # unassign any row that was assigned to object j at the beginning of the iteration
        # for each j \in J
        mask = (assignment.view(-1, 1) == J.view(1, -1)).sum(dim=1).byte()
        assignment.masked_fill_(mask, -1)
        # assign j to i_j for each j \in J
        assignment[iJ] = J

    score = x.gather(dim=1, index=assignment.view(-1, 1)).squeeze()
    if reduce == 'sum':
        score = torch.sum(score)
    elif reduce == 'mean':
        score = torch.mean(score)
    elif reduce == 'none':
        pass
    else:
        raise ValueError('not a valid reduction method: {}'.format(reduce))

    return score, assignment, n_iter


def rankdata_pt(b, tie_method='ordinal', dim=0):
    """
    pytorch equivalent of scipy.stats.rankdata, GPU compatible.
    :param b: torch.Tensor
            The 1-D or 2-D tensor of values to be ranked. The tensor is first flattened
            if tie_method is not 'ordinal'.
    :param tie_method: str, optional
            The method used to assign ranks to tied elements.
                The options are 'average', 'min', 'max', 'dense' and 'ordinal'.
                'average':
                    The average of the ranks that would have been assigned to
                    all the tied values is assigned to each value.
                    Supports 1-D tensors only.
                'min':
                    The minimum of the ranks that would have been assigned to all
                    the tied values is assigned to each value.  (This is also
                    referred to as "competition" ranking.)
                    Supports 1-D tensors only.
                'max':
                    The maximum of the ranks that would have been assigned to all
                    the tied values is assigned to each value.
                    Supports 1-D tensors only.
                'dense':
                    Like 'min', but the rank of the next highest element is assigned
                    the rank immediately after those assigned to the tied elements.
                    Supports 1-D tensors only.
                'ordinal':
                    All values are given a distinct rank, corresponding to the order
                    that the values occur in `a`.
                The default is 'ordinal' to match argsort.
    :param dim: int, optional
            The axis of the observation in the data if the input is 2-D.
            The default is 0.
    :return: torch.Tensor
            An array of length equal to the size of `b`, containing rank scores.
    """
    # b = torch.flatten(b)

    if b.dim() > 2:
        raise ValueError('input has more than 2 dimensions')
    if b.dim() < 1:
        raise ValueError('input has less than 1 dimension')

    order = torch.argsort(b, dim=dim)

    if tie_method == 'ordinal':
        ranks = order + 1
    else:
        if b.dim() != 1:
            raise NotImplementedError('tie_method {} not supported for 2-D tensors'.format(tie_method))
        else:
            n = b.size(0)
            ranks = torch.empty(n).to(b.device)

            dupcount = 0
            total_tie_count = 0
            for i in range(n):
                inext = i + 1
                if i == n - 1 or b[order[i]] != b[order[inext]]:
                    if tie_method == 'average':
                        tie_rank = inext - 0.5 * dupcount
                    elif tie_method == 'min':
                        tie_rank = inext - dupcount
                    elif tie_method == 'max':
                        tie_rank = inext
                    elif tie_method == 'dense':
                        tie_rank = inext - dupcount - total_tie_count
                        total_tie_count += dupcount
                    else:
                        raise ValueError('not a valid tie_method: {}'.format(tie_method))
                    for j in range(i - dupcount, inext):
                        ranks[order[j]] = tie_rank
                    dupcount = 0
                else:
                    dupcount += 1
    return ranks


def cov_pt(x, y=None, rowvar=False):
    """
    Estimate a covariance matrix given data in pytorch, GPU compatible.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    :param x: torch.Tensor
            A 1-D or 2-D array containing multiple variables and observations.
            Each column of `x` represents a variable, and each row a single
            observation of all those variables.
    :param y: torch.Tensor, optional
            An additional set of variables and observations. `y` has the same form
            as that of `x`.
    :param rowvar: bool, optional
            If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
            The default is False.
    :return: torch.Tensor
            The covariance matrix of the variables.
    """
    if y is not None:
        if not x.size() == y.size():
            raise ValueError('x and y have different shapes')
    if x.dim() > 2:
        raise ValueError('x has more than 2 dimensions')
    if x.dim() < 2:
        x = x.view(1, -1)
    if not rowvar and x.size(0) != 1:
        x = x.t()
    if y is not None:
        if y.dim() < 2:
            y = y.view(1, -1)
        if not rowvar and y.size(0) != 1:
            y = y.t()
        x = torch.cat((x, y), dim=0)

    fact = 1.0 / (x.size(1) - 1)
    x -= torch.mean(x, dim=1, keepdim=True)
    xt = x.t()  # if complex: xt = x.t().conj()
    return fact * x.matmul(xt).squeeze()


def corrcoef_pt(x, y=None, rowvar=False):
    """
    Return Pearson product-moment correlation coefficients in pytorch, GPU compatible.
    Implementation very similar to numpy.corrcoef using cov.
    :param x: torch.Tensor
            A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
    :param y: torch.Tensor, optional
            An additional set of variables and observations. `y` has the same form
            as that of `m`.
    :param rowvar: bool, optional
            If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
            The default is False.
    :return: torch.Tensor
            The correlation coefficient matrix of the variables.
    """
    c = cov_pt(x, y, rowvar)
    try:
        d = torch.diag(c)
    except RuntimeError:
        # scalar covariance
        return c / c
    stddev = torch.sqrt(d)
    c /= stddev[:, None]
    c /= stddev[None, :]

    return c


def spearmanr_pt(x, y=None, rowvar=False):
    """
    Calculates a Spearman rank-order correlation coefficient in pytorch, GPU compatible.
    :param x: torch.Tensor
            A 1-D or 2-D array containing multiple variables and observations.
            Each column of `x` represents a variable, and each row a single
            observation of all those variables.
    :param y: torch.Tensor, optional
            An additional set of variables and observations. `y` has the same form
            as that of `x`.
    :param rowvar: bool, optional
            If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
            The default is False.
    :return: torch.Tensor
           Spearman correlation matrix or correlation coefficient.
    """
    xr = rankdata_pt(x, dim=int(rowvar)).float()
    yr = None
    if y is not None:
        yr = rankdata_pt(y, dim=int(rowvar)).float()
    rs = corrcoef_pt(xr, yr, rowvar)
    return rs


def mean_corr_coef_pt(x, y, method='pearson'):
    """
    A differentiable pytorch implementation of the mean correlation coefficient metric.
    :param x: torch.Tensor
    :param y: torch.Tensor
    :param method: str, optional
            The method used to compute the correlation coefficients.
                The options are 'pearson' and 'spearman'
                'pearson':
                    use Pearson's correlation coefficient
                'spearman':
                    use Spearman's nonparametric rank correlation coefficient
    :return: float
    """
    d = x.size(1)
    if method == 'pearson':
        cc = corrcoef_pt(x, y)[:d, d:]
    elif method == 'spearman':
        cc = spearmanr_pt(x, y)[:d, d:]
    else:
        raise ValueError('not a valid method: {}'.format(method))
    cc = torch.abs(cc)
    score, _, _ = auction_linear_assignment(cc, reduce='mean')
    return score


def mean_corr_coef_np(x, y, method='pearson'):
    """
    A numpy implementation of the mean correlation coefficient metric.
    :param x: numpy.ndarray
    :param y: numpy.ndarray
    :param method: str, optional
            The method used to compute the correlation coefficients.
                The options are 'pearson' and 'spearman'
                'pearson':
                    use Pearson's correlation coefficient
                'spearman':
                    use Spearman's nonparametric rank correlation coefficient
    :return: float
    """
    d = x.shape[1]
    if method == 'pearson':
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
    elif method == 'spearman':
        cc = spearmanr(x, y)[0][:d, d:]
    else:
        raise ValueError('not a valid method: {}'.format(method))
    cc = np.abs(cc)
    score = cc[linear_sum_assignment(-1 * cc)].mean()
    return score


def mean_corr_coef(x, y, method='pearson'):
    if type(x) != type(y):
        raise ValueError('inputs are of different types: ({}, {})'.format(type(x), type(y)))
    if isinstance(x, np.ndarray):
        return mean_corr_coef_np(x, y, method)
    elif isinstance(x, torch.Tensor):
        return mean_corr_coef_pt(x, y, method)
    else:
        raise ValueError('not a supported input type: {}'.format(type(x)))


def mean_corr_coef_out_of_sample(x, y, x_test, y_test, method='pearson'):
    """
    we compare mean correlation coefficients out of sample
    -> we use (x,y) to learn permutation and then evaluate the correlations
    determined by this permutation on (x_test, y_test)
    """

    d = x.shape[1]
    if method == 'pearson':
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
        cc_test = np.corrcoef(x_test, y_test, rowvar=False)[:d, d:]
    elif method == 'spearman':
        cc = spearmanr(x, y)[0][:d, d:]
        cc_test = spearmanr(x_test, y_test)[0][:d, d:]
    else:
        raise ValueError('not a valid method: {}'.format(method))
    cc = np.abs(cc)

    score = np.abs(cc_test)[linear_sum_assignment(-1 * cc)].mean()
    return score
