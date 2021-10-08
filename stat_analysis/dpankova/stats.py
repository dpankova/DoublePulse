from __future__ import division, print_function
import numpy as np
from scipy.stats import norm

def ts_to_pval(ts, test_value):
    sorted_ts = np.sort(ts)
    idx = np.searchsorted(sorted_ts, test_value)
    pval = 1 - (idx / len(sorted_ts))
    return pval

def prob2sigma(p):
    p = np.atleast_1d(p)
    return norm.ppf(p + (1. - p) / 2.)

def sample_from_hist(hist, n_samples, random_state):
    '''Sample `n_samples` from a histogram and return them binned as a
    histogram in the same binning.
    Parameters
    ----------
    hist : ndarray
        Histogram to resample
    n_samples: int
        Number of samples to draw from the histogram
    Returns
    -------
    sampled_hist : ndarray
        Returns a histogram with the same shape of `hist` with `n_samples`
        entries.
    '''
    random_state = check_random_state(random_state)

    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]

    values = random_state.rand(n_samples)
    value_bins = np.searchsorted(cdf, values)

    sampled_hist = np.bincount(value_bins,
                               minlength=np.prod(hist.shape))
    sampled_hist = sampled_hist.reshape(hist.shape)
    return sampled_hist

def check_random_state(random_state):
    if isinstance(random_state, np.random.RandomState):
        return random_state
    else:
        if not isinstance(random_state, int):
            raise ValueError('random_state has to be either an int or of ' +
                             'type np.random.RandomState!')
        else:
            random_state = np.random.RandomState(random_state)
            return random_state