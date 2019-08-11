import numpy as np
import autograd.numpy as agnp
from scipy.stats import norm
import scipy.special as special


def gaussian(x, mean, sigma):
    sqrt_2pi = np.sqrt(np.multiply(2, np.pi))
    constant = np.divide(1, np.multiply(sigma, sqrt_2pi))
    exponent = -((x - mean)**2.) / (2 * sigma**2.)
    return np.multiply(constant, agnp.exp(exponent))

def gaussian_with_range(universe, mean, normalize=False):
    """
    This function should always return a gaussian with the range of the initial universe
    However, it may not be centered at the center of that range.
    :param universe: np array that has the universe of points we are analysing in our fuzzy
    :param mean: the "mean" that you want to center your normal curve at
    :param normalize: whether to normalize range to 1
    :return: An antecedent range, the sigma of the gaussian
    """
    total_points = np.size(universe)
    total_range = np.max(universe) - np.min(universe)
    # sigma calculations are 6 sigma from the mean
    sigma = np.divide(total_range, 12)
    revised_universe_range = np.arange(mean - 6 * sigma,
                                       mean + 6 * sigma,
                                       np.divide(12*sigma, total_points))
    if normalize:
        revised_universe_range = np.divide(np.subtract(revised_universe_range, np.min(revised_universe_range)),
                                           np.subtract(np.max(revised_universe_range), np.min(revised_universe_range)))
    return revised_universe_range, sigma