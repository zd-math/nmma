import numpy as np
from scipy.stats import truncnorm

def truncated_gaussian(m_det, loc, scale, upper_lim):
    a = -np.inf  # no lower bound of truncation
    b = (upper_lim - loc) / scale  # upper bound in number of std-deviations
    return truncnorm.logpdf(m_det, a, b, loc=loc, scale=scale)

ERROR_MODELS = {
    "truncated_gaussian": truncated_gaussian
}
