import numpy as np
from scipy.stats import truncnorm

def truncated_gaussian(m_det, loc, scale, upper_lim):
    """Compare a measurement to the model, knowing it was detected.
  
    A detection carries more information than its magnitude alone: it
    also tells us the source was brighter than the detection limit. The
    Gaussian is therefore truncated at that limit, so that the
    probability is normalised over the magnitudes that could have been
    seen at all.
  
    Parameters
    ----------
    m_det: numpy.ndarray
        Measured magnitudes.
        loc: numpy.ndarray
        Magnitudes the model predicts.
    scale: numpy.ndarray
        Total uncertainty on each measurement.
    upper_lim: float
        Detection limit, used as the truncation bound.
  
    Returns
    -------
    numpy.ndarray
        Log-probability of each measurement.
    """

    a = -np.inf  # no lower bound of truncation
    b = (upper_lim - loc) / scale  # upper bound in number of std-deviations
    return truncnorm.logpdf(m_det, a, b, loc=loc, scale=scale)

# Maps --em-error-model value to detection scoring function
ERROR_MODELS = {
    "truncated_gaussian": truncated_gaussian
}
