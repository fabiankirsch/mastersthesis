import numpy as np
from sklearn.preprocessing import StandardScaler

def fit_standard_scaler_to_sequence_batch(X):
    """
    X: 3d array containing sequences

    Returns 2 arrays of means and variances
    """

    X_2d = X.reshape(-1, X.shape[2])

    scaler = StandardScaler().fit(X_2d)

    return scaler.mean_, scaler.var_

def normalize_sequence_batch(X, means, variances):
    """
    X: multivariate sequence batch (numpy 3d array)
    means: array of means, one for each column. Used to for normalization
    variances: array of variances, one for each column. Used for normalization

    Applying normalization function on third axis of array (index 2).

    Returns normalized sequence batch
    """

    return np.apply_along_axis(normalize_single_observation, 2, X, means, variances)

def normalize_single_observation(X, means, variances):
    return (X - means) / np.sqrt(variances)

def normalize_sequence(X, means, variances):
    """
    X: single multivariate sequence (2d numpy array)
    means: array of means, one for each column. Used to for normalization
    variances: array of variances, one for each column. Used for normalization

    Returns normalized sequence.
    """
    return (X - means) / np.sqrt(variances)
