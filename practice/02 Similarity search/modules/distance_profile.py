import numpy as np

from modules.utils import z_normalize
from modules.metrics import ED_distance, norm_ED_distance


def brute_force(ts: np.ndarray, query: np.ndarray, is_normalize: bool = True) -> np.ndarray:
    """
    Calculate the distance profile using the brute force algorithm

    Parameters
    ----------
    ts: time series
    query: query, shorter than time series
    is_normalize: normalize or not time series and query

    Returns
    -------
    dist_profile: distance profile between query and time series
    """

    n = len(ts)
    m = len(query)
    N = n-m+1
    distance_profile = np.zeros(n - m + 1)
    
    if is_normalize:
        ts = (ts - np.mean(ts)) / np.std(ts)
        query = (query - np.mean(query)) / np.std(query)
    
    for i in range(n - m + 1):
        distance_profile[i] = euclidean(ts[i:i+m], query)
    
    return distance_profile
