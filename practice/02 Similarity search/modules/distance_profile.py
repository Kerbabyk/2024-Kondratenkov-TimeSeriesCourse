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
    
    for i in range(n - m + 1):
        subsequence = ts[i:i+m]
        if is_normalize:
            subsequence = (subsequence - np.mean(subsequence)) / np.std(subsequence)
            query = (query - np.mean(query)) / np.std(query)
        distance_profile[i] = np.linalg.norm(subsequence - query)
    
    return distance_profile

distance_profile = brute_force(ts, query, is_normalize=True)
