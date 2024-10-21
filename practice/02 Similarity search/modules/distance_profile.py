import numpy as np
from modules.utils import z_normalize
from modules.metrics import ED_distance, norm_ED_distance

def brute_force(ts: np.ndarray, query: np.ndarray, is_normalize: bool = True) -> np.ndarray:
    """
    Calculate the distance profile using the brute force algorithm.

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
    N = n - m + 1
    distance_profile = np.zeros(N)
    
    if is_normalize:
        query_normalized = z_normalize(query)
    else:
        query_normalized = query
    
    for i in range(N):
        if is_normalize:
            subsequence_normalized = z_normalize(ts[i:i+m])
        else:
            subsequence_normalized = ts[i:i+m]
        
        distance_profile[i] = norm_ED_distance(subsequence_normalized, query_normalized)
        
        if distance_profile[i] == 0.0:
            print(f"Zero distance at index {i}:")
            print("Subsequence:", subsequence_normalized)
            print("Query:", query_normalized)
    
    return distance_profile
