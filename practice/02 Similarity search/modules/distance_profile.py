import numpy as np

def brute_force(ts: np.ndarray, query: np.ndarray, is_normalize: bool = True) -> np.ndarray:
    """
    Calculate the distance profile using the brute force algorithm.

    Parameters
    ----------
    ts: np.ndarray
        Time series.
    query: np.ndarray
        Query, shorter than time series.
    is_normalize: bool, optional
        Normalize or not time series and query (default is True).

    Returns
    -------
    dist_profile: np.ndarray
        Distance profile between query and time series.
    """
    # Lengths of time series and query
    n = len(ts)
    m = len(query)

    # Normalize time series and query if is_normalize is True
    if is_normalize:
        ts = (ts - np.mean(ts)) / np.std(ts)
        query = (query - np.mean(query)) / np.std(query)

    # Initialize distance profile
    dist_profile = np.zeros(n - m + 1)

    # Calculate distance profile using brute force
    for i in range(n - m + 1):
        subseq = ts[i:i + m]
        # Calculate Euclidean distance between query and subsequence
        dist = np.linalg.norm(query - subseq)
        dist_profile[i] = dist

    return dist_profile
