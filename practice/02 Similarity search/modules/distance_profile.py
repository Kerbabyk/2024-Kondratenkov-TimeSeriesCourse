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
    n = len(ts)
    m = len(query)

    if is_normalize:
        query = (query - np.mean(query)) / np.std(query)

    dist_profile = np.zeros(n - m + 1)

    for i in range(n - m + 1):
        subseq = ts[i:i + m]
        if is_normalize:
            subseq = (subseq - np.mean(subseq)) / np.std(subseq)
        dist = np.linalg.norm(query - subseq)
        dist_profile[i] = dist

    return dist_profile
