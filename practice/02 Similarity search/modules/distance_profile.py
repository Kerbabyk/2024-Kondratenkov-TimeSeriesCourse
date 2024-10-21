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

def topK_match(dist_profile: np.ndarray, excl_zone: int, topK: int) -> dict:
    """
    Find the top K indices with the smallest distances in the distance profile,
    considering the exclusion zone.

    Parameters
    ----------
    dist_profile: np.ndarray
        Distance profile.
    excl_zone: int
        Exclusion zone size.
    topK: int
        Number of top indices to return.

    Returns
    -------
    dict
        Dictionary containing 'indices' and 'distances' of the top K matches.
    """
    n = len(dist_profile)
    topK_indices = []
    topK_distances = []

    for _ in range(topK):
        min_index = np.argmin(dist_profile)
        min_distance = dist_profile[min_index]
        topK_indices.append(min_index)
        topK_distances.append(min_distance)

        # Apply exclusion zone
        start_excl = max(0, min_index - excl_zone)
        end_excl = min(n, min_index + excl_zone + 1)
        dist_profile[start_excl:end_excl] = np.inf

    return {'indices': topK_indices, 'distances': topK_distances}

# Пример использования
if __name__ == "__main__":
    # Чтение данных
    ts = read_ts('path_to_ts_file').values.flatten()
    query = read_ts('path_to_query_file').values.flatten()

    # Параметры
    excl_zone_frac = 0.5
    m = len(query)
    excl_zone = int(np.ceil(excl_zone_frac * m))

    # Вычисление профиля расстояния
    distance_profile = brute_force(ts, query)

    # Поиск topK похожих подпоследовательностей
    topK_results = topK_match(distance_profile, excl_zone=excl_zone, topK=2)

    print("Top K indices:", topK_results['indices'])
    print("Top K distances:", topK_results['distances'])
