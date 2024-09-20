import numpy as np

def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    
    if len(ts1) != len(ts2):
        raise ValueError("Временные ряды должны иметь одинаковую длину")
    
    # Преобразуем входные данные в numpy массивы для удобства вычислений
    ts1 = np.array(ts1)
    ts2 = np.array(ts2)
    
    # Вычисляем евклидово расстояние
    distance = np.sqrt(np.sum((ts1 - ts2) ** 2))
    
    return distance


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    norm_ed_dist = 0

    # INSERT YOUR CODE

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    dtw_dist = 0

    # INSERT YOUR CODE

    return dtw_dist
