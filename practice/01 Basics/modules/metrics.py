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
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2
    """
    
    if len(ts1) != len(ts2):
        raise ValueError("Временные ряды должны иметь одинаковую длину")
    
    n = len(ts1)
    
    # Вычисляем средние значения и стандартные отклонения
    mu_ts1 = np.mean(ts1)
    mu_ts2 = np.mean(ts2)
    sigma_ts1 = np.std(ts1)
    sigma_ts2 = np.std(ts2)
    
    # Вычисляем скалярное произведение
    dot_product = np.dot(ts1, ts2)
    
    # Вычисляем нормализованное евклидово расстояние
    norm_ed_dist = np.sqrt(2 * n * (1 - (dot_product - n * mu_ts1 * mu_ts2) / (n * sigma_ts1 * sigma_ts2)))
    
    return norm_ed_dist


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))
    

def DTW_distance(T1, T2):
    """
    Вычисляет DTW расстояние между двумя временными рядами T1 и T2.
    """
    n = len(T1)
    m = len(T2)
    
    # Инициализация матрицы расстояний
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Заполнение матрицы расстояний
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean_distance(T1[i-1], T2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    
    # Возвращаем DTW расстояние
    return np.sqrt(dtw_matrix[n, m])

    # INSERT YOUR CODE

    #return dtw_dist
