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


def euclidean_distance(x, y):
    """
    Вычисляет евклидово расстояние между двумя точками.
    
    Параметры:
    x (float): Первая точка.
    y (float): Вторая точка.
    
    Возвращает:
    float: Евклидово расстояние между x и y.
    """
    return np.sqrt((x - y) ** 2)

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
    n = len(ts1)
    m = len(ts2)
    
    # Создаем матрицу для хранения DTW расстояний
    dtw_matrix = np.zeros((n + 1, m + 1))
    
    # Инициализируем первую строку и первый столбец матрицы бесконечностями
    for i in range(1, n + 1):
        dtw_matrix[i, 0] = np.inf
    for j in range(1, m + 1):
        dtw_matrix[0, j] = np.inf
    
    # Устанавливаем начальное значение в (0,0) равным 0
    dtw_matrix[0, 0] = 0
    
    # Заполняем матрицу DTW
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean_distance(ts1[i-1], ts2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                         dtw_matrix[i, j-1],
                                         dtw_matrix[i-1, j-1])
    
    # Возвращаем DTW расстояние
    return dtw_matrix[n, m]

    # INSERT YOUR CODE

    #return dtw_dist
