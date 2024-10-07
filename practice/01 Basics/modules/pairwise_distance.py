import numpy as np
from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.utils import z_normalize
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt  # Добавлен импорт matplotlib.pyplot
from sklearn.metrics import silhouette_score

class PairwiseDistance:
    def __init__(self, metric='euclidean', is_normalize=False):
        self.metric = metric
        self.is_normalize = is_normalize

    def norm_ED_distance(self, x, y):
        """
        Вычисляет нормализованное евклидово расстояние между двумя временными рядами.
        """
        return np.linalg.norm(x - y) / np.sqrt(len(x))

    def euclidean_distance(self, x, y):
        """
        Вычисляет евклидово расстояние между двумя точками.
        """
        return np.linalg.norm(x - y)

    def dtw_distance(self, T1, T2):
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
                cost = (T1[i-1] - T2[j-1]) ** 2
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
        
        # Возвращаем DTW расстояние
        return np.sqrt(dtw_matrix[n, m])

    def compute_distance_matrix(self, sequences):
        """
        Вычисляет матрицу расстояний между всеми парами временных рядов.
        """
        n = len(sequences)
        distance_matrix = np.zeros((n, n))

        if self.metric == 'norm_euclidean':
            for i in range(n):
                for j in range(i, n):
                    distance_matrix[i, j] = self.norm_ED_distance(sequences[i], sequences[j])
                    distance_matrix[j, i] = distance_matrix[i, j]
        elif self.metric == 'euclidean':
            for i in range(n):
                for j in range(i, n):
                    distance_matrix[i, j] = self.euclidean_distance(sequences[i], sequences[j])
                    distance_matrix[j, i] = distance_matrix[i, j]
        elif self.metric == 'dtw':
            for i in range(n):
                for j in range(i, n):
                    distance_matrix[i, j] = self.dtw_distance(sequences[i], sequences[j])
                    distance_matrix[j, i] = distance_matrix[i, j]
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        return distance_matrix

class TimeSeriesHierarchicalClustering:
    def __init__(self, n_clusters=3, method='ward'):
        self.n_clusters = n_clusters
        self.method = method
        self.model = None

    def fit(self, distance_matrix):
        Z = linkage(distance_matrix, method=self.method)
        self.model = Z
        return Z

    def plot_dendrogram(self, data, labels, title='Dendrogram'):
        plt.figure(figsize=(10, 5))
        dendrogram(self.model, labels=labels, leaf_rotation=90., leaf_font_size=8.)
        plt.title(title)
        plt.show()

    def get_cluster_labels(self, distance_matrix, max_d):
        return fcluster(self.model, max_d, criterion='distance')

    def silhouette_score(self, distance_matrix, cluster_labels):
        return silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
