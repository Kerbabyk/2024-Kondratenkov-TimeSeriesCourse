import numpy as np
from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.utils import z_normalize

class PairwiseDistance:
    def __init__(self, metric='euclidean', is_normalize=False):
        self.metric = metric
        self.is_normalize = is_normalize

    def norm_ED_distance(self, x, y):
        """
        Вычисляет нормализованное евклидово расстояние между двумя временными рядами.
        """
        return np.linalg.norm(x - y) / np.sqrt(len(x))

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
        else:
            # Применяем z-нормализацию для всех временных рядов, если is_normalize=True
            if self.is_normalize:
                sequences = [z_normalize(seq) for seq in sequences]
            for i in range(n):
                for j in range(i, n):
                    if self.metric == 'euclidean':
                        distance_matrix[i, j] = np.linalg.norm(sequences[i] - sequences[j])
                    else:
                        raise ValueError(f"Unsupported metric: {self.metric}")
                    distance_matrix[j, i] = distance_matrix[i, j]

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
