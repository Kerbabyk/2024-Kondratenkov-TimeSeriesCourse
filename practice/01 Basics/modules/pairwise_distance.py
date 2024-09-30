import numpy as np
from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.utils import z_normalize

class PairwiseDistance:
    """
    Distance matrix between time series 

    Parameters
    ----------
    metric: distance metric between two time series
            Options: {euclidean, dtw}
    is_normalize: normalize or not time series
    """

    def __init__(self, metric: str = 'euclidean', is_normalize: bool = False) -> None:
        self.metric: str = metric
        self.is_normalize: bool = is_normalize
    
    @property
    def distance_metric(self) -> str:
        """Return the distance metric

        Returns
        -------
            string with metric which is used to calculate distances between set of time series
        """
        norm_str = "normalized " if self.is_normalize else "non-normalized "
        return norm_str + self.metric + " distance"

    def _choose_distance(self):
        """ Choose distance function for calculation of matrix
        
        Returns
        -------
        dist_func: function reference
        """
        if self.metric == 'euclidean':
            return norm_ED_distance if self.is_normalize else ED_distance
        elif self.metric == 'dtw':
            return DTW_distance
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def calculate(self, input_data: np.ndarray) -> np.ndarray:
        """ Calculate distance matrix
        
        Parameters
        ----------
        input_data: time series set
        
        Returns
        -------
        matrix_values: distance matrix
        """
        matrix_shape = (input_data.shape[0], input_data.shape[0])
        matrix_values = np.zeros(shape=matrix_shape)
        
        dist_func = self._choose_distance()
        
        for i in range(input_data.shape[0]):
            for j in range(i, input_data.shape[0]):
                series_i = z_normalize(input_data[i]) if self.is_normalize else input_data[i]
                series_j = z_normalize(input_data[j]) if self.is_normalize else input_data[j]
                
                distance = dist_func(series_i, series_j)
                matrix_values[i, j] = distance
                matrix_values[j, i] = distance

        return matrix_values
