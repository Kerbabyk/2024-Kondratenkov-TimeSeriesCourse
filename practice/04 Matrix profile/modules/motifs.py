import numpy as np

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    mp = matrix_profile['mp']
    pi = matrix_profile['pi']

    # Сортируем индексы по значению матричного профиля
    sorted_indices = np.argsort(mp)

    for idx in sorted_indices:
        if len(motifs_idx) >= top_k:
            break
        if idx not in motifs_idx:
            motifs_idx.append(idx)
            motifs_dist.append(mp[idx])
            # Применяем зону исключения
            mp = apply_exclusion_zone(mp, idx, excl_zone, np.inf)

    return {
        "indices": motifs_idx,
        "distances": motifs_dist
    }
