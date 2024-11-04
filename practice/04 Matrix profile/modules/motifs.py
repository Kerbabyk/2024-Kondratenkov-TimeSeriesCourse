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

    mp = matrix_profile.get('mp', np.array([]))
    mpi = matrix_profile.get('mpi', np.array([]))
    excl_zone = matrix_profile.get('excl_zone', 0)

    if mp.size == 0 or mpi.size == 0:
        raise KeyError("Matrix profile structure must contain 'mp' and 'mpi' keys.")

    # Сортируем индексы по значению матричного профиля
    sorted_indices = np.argsort(mp)

    for idx in sorted_indices:
        if len(motifs_idx) >= top_k:
            break
        if idx not in motifs_idx:
            motifs_idx.append((idx, idx))  # Добавляем индекс дважды, чтобы соответствовать формату
            motifs_dist.append(mp[idx])
            # Применяем зону исключения
            mp = apply_exclusion_zone(mp, idx, excl_zone, np.inf)

    return {
        "indices": motifs_idx,
        "distances": motifs_dist
    }
