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

    motifs_idx = []
    motifs_dist = []

    mp = matrix_profile['mp']
    mpi = matrix_profile['mpi']
    excl_zone = matrix_profile['excl_zone']

    sorted_indices = np.argsort(mp)[::-1]

    for idx in sorted_indices:
        if len(motifs_idx) >= top_k:
            break
        if mp[idx] != np.inf:
            zone_start = max(0, idx - excl_zone)
            zone_stop = min(mp.shape[-1], idx + excl_zone)
            motifs_idx.append([zone_start, zone_stop])
            motifs_dist.append(mp[idx])
            mp = apply_exclusion_zone(mp, idx, excl_zone, np.inf)
            


    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
        }
