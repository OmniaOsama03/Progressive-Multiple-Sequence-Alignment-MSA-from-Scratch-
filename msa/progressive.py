from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

from .profile import Profile
from .alignment import align
from .distance import distance_matrix
from .scoring import scoring


#Task 6! (yippee)

def progressive_align(sequences, scoring=scoring, gap=-1):
    #Edge cases
    if len(sequences) == 0:
        raise ValueError("progressive_align requires at least one sequence.")
    if len(sequences) == 1:
        return Profile([sequences[0]])
    
    #compute the pairwise distance matrix from Task 5.
    D = distance_matrix(sequences, scoring=scoring, gap=gap)

    #convert the square distance matrix into condensed form (squareform),
    #then build the UPGMA guide tree using average linkage.
    condensed = squareform(D.values)
    Z = linkage(condensed, method='average')

    #start with one singleton Profile per original sequence.
    #Original leaf cluster IDs are 0, 1, ..., n-1.
    profiles = {i: Profile([seq]) for i, seq in enumerate(sequences)}
    n = len(sequences)

    #Each row of Z creates a new cluster with ID n + row_index.
    #Reminder- row format: [left_cluster, right_cluster, distance, cluster_size]
    for row_idx, row in enumerate(Z):
        left_id = int(row[0])
        right_id = int(row[1])
        new_id = n + row_idx

        merged = align(profiles[left_id], profiles[right_id],
                       scoring=scoring, gap=gap)

        profiles[new_id] = merged

    #the last created cluster is the root/final alignment.
    root_id = n + len(Z) - 1
    return profiles[root_id]