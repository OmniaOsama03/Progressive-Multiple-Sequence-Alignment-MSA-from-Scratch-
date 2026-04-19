from .constants import BLOSUM62

#TASK 2 
def delta_profile(col_v, col_w, scoring_matrix):
    # Score between two profile columns using the bilinear form:
    # delta = col_v^T * scoring_matrix * col_w
    # col_v and col_w are frequency vectors of shape (21,).
    # With identity matrix this reduces to the dot product col_v @ col_w.
    return col_v @ scoring_matrix @ col_w

# Default scoring function: wraps delta_profile with BLOSUM62.
# This is what gets passed as 'delta' into dp().
scoring = lambda col1, col2: delta_profile(col1, col2, BLOSUM62)