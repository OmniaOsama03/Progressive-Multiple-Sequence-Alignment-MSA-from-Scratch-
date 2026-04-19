import numpy as np
import pandas as pd

from .profile import Profile
from .dp_core import dp
from .scoring import scoring

# ── TASK 5 — OUR CODE ─────────────────────────────────────────────────────────
def distance_matrix(sequences, scoring=scoring, gap=-1):
    n = len(sequences)

    # Step 1 — wrap each sequence in a singleton Profile.
    # Each profile has exactly one sequence and one set of counts.
    profiles = [Profile([s]) for s in sequences]

    # Step 2 — compute alignment scores for every pair (i, j).
    # This includes self-scores: score(i, i) = best possible score for seq i.
    # scores is an (n x n) numpy array of raw alignment scores.
    scores = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # dp() returns (score_matrix, backtrack_matrix).
            # We only need the final score at position [len(p_i), len(p_j)].
            s, _ = dp(profiles[i], profiles[j], scoring, gap=gap)
            scores[i, j] = s[len(profiles[i]), len(profiles[j])]

    # Step 3 — normalize into distances using the project formula:
    # d(i, j) = 1 - score(i, j) / max(score(i, i), score(j, j))
    # max() is used so d always stays in [0, 1] regardless of sequence length.
    # Using min() would be too lenient; geometric mean can exceed 1 in edge cases.
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            self_max = max(scores[i, i], scores[j, j])
            D[i, j]  = 1 - scores[i, j] / self_max

    # Step 4 — return as a pandas DataFrame.
    # Rows and columns are labeled with the original sequence strings.
    return pd.DataFrame(D, index=sequences, columns=sequences)