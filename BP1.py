import numpy as np
import pandas as pd
from numpy import zeros, arange, argmax

BLOSUM62 = np.array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2],
       [ 0,  0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],
       [ 0, -2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3],
       [ 0, -1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2],
       [ 0, -2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3],
       [ 0,  0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3],
       [ 0, -2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2],
       [ 0, -1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1],
       [ 0, -1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2],
       [ 0, -1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1],
       [ 0, -1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1],
       [ 0, -2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2],
       [ 0, -1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3],
       [ 0, -1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1],
       [ 0, -1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2],
       [ 0,  1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2],
       [ 0,  0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2],
       [ 0,  0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1],
       [ 0, -3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2],
       [ 0, -2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7]])


def dp(v, w, delta, gap=0, local=False):
    # ── DR'S CODE ──────────────────────────────────────────────────────────
    n = len(v)
    m = len(w)
    s = zeros((n+1, m+1), dtype=int)
    b = zeros((n+1, m+1), dtype=int)
    b[0, 1:] = 1
    if not local:
        s[0] = arange(m+1) * gap
        s[:,0] = arange(n+1) * gap
    for i in range(1, n+1):
        for j in range(1, m+1):
            options = [s[i-1,j] + gap,
                       s[i,j-1] + gap,
                       s[i-1,j-1] + delta(v[i-1], w[j-1])]
            if local: options.append(0)
            bestOption = argmax(options)
            s[i,j] = options[bestOption]
            b[i,j] = bestOption
    return s, b


class Profile:
    # ── DR'S CODE ──────────────────────────────────────────────────────────
    AA = sorted("ACDEFGHIKLMNPQRSTVWY-")

    def __init__(self, sequences=[], length=0, alphabet=None):
        # ── DR'S CODE ──────────────────────────────────────────────────────
        self.alphabet  = alphabet if alphabet is not None else Profile.AA
        self.sym2idx   = {s: i for i, s in enumerate(self.alphabet)}
        self.sequences = sequences

        # ── OUR CODE ───────────────────────────────────────────────────────
        if sequences:
            # CASE 1: real sequence given e.g. Profile(['ACDE'])
            self.length = len(sequences[0])
            self.n_seq  = len(sequences)
            self.counts = np.zeros((len(self.alphabet), self.length), dtype=float)
            for seq in sequences:
                for pos, aa in enumerate(seq):
                    self.counts[self.sym2idx[aa], pos] += 1
        else:
            # CASE 2: blank profile, used internally during merging
            self.length = length
            self.n_seq  = 0
            self.counts = np.zeros((len(self.alphabet), length), dtype=float)

    # ── DR'S CODE ──────────────────────────────────────────────────────────
    # Storing counts (not freqs) means merging = simple addition.
    # Freqs are computed on demand only.
    @property
    def freqs(self):
        return self.counts / self.n_seq

    # ── OUR CODE ───────────────────────────────────────────────────────────
    def insert_gap_column(self, pos, n=0):
        # Build new column: all zeros except gap row (index 0) = n
        new_col = np.zeros(len(self.alphabet), dtype=float)
        new_col[0] = n
        # Insert into counts matrix along axis=1 (columns)
        self.counts = np.insert(self.counts, pos, new_col, axis=1)
        # Insert '-' at position pos in every sequence string
        self.sequences = [s[:pos] + '-' + s[pos:] for s in self.sequences]
        # Profile is now one column longer
        self.length += 1

    def __add__(self, other):
        assert self.length == other.length, "Profiles must have the same length to merge"
        # Create a blank profile of the same length
        merged          = Profile([], length=self.length, alphabet=self.alphabet)
        # Add count matrices element-wise
        merged.counts   = self.counts + other.counts
        # Combine sequence lists
        merged.sequences = self.sequences + other.sequences
        # Sum the number of sequences
        merged.n_seq    = self.n_seq + other.n_seq
        return merged

    # ── DR'S CODE ──────────────────────────────────────────────────────────
    def __len__(self):
        return self.length

    def __getitem__(self, col):
        return self.counts[:, col] / self.n_seq

    def __repr__(self):
        return pd.DataFrame(self.freqs, index=self.alphabet).to_string()


# ── TASK 2 — OUR CODE ───────────────────────────────────────────────────────── 
def delta_profile(col_v, col_w, scoring_matrix):
    # Expected substitution score between two profile columns: col_v^T * M * col_w
    return col_v @ scoring_matrix @ col_w
 
# Default scoring function using BLOSUM62
scoring = lambda col1, col2: delta_profile(col1, col2, BLOSUM62)
 
 
# ── TASK 3 — OUR CODE ───────────────────────────────────────────────────────── 
if __name__ == "__main__":
    # Verify dp() works with Profile objects
    pa = Profile(['ACDE'])
    pb = Profile(['ADE'])
    s, b = dp(pa, pb, scoring, gap=-1)
    print("Score matrix s:\n", s)
    print("Backtrack matrix b:\n", b)
    print("Final alignment score:", s[len(pa), len(pb)])
 
