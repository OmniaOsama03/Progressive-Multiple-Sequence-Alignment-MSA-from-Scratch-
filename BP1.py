import copy
import numpy as np
import pandas as pd
from numpy import zeros, arange, argmax
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

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


# ── DR'S CODE ─────────────────────────────────────────────────────────────────
def dp(v, w, delta, gap=0, local=False):
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

    # ── TASK 1 — DR'S CODE ────────────────────────────────────────────────
    # sorted() on the string gives a list sorted by ASCII value.
    # '-' has ASCII 45, all letters start at 65, so '-' lands at index 0.
    # This matters because BLOSUM62 row/col 0 are all zeros (gap score = 0).
    AA = sorted("ACDEFGHIKLMNPQRSTVWY-")

    def __init__(self, sequences=[], length=0, alphabet=None):

        # ── DR'S CODE ─────────────────────────────────────────────────────
        self.alphabet  = alphabet if alphabet is not None else Profile.AA
        self.sym2idx   = {s: i for i, s in enumerate(self.alphabet)}
        self.sequences = sequences

        # ── TASK 1 — OUR CODE ─────────────────────────────────────────────
        if sequences:
            # CASE 1: a real sequence was given, e.g. Profile(['ACDE'])
            # Derive length from first sequence (all must be same length)
            self.length = len(sequences[0])
            self.n_seq  = len(sequences)

            # counts matrix shape: (21 symbols, length positions)
            # dtype=float so that freqs = counts/n_seq works correctly
            self.counts = np.zeros((len(self.alphabet), self.length), dtype=float)

            # Walk every sequence character by character.
            # Find the row index for that amino acid and increment.
            for seq in sequences:
                for pos, aa in enumerate(seq):
                    self.counts[self.sym2idx[aa], pos] += 1
        else:
            # CASE 2: no sequence given, just a length.
            # Used internally by __add__ to create a blank merged profile.
            self.length = length
            self.n_seq  = 0
            self.counts = np.zeros((len(self.alphabet), length), dtype=float)

    # ── TASK 1 — DR'S CODE ────────────────────────────────────────────────
    # freqs is a property: computed on demand, never stored.
    # We store counts (not freqs) so merging is just counts_a + counts_b.
    # If we stored freqs, merging would require re-weighting by n_seq first.
    @property
    def freqs(self):
        return self.counts / self.n_seq

    # ── TASK 1 — OUR CODE ─────────────────────────────────────────────────
    def insert_gap_column(self, pos, n=0):
        # Build a new column vector: all zeros except the gap row (index 0).
        # n = number of sequences in the OTHER profile that caused this gap.
        new_col = np.zeros(len(self.alphabet), dtype=float)
        new_col[0] = n

        # np.insert(array, position, values, axis=1) inserts a new COLUMN.
        # counts goes from shape (21, L) to (21, L+1).
        self.counts = np.insert(self.counts, pos, new_col, axis=1)

        # Insert '-' at position pos in every sequence string.
        # s[:pos] = everything before pos, s[pos:] = everything from pos onward.
        self.sequences = [s[:pos] + '-' + s[pos:] for s in self.sequences]

        # Profile is now one column longer.
        self.length += 1

    def __add__(self, other):
        # Both profiles must have the same number of columns after alignment.
        assert self.length == other.length, "Profiles must have the same length to merge"

        # Create a blank profile of the same length (Case 2 of __init__).
        merged           = Profile([], length=self.length, alphabet=self.alphabet)

        # Element-wise addition of count matrices.
        # This is why we store counts: merging is trivially just addition.
        merged.counts    = self.counts + other.counts

        # Concatenate the two sequence lists.
        merged.sequences = self.sequences + other.sequences

        # Total sequences = sum of both.
        merged.n_seq     = self.n_seq + other.n_seq

        return merged

    # ── TASK 1 — DR'S CODE ────────────────────────────────────────────────
    def __len__(self):
        # Makes len(profile) return number of columns.
        # Used by dp() to set loop bounds: range(1, n+1).
        return self.length

    def __getitem__(self, col):
        # Makes profile[i] return column i as a frequency vector, shape (21,).
        # Used by dp() as: delta(v[i-1], w[j-1]).
        return self.counts[:, col] / self.n_seq

    def __repr__(self):
        # Prints the profile as a pandas DataFrame when you call print(profile).
        # Rows = alphabet symbols, columns = positions.
        return pd.DataFrame(self.freqs, index=self.alphabet).to_string()


# ── TASK 2 — OUR CODE ─────────────────────────────────────────────────────────
def delta_profile(col_v, col_w, scoring_matrix):
    # Score between two profile columns using the bilinear form:
    # delta = col_v^T * scoring_matrix * col_w
    # col_v and col_w are frequency vectors of shape (21,).
    # With identity matrix this reduces to the dot product col_v @ col_w.
    return col_v @ scoring_matrix @ col_w

# Default scoring function: wraps delta_profile with BLOSUM62.
# This is what gets passed as 'delta' into dp().
scoring = lambda col1, col2: delta_profile(col1, col2, BLOSUM62)


# ── TASK 4 — OUR CODE ─────────────────────────────────────────────────────────
def alignProfiles(profile1, profile2, b):
    # Work on deep copies so we never mutate the original profiles.
    # Deep copy means every nested object (counts matrix, sequences list)
    # is also copied, not just the Profile object itself.
    p1 = copy.deepcopy(profile1)
    p2 = copy.deepcopy(profile2)

    # STEP 1 — BACKTRACK from bottom-right corner to top-left corner.
    # i = row index (profile1 length), j = col index (profile2 length).
    i = len(p1)
    j = len(p2)

    moves = []   # stores moves in REVERSE order as we backtrack

    while i > 0 or j > 0:
        move = b[i, j]
        moves.append(move)

        if move == 0:
            i -= 1       # came from above: gap in profile2, only i moves
        elif move == 1:
            j -= 1       # came from left: gap in profile1, only j moves
        else:
            i -= 1       # came from diagonal: match, both move
            j -= 1

    # STEP 2 — REVERSE so moves are in forward (left to right) order.
    moves = moves[::-1]

    # STEP 3 — REPLAY FORWARD.
    # pos tracks the current column we are inserting at.
    # Every insertion shifts the profile right by 1, so pos always increments.
    pos = 0

    for move in moves:
        if move == 0:
            # Gap in profile2: insert a '-' column into p2 at current pos.
            # n = p1.n_seq because that many sequences face this gap.
            p2.insert_gap_column(pos, n=p1.n_seq)
            pos += 1
        elif move == 1:
            # Gap in profile1: insert a '-' column into p1 at current pos.
            # n = p2.n_seq because that many sequences face this gap.
            p1.insert_gap_column(pos, n=p2.n_seq)
            pos += 1
        else:
            # Match/mismatch: both columns align, no gap needed.
            pos += 1

    # Merge the two now-equal-length profiles and return.
    return p1 + p2


def align(pa, pb, scoring=scoring, gap=-1):
    # Convenience wrapper: runs dp then alignProfiles in one call.
    # Returns the merged profile directly.
    _, b = dp(pa, pb, scoring, gap=gap)
    return alignProfiles(pa, pb, b)


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


# ── TESTS ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 50)
    print("TEST 1 — Basic construction")
    print("=" * 50)
    p = Profile(['ACD'])
    print("p.length      :", p.length)           # expect: 3
    print("p.n_seq       :", p.n_seq)             # expect: 1
    print("p.counts.shape:", p.counts.shape)      # expect: (21, 3)

    print()
    print("=" * 50)
    print("TEST 2 — Counts are correct")
    print("=" * 50)
    p = Profile(['ACD'])
    # alphabet is sorted: '-'=0, 'A'=1, 'C'=2, 'D'=3, 'E'=4 ...
    print("A at pos 0    :", p.counts[1, 0])      # expect: 1.0
    print("C at pos 1    :", p.counts[2, 1])      # expect: 1.0
    print("D at pos 2    :", p.counts[3, 2])      # expect: 1.0
    print("Gap at pos 0  :", p.counts[0, 0])      # expect: 0.0

    print()
    print("=" * 50)
    print("TEST 3 — Print full profile")
    print("=" * 50)
    p = Profile(['ACD'])
    print(p)
    # expect: 21-row DataFrame, 3 columns
    # A row col 0 = 1.0, C row col 1 = 1.0, D row col 2 = 1.0, rest = 0.0

    print()
    print("=" * 50)
    print("TEST 4 — insert_gap_column")
    print("=" * 50)
    p = Profile(['ACD'])
    print("Before:", p.sequences)
    p.insert_gap_column(1, n=1)
    print("After :", p.sequences)                 # expect: ['A-CD']
    print("Length:", p.length)                    # expect: 4
    print("Gap count at pos 1:", p.counts[0, 1])  # expect: 1.0

    print()
    print("=" * 50)
    print("TEST 5 — __add__ merging")
    print("=" * 50)
    p1 = Profile(['ACD'])
    p2 = Profile(['AKD'])
    merged = p1 + p2
    print("Sequences:", merged.sequences)          # expect: ['ACD', 'AKD']
    print("n_seq    :", merged.n_seq)              # expect: 2
    print("Length   :", merged.length)             # expect: 3

    print()
    print("=" * 50)
    print("TEST 6 — __len__ and __getitem__")
    print("=" * 50)
    p = Profile(['ACDE'])
    print("len(p)    :", len(p))                  # expect: 4
    col0 = p[0]
    print("p[0].shape:", col0.shape)              # expect: (21,)
    print("p[0].sum():", col0.sum())              # expect: 1.0

    print()
    print("=" * 50)
    print("TEST 7 — Official project test (from project doc)")
    print("=" * 50)
    p = Profile(['ACD'])
    print("Before:\n", p)
    p.insert_gap_column(1, n=1)
    print("\nAfter inserting gap at position 1:\n", p)
    p1 = Profile(['ACD'])
    p2 = Profile(['AKD'])
    merged = p1 + p2
    print("\nMerged profile of ['ACD'] and ['AKD']:\n", merged)

    print()
    print("=" * 50)
    print("TEST 8 — Task 2: delta_profile with identity matrix")
    print("=" * 50)
    p = Profile(['ACDE'])
    col0            = p[0]
    result_identity = delta_profile(col0, col0, np.eye(len(Profile.AA)))
    result_dot      = col0 @ col0
    print("delta_profile with I :", result_identity)
    print("dot product          :", result_dot)
    print("Match                :", np.isclose(result_identity, result_dot))  # expect: True

    print()
    print("=" * 50)
    print("TEST 9 — Task 2: scoring with BLOSUM62")
    print("=" * 50)
    pa    = Profile(['ACDE'])
    pb    = Profile(['ADE'])
    col_a = pa[0]
    col_b = pb[0]
    score = scoring(col_a, col_b)
    print("Score of A vs A using BLOSUM62:", score)  # expect: 4.0

    print()
    print("=" * 50)
    print("TEST 10 — Task 3: dp works with Profile objects")
    print("=" * 50)
    pa   = Profile(['ACDE'])
    pb   = Profile(['ADE'])
    s, b = dp(pa, pb, scoring, gap=-1)
    print("Score matrix s:\n", s)
    print("Backtrack matrix b:\n", b)
    print("Final alignment score:", s[len(pa), len(pb)])  # expect: 14

    print()
    print("=" * 50)
    print("TEST 11 — Task 4: alignProfiles basic test")
    print("=" * 50)
    pa     = Profile(['ACDE'])
    pb     = Profile(['ADE'])
    merged = align(pa, pb)
    print("Sequences:", merged.sequences)  # expect: ['ACDE', 'A-DE']
    print("Length   :", merged.length)     # expect: 4
    print("n_seq    :", merged.n_seq)      # expect: 2

    print()
    print("=" * 50)
    print("TEST 12 — Task 4: official three-profile test from project doc")
    print("=" * 50)
    pa   = Profile(['ACDE'])
    pb   = Profile(['ADE'])
    pc   = Profile(['CDER'])
    pabc = align(align(pa, pb), pc)
    print("Aligned sequences:")
    for seq in pabc.sequences:
        print(" ", seq)
    # expect:
    # ACDE-
    # A-DE-
    # -CDER

    print()
    print("=" * 50)
    print("TEST 13 — Task 5: distance matrix shape and diagonal")
    print("=" * 50)
    seqs = ['ACDE', 'ACDE', 'ACDF']
    D = distance_matrix(seqs)
    print(D)
    # diagonal must be 0.0 (sequence vs itself = distance 0)
    # ACDE vs ACDE = 0.0 (identical sequences)
    # ACDE vs ACDF = small positive number (one letter differs)
    print("\nDiagonal all zero:", np.allclose(np.diag(D.values), 0))  # expect: True
    print("Identical seqs distance:", D.iloc[0, 1])                   # expect: 0.0
    print("Different seqs distance:", round(D.iloc[0, 2], 4))         # expect: small positive

    print()
    print("=" * 50)
    print("TEST 14 — Task 5: distance matrix on project sequences")
    print("=" * 50)
    seqs = [
        'MKTAYIAKQRQISFVKSHFSRQ',
        'MKTAYIAKQRQISFVKSHFSRL',
        'MKTAYIAQRQISFVKSHFSRQ',
        'MKTAIAKQRQISFVKSHFSR'
    ]
    D = distance_matrix(seqs)
    print(D.round(4))
    # expect: symmetric matrix, diagonal = 0, off-diagonal > 0
    # sequences 0 and 1 differ only in last letter so should be very close
