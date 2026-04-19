from msa.constants import BLOSUM62
from msa.dp_core import dp
from msa.profile import Profile
from msa.scoring import delta_profile, scoring
from msa.alignment import  align
from msa.distance import distance_matrix
from msa.progressive import progressive_align

import numpy as np

# ── TESTS 
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
    print("TEST 7 — Official project test")
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

    print()
    print("=" * 50)
    print("TEST 15 — Task 6: progressive_align basic sanity test")
    print("=" * 50)
    seqs = ['ACDE', 'ADE', 'CDER']
    result = progressive_align(seqs)
    print("Aligned sequences:")
    for seq in result.sequences:
        print(" ", seq)

    aligned_lengths = [len(seq) for seq in result.sequences]
    recovered = sorted(seq.replace('-', '') for seq in result.sequences)
    original  = sorted(seqs)

    print("n_seq               :", result.n_seq)                      # expect: 3
    print("All same length     :", len(set(aligned_lengths)) == 1)   # expect: True
    print("Recovered originals :", recovered)                        # expect: sorted input seqs
    print("Matches input seqs  :", recovered == original)            # expect: True

    print()
    print("=" * 50)
    print("TEST 16 — Task 6: progressive_align on project sequences")
    print("=" * 50)
    seqs = [
        'MKTAYIAKQRQISFVKSHFSRQ',
        'MKTAYIAKQRQISFVKSHFSRL',
        'MKTAYIAQRQISFVKSHFSRQ',
        'MKTAIAKQRQISFVKSHFSR'
    ]
    result = progressive_align(seqs)
    print("Final progressive alignment:")
    for seq in result.sequences:
        print(" ", seq)

    aligned_lengths = [len(seq) for seq in result.sequences]
    recovered = sorted(seq.replace('-', '') for seq in result.sequences)
    original  = sorted(seqs)

    print("\nNumber of aligned sequences:", result.n_seq)               # expect: 4
    print("Aligned length            :", result.length)                # expect: common aligned length
    print("All same length           :", len(set(aligned_lengths)) == 1)  # expect: True
    print("Recovered originals match :", recovered == original)        # expect: True