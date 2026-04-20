import copy
import numpy as np

from Bio import Align
from Bio.Align import substitution_matrices

from msa.profile import Profile
from msa.dp_core import dp
from msa.scoring import scoring
from msa.alignment import align


# ============================================================
# 1) Biopython pairwise verification
# ============================================================

def make_biopython_aligner(gap=-1.0):
    """
    Create a Biopython global pairwise aligner using BLOSUM62
    and a linear gap score.
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.gap_score = float(gap)
    return aligner


def alignment_to_gapped_strings(alignment, seq1, seq2):
    """
    Reconstruct gapped strings from a Biopython Alignment object.

    alignment.aligned gives matched/aligned blocks as coordinate pairs.
    We rebuild the full gapped strings by filling the unmatched regions
    with '-' on the opposite sequence.
    """
    seq1_blocks, seq2_blocks = alignment.aligned

    out1 = []
    out2 = []

    i = 0
    j = 0

    for (s1_start, s1_end), (s2_start, s2_end) in zip(seq1_blocks, seq2_blocks):
        # Gap in seq2 before the next aligned block
        if i < s1_start:
            out1.append(seq1[i:s1_start])
            out2.append("-" * (s1_start - i))
            i = s1_start

        # Gap in seq1 before the next aligned block
        if j < s2_start:
            out1.append("-" * (s2_start - j))
            out2.append(seq2[j:s2_start])
            j = s2_start

        # Aligned block
        out1.append(seq1[s1_start:s1_end])
        out2.append(seq2[s2_start:s2_end])
        i = s1_end
        j = s2_end

    # Trailing gaps
    if i < len(seq1):
        out1.append(seq1[i:])
        out2.append("-" * (len(seq1) - i))

    if j < len(seq2):
        out1.append("-" * (len(seq2) - j))
        out2.append(seq2[j:])

    return "".join(out1), "".join(out2)


def our_pairwise_score(seq1, seq2, gap=-1):
    p1 = Profile([seq1])
    p2 = Profile([seq2])
    s, _ = dp(p1, p2, scoring, gap=gap)
    return s[len(p1), len(p2)]


def our_pairwise_alignment(seq1, seq2, gap=-1):
    merged = align(Profile([seq1]), Profile([seq2]), gap=gap)
    return merged.sequences[0], merged.sequences[1]


def verify_pairwise_against_biopython(examples, gap=-1):
    aligner = make_biopython_aligner(gap=gap)

    print("=" * 80)
    print("PAIRWISE VERIFICATION AGAINST BIOPYTHON")
    print("=" * 80)

    for idx, (seq1, seq2) in enumerate(examples, start=1):
        print(f"\nExample {idx}")
        print("-" * 80)
        print("Input seq1:", seq1)
        print("Input seq2:", seq2)

        # Our implementation
        our_score = our_pairwise_score(seq1, seq2, gap=gap)
        our_a1, our_a2 = our_pairwise_alignment(seq1, seq2, gap=gap)

        # Biopython
        bio_score = aligner.score(seq1, seq2)
        bio_alignment = aligner.align(seq1, seq2)[0]
        bio_a1, bio_a2 = alignment_to_gapped_strings(bio_alignment, seq1, seq2)

        print("\nOur alignment:")
        print(" ", our_a1)
        print(" ", our_a2)
        print("Our score:", our_score)

        print("\nBiopython alignment:")
        print(" ", bio_a1)
        print(" ", bio_a2)
        print("Biopython score:", bio_score)

        print("\nChecks:")
        print(" score matches         :", np.isclose(our_score, bio_score))
        print(" recovered originals   :", our_a1.replace("-", "") == seq1 and our_a2.replace("-", "") == seq2)
        print(" same aligned length   :", len(our_a1) == len(our_a2))
        print(" Biopython same length :", len(bio_a1) == len(bio_a2))


# ============================================================
# 2) Task 4 dispute: two competing variants
# ============================================================

def alignProfiles_variant(profile1, profile2, b, use_other_profile_n=False):
    """
    Compare the two Task 4 interpretations.

    use_other_profile_n = False
        Insert gap counts using the profile being modified:
        - gap in p2 -> use p2.n_seq
        - gap in p1 -> use p1.n_seq

    use_other_profile_n = True
        Insert gap counts using the opposite profile:
        - gap in p2 -> use p1.n_seq
        - gap in p1 -> use p2.n_seq
    """
    p1 = copy.deepcopy(profile1)
    p2 = copy.deepcopy(profile2)

    i = len(p1)
    j = len(p2)
    moves = []

    while i > 0 or j > 0:
        move = b[i, j]
        moves.append(move)

        if move == 0:
            i -= 1
        elif move == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    moves.reverse()

    pos = 0
    for move in moves:
        if move == 0:
            n = p1.n_seq if use_other_profile_n else p2.n_seq
            p2.insert_gap_column(pos, n=n)
            pos += 1
        elif move == 1:
            n = p2.n_seq if use_other_profile_n else p1.n_seq
            p1.insert_gap_column(pos, n=n)
            pos += 1
        else:
            pos += 1

    return p1 + p2


def align_variant(pa, pb, gap=-1, use_other_profile_n=False):
    _, b = dp(pa, pb, scoring, gap=gap)
    return alignProfiles_variant(pa, pb, b, use_other_profile_n=use_other_profile_n)


def profile_invariant_report(profile):
    """
    In a valid Profile:
      - each column count sum should equal profile.n_seq
      - each frequency column sum should equal 1
    """
    col_sums = profile.counts.sum(axis=0)
    freq_sums = [profile[col].sum() for col in range(len(profile))]

    return {
        "n_seq": profile.n_seq,
        "length": profile.length,
        "column_count_sums": col_sums.tolist(),
        "frequency_sums": [float(x) for x in freq_sums],
        "count_sum_ok": np.allclose(col_sums, profile.n_seq),
        "freq_sum_ok": np.allclose(freq_sums, 1.0),
    }


def print_profile_report(title, profile):
    report = profile_invariant_report(profile)
    print(f"\n{title}")
    print("-" * 80)
    print("n_seq              :", report["n_seq"])
    print("length             :", report["length"])
    print("sequences:")
    for s in profile.sequences:
        print(" ", s)
    print("column count sums  :", report["column_count_sums"])
    print("frequency sums     :", [round(x, 4) for x in report["frequency_sums"]])
    print("count_sum_ok       :", report["count_sum_ok"])
    print("freq_sum_ok        :", report["freq_sum_ok"])


def verify_task4_variants():
    print("\n" + "=" * 80)
    print("TASK 4 VARIANT CHECK")
    print("=" * 80)

    # --------------------------------------------------------
    # Case A: 1-vs-1 (both variants often look the same here)
    # --------------------------------------------------------
    pa = Profile(['ACDE'])
    pb = Profile(['ADE'])

    merged_self = align_variant(pa, pb, use_other_profile_n=False)
    merged_other = align_variant(pa, pb, use_other_profile_n=True)

    print("\nCASE A — 1-vs-1")
    print_profile_report("Variant 1: use modified profile's own n_seq", merged_self)
    print_profile_report("Variant 2: use other profile's n_seq", merged_other)

    # --------------------------------------------------------
    # Case B: 2-vs-1 (this is where the disagreement matters)
    # --------------------------------------------------------
    left = align(Profile(['ACDE']), Profile(['ADE']))   # n_seq = 2
    right = Profile(['CDER'])                           # n_seq = 1

    merged_self = align_variant(left, right, use_other_profile_n=False)
    merged_other = align_variant(left, right, use_other_profile_n=True)

    print("\nCASE B — 2-vs-1")
    print_profile_report("Variant 1: use modified profile's own n_seq", merged_self)
    print_profile_report("Variant 2: use other profile's n_seq", merged_other)

    # --------------------------------------------------------
    # Case C: 2-vs-2 (can hide the difference again)
    # --------------------------------------------------------
    left = align(Profile(['ACDE']), Profile(['ADE']))   # n_seq = 2
    right = align(Profile(['ACDF']), Profile(['ADF']))  # n_seq = 2

    merged_self = align_variant(left, right, use_other_profile_n=False)
    merged_other = align_variant(left, right, use_other_profile_n=True)

    print("\nCASE C — 2-vs-2")
    print_profile_report("Variant 1: use modified profile's own n_seq", merged_self)
    print_profile_report("Variant 2: use other profile's n_seq", merged_other)

    # --------------------------------------------------------
    # Case D: 3-vs-1 (stronger stress test)
    # FIXED: build the 3-sequence left profile separately for
    # each interpretation, so the test is not contaminated by
    # an earlier disputed 2-vs-1 alignment.
    # --------------------------------------------------------
    seed = align(Profile(['ACDE']), Profile(['ADE']))  # 1-vs-1, usually identical across variants

    left_self = align_variant(seed, Profile(['CDER']), use_other_profile_n=False)  # n_seq = 3
    left_other = align_variant(seed, Profile(['CDER']), use_other_profile_n=True)  # n_seq = 3

    right = Profile(['ADER'])  # n_seq = 1

    merged_self = align_variant(left_self, right, use_other_profile_n=False)
    merged_other = align_variant(left_other, right, use_other_profile_n=True)

    print("\nCASE D — 3-vs-1")
    print_profile_report("Variant 1: use modified profile's own n_seq", merged_self)
    print_profile_report("Variant 2: use other profile's n_seq", merged_other)

# ============================================================
# 3) Main
# ============================================================

if __name__ == "__main__":
    pairwise_examples = [
        ("ACDE", "ADE"),
        ("ACDE", "ACDF"),
        ("MKTAYIAKQRQISFVKSHFSRQ", "MKTAYIAKQRQISFVKSHFSRL"),
        ("MKTAYIAKQRQISFVKSHFSRQ", "MKTAYIAQRQISFVKSHFSRQ"),
        ("MKTAIAKQRQISFVKSHFSR", "MKTAYIAKQRQISFVKSHFSRQ"),
    ]

    verify_pairwise_against_biopython(pairwise_examples, gap=-1)
    verify_task4_variants()