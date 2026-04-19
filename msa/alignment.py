import copy

from .dp_core import dp
from .scoring import scoring

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