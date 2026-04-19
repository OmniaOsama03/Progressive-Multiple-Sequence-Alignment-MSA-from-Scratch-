import numpy as np
import pandas as pd

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