import numpy as np
import pandas as pd

class Profile:
    # TASK 1 
    # sorted() on the string gives a list sorted by ASCII value.
    # '-' has ASCII 45, all letters start at 65, so '-' lands at index 0.
    #  BLOSUM62 row/col 0 are all zeros (gap score = 0).
    AA = sorted("ACDEFGHIKLMNPQRSTVWY-")

    def __init__(self, sequences=None, length=0, alphabet=None):

        self.alphabet  = alphabet if alphabet is not None else Profile.AA
        self.sym2idx   = {s: i for i, s in enumerate(self.alphabet)}
        self.sequences = list(sequences) if sequences is not None else []

        if self.sequences:
            expected_len = len(self.sequences[0])
            if any(len(seq) != expected_len for seq in self.sequences):
                raise ValueError("All sequences in a Profile must have the same length.")
    
        if sequences:
            # CASE 1: a real sequence was given, e.g. Profile(['ACDE'])
            # Derive length from first sequence (all must be same length)
            self.length = len(sequences[0])
            self.n_seq  = len(sequences)

            # counts matrix shape: (21 symbols, length positions)
            # dtype=float so that freqs = counts/n_seq works correctly
            self.counts = np.zeros((len(self.alphabet), self.length), dtype=int)
            
            #Validation: all symbols must be valid
            for seq in self.sequences:
                for aa in seq:
                    if aa not in self.sym2idx:
                        raise ValueError(f"Invalid symbol '{aa}' in sequence.")
        
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
            self.counts = np.zeros((len(self.alphabet), length), dtype=int)

    # freqs is a property: computed on demand, never stored.
    # We store counts (not freqs) so merging is just counts_a + counts_b.
    # If we stored freqs, merging would require re-weighting by n_seq first.
    @property
    def freqs(self):
        if self.n_seq == 0:
            raise ValueError("Frequencies are undefined for an empty profile.")
        return self.counts / self.n_seqq

    def insert_gap_column(self, pos, n=0):
        # Build a new column vector: all zeros except the gap row (index 0).
        # n = number of sequences in the OTHER profile that caused this gap.
        new_col = np.zeros(len(self.alphabet), dtype=int)
        new_col[self.sym2idx['-']] = n

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
        if self.length != other.length:
            raise ValueError("Profiles must have the same length to merge.")
        if self.alphabet != other.alphabet:
            raise ValueError("Profiles must use the same alphabet to merge.")

        # Create a blank profile of the same length (Case 2 of __init__).
        merged = Profile([], length=self.length, alphabet=self.alphabet)

        # Element-wise addition of count matrices.
        # This is why we store counts: merging is trivially just addition.
        merged.counts    = self.counts + other.counts

        # Concatenate the two sequence lists.
        merged.sequences = self.sequences + other.sequences

        # Total sequences = sum of both.
        merged.n_seq     = self.n_seq + other.n_seq

        return merged

    def __len__(self):
        # Makes len(profile) return number of columns.
        # Used by dp() to set loop bounds: range(1, n+1).
        return self.length
    
    def __getitem__(self, col):
        if self.n_seq == 0:
         raise ValueError("Cannot access frequency column of an empty profile.")
        return self.counts[:, col] / self.n_seq

    def __repr__(self):
        # Prints the profile as a pandas DataFrame when you call print(profile).
        # Rows = alphabet symbols, columns = positions.
        return pd.DataFrame(self.freqs, index=self.alphabet).to_string()