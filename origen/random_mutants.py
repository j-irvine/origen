"""Functions for generating random DNA sequences and matched random mutants."""

import numpy as np
from typing import List

from .alignment import align, reverse_complement

def generate_random_dna(length: int, n: int = 1) -> List[str]:
    """Generate random DNA sequences of specified length."""
    return [''.join(np.random.choice(['A', 'C', 'G', 'T'], size=length)) for _ in range(n)]

def generate_matched_random_mutant(ml_seq: str, wt_seq: str) -> str:
    """Generate a random mutant of wt_seq matching mutation patterns in ml_seq."""
    # Get mutation pattern from best alignment (direct or reverse complement)
    alignment = align(ml_seq, wt_seq)
    alignment_rev = align(ml_seq, reverse_complement(wt_seq))
    if alignment.score < alignment_rev.score:
        alignment = alignment_rev
        wt_seq = reverse_complement(wt_seq)

    # Create mutable sequence
    seq = list(wt_seq)
    
    # Apply deletions
    deletion_sites = np.random.choice(
        np.arange(len(seq)), 
        size=alignment.deletions, 
        replace=False
    )
    for i, site in enumerate(deletion_sites):
        seq.pop(site - i)  # Adjust index for shrinking sequence
    
    # Apply mismatches
    bases = ['A', 'C', 'G', 'T']
    mismatch_sites = np.random.choice(
        np.arange(len(seq)), 
        size=alignment.mismatches, 
        replace=False
    )
    for site in mismatch_sites:
        current_base = seq[site]
        available_bases = list(set(bases) - {current_base})
        seq[site] = np.random.choice(available_bases)
    
    # Apply insertions
    insertion_sites = np.random.choice(
        np.arange(len(seq)), 
        size=alignment.insertions, 
        replace=True
    )
    for site in insertion_sites:
        seq.insert(site, np.random.choice(bases))
    
    return ''.join(seq)

def generate_matched_random_mutants(ml_seq: str, wt_seq: str, n: int) -> List[str]:
    """Generate multiple random mutants matching mutation patterns in ml_seq."""
    return [generate_matched_random_mutant(ml_seq, wt_seq) for _ in range(n)]