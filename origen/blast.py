"""Utilities for BLAST sequence analysis and similarity visualization.

Contains functions for running BLAST searches against sequence databases and analyzing/
visualizing the similarity between generated sequences and training data, as described
in the Results section of the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
import subprocess
from pathlib import Path

# Valid characters for sequence type detection
NUCLEOTIDE_CHARS = set("ACGTNRSWDYMK")
PROTEIN_CHARS = set("ACDEFGHIKLMNPQRSTVWYX*")


class BlastHit:
    """Stores and parses BLAST alignment results."""
    
    def __init__(self, line):
        results = line.split()
        self.query_id = results[0]
        self.subject_id = results[1]
        self.pident = float(results[2])
        self.length = int(results[3])
        self.mismatches = int(results[4])
        self.gap = int(results[5])
        self.qstart = int(results[6])
        self.qend = int(results[7])
        self.sstart = int(results[8])
        self.send = int(results[9])
        self.evalue = float(results[10])
        self.bit_score = float(results[11])

    def __repr__(self):
        return f"BlastHit(pident={self.pident:.1f}, length={self.length}, evalue={self.evalue:e})"


def get_seq_type(seq):
    """Determine if a sequence is nucleotide or protein."""
    seq_chars = set(seq.upper())
    if seq_chars.issubset(NUCLEOTIDE_CHARS):
        return "nucleotide"
    if seq_chars.issubset(PROTEIN_CHARS):
        return "protein" 
    return None


def get_seq_types(seqs):
    """Verify all sequences are the same type and return it."""
    seq_types = {get_seq_type(seq) for seq in seqs}
    if len(seq_types) > 1:
        raise ValueError("Sequences must be all nucleotide or all protein")
    seq_type = seq_types.pop()
    if seq_type is None:
        raise ValueError("Sequences must contain valid nucleotide or protein characters")
    return seq_type


def create_blast_db(seqs, fasta_path, db_name, seq_type=None):
    """Create a BLAST database from sequences."""
    if not seq_type:
        seq_type = get_seq_types(seqs)
        
    # Write sequences to FASTA
    records = [
        SeqRecord(Seq(seq), id=f"seq{i}") 
        for i, seq in enumerate(seqs)
    ]
    SeqIO.write(records, fasta_path, "fasta")
    
    # Create BLAST DB
    dbtype = "prot" if seq_type == "protein" else "nucl"
    cmd = ["makeblastdb", "-in", str(fasta_path), "-dbtype", dbtype, "-out", db_name]
    subprocess.run(cmd, check=True)


def blast(query_seq, db_name, seq_type=None, rev_comp=True):
    """Run BLAST for a single query sequence against a database."""
    if not seq_type:
        seq_type = get_seq_type(query_seq)

    # Create query FASTA
    query_path = Path("query_seq.fasta")
    SeqIO.write(SeqRecord(Seq(query_seq), id="query"), query_path, "fasta")
    
    # Run BLAST
    output_path = Path("blast_results.txt")
    blast_cmd = "blastn" if seq_type == "nucleotide" else "blastp"
    cmd = [
        blast_cmd, 
        "-query", str(query_path),
        "-db", db_name,
        "-out", str(output_path),
        "-num_threads", "4",
        "-outfmt", "7"
    ]
    if not rev_comp:
        cmd += ["-strand", "plus"]
        
    try:
        subprocess.run(cmd, check=True)
        with open(output_path) as f:
            lines = f.readlines()
        return BlastHit(lines[5]) if len(lines) >= 6 else None
    finally:
        # Clean up temporary files
        query_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def blast_hits(query_seqs, db_name):
    """Run BLAST for multiple query sequences."""
    hits = {}
    for i, query_seq in enumerate(query_seqs):
        hit = blast(query_seq, db_name)
        if hit:
            hits[i] = hit
    return hits


def plot_hits(hits, input_seqs, title=None):
    """Create similarity distribution heatmap from BLAST results."""
    # Extract hit statistics
    pident_values = []
    plength_values = []
    for i, hit in hits.items():
        pident_values.append(hit.pident)
        plength_values.append(hit.length / len(input_seqs[i]))
        
    # Define distribution buckets
    pident_buckets = [(75, 90), (90, 95), (95, 99), (99, 100), (100, 200)]
    plength_buckets = [(0, 0.5), (0.5, 0.9), (0.9, 0.95), 
                       (0.95, 0.99), (0.99, 1.0), (1.0, 2.)]

    # Assign values to buckets
    def assign_bucket(value, boundaries):
        for i, (low, high) in enumerate(boundaries):
            if low <= value < high:
                return i
        return -1

    pident_bins = [assign_bucket(val, pident_buckets) for val in pident_values]
    plength_bins = [assign_bucket(val, plength_buckets) for val in plength_values]

    # Create frequency matrix
    matrix = np.zeros((len(pident_buckets), len(plength_buckets)), dtype=int)
    for i, j in zip(pident_bins, plength_bins):
        if i != -1 and j != -1:
            matrix[i, j] += 1

    # Plot heatmap
    matrix = matrix[::-1, :]  # Reverse rows for better visualization
    pident_labels = [f"{low}+" for low, high in pident_buckets][::-1]
    plength_labels = [f"{low}+" for low, high in plength_buckets]

    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, cmap='viridis', fmt='d',
                xticklabels=plength_labels, yticklabels=pident_labels)

    if title:
        plt.title(title)
    plt.xlabel('Sequence Coverage')
    plt.ylabel('Percent Identity')
    
    # Print summary statistics
    print(f"Total hits: {len(hits)} ({len(hits)/len(input_seqs):.1%} of sequences)")