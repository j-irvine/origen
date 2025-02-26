"""DNA sequence alignment and similarity computation using EMBOSS Needle."""

import tempfile
from dataclasses import dataclass
from typing import Tuple

from Bio.Emboss.Applications import NeedleCommandline
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


@dataclass
class Alignment:
    """Holds alignment results and statistics."""
    alignment: str
    identity: float
    similarity: float
    gaps: float
    score: float
    mismatches: int
    insertions: int
    deletions: int


def reverse_complement(sequence: str) -> str:
    """Return reverse complement of a DNA sequence."""
    return str(Seq(sequence).reverse_complement())


class AlignmentParser:
    @staticmethod
    def get_alignment_stats(alignment_str: str) -> Tuple[int, int, int]:
        """Extract mismatch and indel counts from alignment string."""
        mismatches = 0
        insertions = 0
        deletions = 0
        for line in str(alignment_str).split('\n'):
            if line.startswith('seq1'):
                deletions += line.count('-')
            elif line.startswith('seq2'):
                insertions += line.count('-')
            else:
                mismatches += line.count('.')
        return mismatches, insertions, deletions

    @classmethod
    def from_file(cls, filepath: str) -> Alignment:
        """Parse EMBOSS needle output file into Alignment object."""
        def parse_fraction(line: str) -> float:
            num_and_denom = line.split(":")[1].split("(")[0].strip()
            num = float(num_and_denom.split("/")[0])
            denom = float(num_and_denom.split("/")[1])
            return num / denom

        with open(filepath, 'r') as f:
            lines = f.readlines()
            alignment_str = ""
            for line in lines:
                line = line.rstrip()
                if len(line.strip()) == 0:
                    continue
                if not line.startswith('#'):
                    alignment_str += line + "\n"
                elif line.startswith('# Identity:'):
                    identity = parse_fraction(line)
                elif line.startswith('# Similarity:'):
                    similarity = parse_fraction(line)
                elif line.startswith('# Gaps:'):
                    gaps = parse_fraction(line)
                elif line.startswith('# Score:'):
                    score = float(line.split(':')[1].strip())
            
            mismatches, insertions, deletions = cls.get_alignment_stats(alignment_str)
            return Alignment(
                alignment=alignment_str,
                identity=identity,
                similarity=similarity,
                gaps=gaps,
                score=score,
                mismatches=mismatches,
                insertions=insertions,
                deletions=deletions
            )


def align(seq1: str, seq2: str, gap_open: float = 10.0, gap_extend: float = 0.5) -> Alignment:
    """Align two DNA sequences using EMBOSS needle."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta') as temp1, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.fasta') as temp2, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as output:
        
        SeqIO.write(SeqRecord(Seq(seq1), id="seq1"), temp1.name, "fasta")
        SeqIO.write(SeqRecord(Seq(seq2), id="seq2"), temp2.name, "fasta")
        temp1.flush()
        temp2.flush()

        needle_cline = NeedleCommandline(
            asequence=temp1.name,
            bsequence=temp2.name,
            gapopen=gap_open,
            gapextend=gap_extend,
            outfile=output.name
        )
        stdout, stderr = needle_cline()
        return AlignmentParser.from_file(output.name)


def compute_sequence_similarity(seq1: str, 
                              seq2: str,
                              gap_open: float = 10.0,
                              gap_extend: float = 0.5,
                              identity: bool = False,
                              check_reverse_complement: bool = True,
                              reverse_complement_threshold: float = 0.7) -> float:
    """Calculate sequence similarity using needle alignment.
    
    Returns similarity on 0-1 scale. Checks reverse complement if direct similarity
    is below threshold (default 0.7)."""
    
    alignment = align(seq1, seq2, gap_open, gap_extend)
    similarity = alignment.identity if identity else alignment.similarity
    
    if (check_reverse_complement and 
        similarity < reverse_complement_threshold):
        rev_complement_alignment = align(
            seq1,
            reverse_complement(seq2),
            gap_open,
            gap_extend
        )
        rev_similarity = (rev_complement_alignment.identity 
                         if identity 
                         else rev_complement_alignment.similarity)
        similarity = max(similarity, rev_similarity)
        
    return similarity