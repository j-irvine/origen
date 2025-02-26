"""Tokenization utilities for OriGen.

Handles tokenization of DNA sequences, species names, and Rep protein sequences
for input to the OriGen model, including special token handling for metadata.
"""

import pandas as pd
import re
from tokenizers import (
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast

# Sequence vocabularies
DNA_VOCABULARY = "ACGT"
DNA_AND_GENES_VOCABULARY = "acgtGAVLITSMCPFYWHKRDENQ"

# Dataset column names
REP_SEQ_COL = "rep_protein"
ORIV_SEQ_COL = "oriv_sequence"
SPECIES_COL = "host_species"


def make_seq_tokenizer(vocabulary=DNA_VOCABULARY):
    """Create a sequence tokenizer with special tokens."""
    special_tokens = [
        "[START]",
        "[END]",
        "[PAD]",
        "[UNK]",
        "[SEP]",
    ]
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=len(vocabulary) + len(special_tokens),
        special_tokens=special_tokens,
        initial_alphabet=list(vocabulary[0]),
    )
    tokenizer.train_from_iterator(vocabulary, trainer=trainer, length=len(vocabulary))
    
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[END]",
        bos_token="[START]",
        sep_token="[SEP]",
    )


# Token name utilities
def get_unannotated_token_name(col_name):
    """Generate token for unannotated data."""
    return f"[UNANNOTATED_{col_name.upper()}]"


def get_unknown_token_name(col_name):
    """Generate token for unknown values."""
    return f"[UNKNOWN_{col_name.upper()}]"


# Helper function extract all text not in square brackets that is lowercase
def extract_oriv(decoded_seq):
    oriv_seq = re.sub(r'\[[^\[\]]*\]', '', decoded_seq)
    oriv_seq = re.sub(r'[^a-z]+', '', oriv_seq)
    return oriv_seq


def tokenize_dataset(
    dataset,
    tokenizer,
    max_len,
    num_proc=12,
    batched=True,
    species_col_name=SPECIES_COL,
    rep_col_name=REP_SEQ_COL,
    windows=False,
):
    """Tokenize a dataset including sequence and metadata.
    
    Args:
        dataset: Input dataset
        tokenizer: Tokenizer to use
        max_len: Maximum sequence length
        num_proc: Number of processes for parallel processing
        batched: Whether to process in batches
        species_col_name: Column containing species data
        rep_col_name: Column containing rep protein data
        windows: Whether to use sliding windows
        
    Returns:
        Tokenized dataset
    """
    def get_species_token(species):
        if species is None:
            return get_unannotated_token_name("species")
        species_token = "[" + species + "]"
        return (species_token if species_token in tokenizer.vocab 
                else get_unknown_token_name("species"))

    def get_rep_tokens(reps):
        if reps is None:
            return ""
        rep_tokens = ""
        for rep in reps:
            if rep is None:
                rep_tokens += get_unannotated_token_name("reps")
            else:
                rep_token = "[" + rep + "]"
                rep_tokens += (rep_token if rep_token in tokenizer.vocab 
                             else get_unknown_token_name("reps"))
        return rep_tokens
    
    def get_rep_gene_tokens(rep_seq):
        if rep_seq is None:
            return ""
        return rep_seq.replace(":", tokenizer.sep_token)

    def preprocess_function(sample):
        seqs = []
        for i, seq in enumerate(sample[ORIV_SEQ_COL]):
            rep_tokens, species_token = "", ""
            
            # Get rep tokens
            if rep_col_name is not None:
                rep_data = sample[rep_col_name][i]
                if rep_col_name == REP_SEQ_COL:
                    rep_tokens = get_rep_gene_tokens(rep_data)
                    seq = seq.lower()  # Avoid collisions with amino acid seq
                else:
                    rep_tokens = get_rep_tokens(rep_data)
                    
            # Get species token
            if species_col_name is not None:
                species = sample[species_col_name][i]
                species_token = get_species_token(species)
                
            # Combine tokens
            seq = (tokenizer.bos_token + species_token + 
                  rep_tokens + seq + tokenizer.eos_token)
                  
            if windows:
                # Slide window by max_len / 2
                seqs.append(seq[:max_len])
                while len(seq) > max_len:
                    seq = seq[int(max_len/2):]
                    seqs.append(seq[:max_len])
            else:
                seqs.append(seq)
                
        return tokenizer(
            seqs,
            padding="max_length",
            truncation=True,
            max_length=max_len
        )

    return dataset.map(
        preprocess_function,
        batched=batched,
        num_proc=num_proc,
        remove_columns=dataset.features
    )