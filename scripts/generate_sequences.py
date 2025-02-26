#!/usr/bin/env python
"""
Sequence generation script for OriGen models.

This script generates novel plasmid replicon sequences using a trained OriGen model.
It supports generation conditioned on bacterial host species and/or Rep protein 
sequences, with configurable generation parameters like temperature and sequence length.
"""
import argparse
import os
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from origen.generate import generate_sequences

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate novel replicon sequences using a trained OriGen model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument('--tokenizer', type=str, required=True, help="Name of tokenizer in HF")
    parser.add_argument('--max_generation_len', type=int, required=True, help="Maximum length of generated sequences")
    parser.add_argument('--temperature', type=float, required=True, help="Sampling temperature (0 for greedy decoding)")
    parser.add_argument('--num_generations', type=int, required=True, help="Number of sequences to generate")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save generated sequences")
    parser.add_argument('--species', type=str, default='Escherichia coli', help="Bacterial host species to generate sequences for")
    parser.add_argument('--seq_prompt', type=str, default='', help="Optional prompt of sequence after host species token to further condition generation")
    parser.add_argument('--append', action='store_true', help="Append to output file if it exists")
    parser.add_argument('--top_k', type=int, default=None, help="Top-k sampling parameter")
    parser.add_argument('--top_p', type=float, default=None, help="Top-p sampling parameter")
    args = parser.parse_args()

    if os.path.exists(args.output_file) and not args.append:
        raise ValueError("Output file already exists but --append is not specified.")

    # Set up output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model and tokenizer
    logging.info(f"Loading tokenizer {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    logging.info(f"Loading model {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model = model.to(device).eval()

    # Generate sequences
    logging.info(f"Generating {args.num_generations} sequences...")
    prompt = f"[{args.species}]{args.seq_prompt}"
    sequences = generate_sequences(
        model=model,
        tokenizer=tokenizer,
        num_sequences=args.num_generations,
        max_length=args.max_generation_len,
        prompt=prompt,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device
    )
    
    # Save sequences
    with open(args.output_file, 'a' if args.append else 'w') as f:
        for seq in sequences:
            f.write(f"{seq}\n")

    logger.info("Generation complete!")

if __name__ == "__main__":
    main()