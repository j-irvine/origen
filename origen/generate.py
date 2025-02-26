"""Generation utilities for OriGen models."""

import torch
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_sequences(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    num_sequences: int,
    max_length: int,
    prompt: str = "",
    temperature: float = 1.0,
    batch_size: int = 8,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: Optional[str] = None,
) -> List[str]:
    """Generate novel replicon sequences using a trained OriGen model.
    
    Args:
        model: Trained OriGen model
        tokenizer: Associated tokenizer
        num_sequences: Number of sequences to generate
        max_length: Maximum sequence length
        prompt: Input prompt (e.g. species name and/or Rep sequence)
        temperature: Sampling temperature (0 for greedy decoding)
        batch_size: Number of sequences to generate per batch
        top_k: Top-k sampling parameter 
        top_p: Top-p sampling parameter
        device: Device to run generation on (defaults to model device)
    
    Returns:
        List of generated sequences
    """
    if device is None:
        device = next(model.parameters()).device
        
    # Prepare input tokens
    input_ids = torch.tensor([[tokenizer.bos_token_id] + tokenizer.encode(prompt)]).to(device)
    input_ids = input_ids.repeat(batch_size, 1)
    
    generated_sequences = []
    num_batches = (num_sequences + batch_size - 1) // batch_size
    final_batch_size = num_sequences % batch_size or batch_size
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            current_batch_size = final_batch_size if batch_idx == num_batches - 1 else batch_size
            current_input_ids = input_ids[:current_batch_size]
                
            output = model.generate(
                current_input_ids,
                attention_mask=torch.ones_like(current_input_ids),
                max_length=max_length,
                num_return_sequences=1,
                do_sample=temperature > 0,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            for seq in output:
                decoded_seq = tokenizer.decode(seq)
                if tokenizer.eos_token in decoded_seq:
                    decoded_seq = decoded_seq[:decoded_seq.find(tokenizer.eos_token)]
                generated_sequences.append(decoded_seq)
            
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
                
    return generated_sequences