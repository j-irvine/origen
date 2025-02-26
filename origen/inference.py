"""Inference utilities for running trained OriGen models.

Contains functions for efficiently running inference on trained models,
used for the evaluations and analyses described in the paper. The model
architecture is a standard transformer from HuggingFace, configured as 
described in the Methods section.
"""

import torch
from torch.utils.data import DataLoader


def model_inference(model, tokenized_data, data_collator, batch_size=16):
    """Run model inference on tokenized sequences.
    
    Args:
        model: Trained OriGen model
        tokenized_data: Dataset of tokenized sequences
        data_collator: Collation function for batching
        batch_size: Batch size for inference
        
    Returns:
        torch.Tensor: Model logits for all sequences
        
    Note:
        Uses CUDA if available, falling back to CPU if not.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    
    dataloader = DataLoader(
        tokenized_data,
        batch_size=batch_size,
        collate_fn=data_collator
    )
    
    all_results = []
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Run inference
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False
            )
            
            # Collect results
            all_results.append(outputs.logits.detach().cpu())
    
    return torch.cat(all_results, dim=0)