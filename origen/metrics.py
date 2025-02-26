"""Evaluation metrics for sequence generation.

Contains implementations of accuracy and loss metrics used to evaluate
the OriGen model's performance on next-token prediction, as described
in the Methods section.
"""

import torch
from torch.nn import CrossEntropyLoss


def compute_accuracy(input_ids, logits, tokenizer):
    """Compute both overall and per-example accuracy scores.
    
    Args:
        input_ids: Ground truth token IDs
        logits: Model prediction logits
        tokenizer: Tokenizer used for padding token ID
        
    Returns:
        overall_accuracy: Float accuracy across all tokens
        per_example_accuracy: Tensor of per-sequence accuracies
    """
    labels = torch.tensor(input_ids)
    labels = labels[:, 1:].contiguous()  # Shift right to align with predictions
    
    # Create mask to handle padding
    mask = (labels != tokenizer.pad_token_id).float()
    
    # Get predictions and compute accuracy
    pred_tokens = logits.argmax(dim=2)[:, :-1]  # Get highest probability tokens
    correct_predictions = (pred_tokens == labels).float() * mask
    
    # Compute metrics
    overall_accuracy = correct_predictions.sum() / mask.sum()
    per_example_accuracy = correct_predictions.sum(dim=1) / mask.sum(dim=1)
    
    return overall_accuracy.item(), per_example_accuracy


def compute_overall_accuracy(input_ids, logits, tokenizer):
    """Compute overall accuracy score only.
    
    Convenience wrapper around compute_accuracy() that returns just
    the overall accuracy value.
    """
    return compute_accuracy(input_ids, logits, tokenizer)[0]


def compute_cross_entropy(input_ids, logits, tokenizer):
    """Compute cross entropy loss.
    
    Args:
        input_ids: Ground truth token IDs
        logits: Model prediction logits  
        tokenizer: Tokenizer used for padding token ID
        
    Returns:
        loss: Float cross entropy loss value
    """
    # Align predictions with labels
    shifted_prediction_scores = logits[:, :-1, :].contiguous()
    labels = torch.tensor(input_ids)
    labels = labels[:, 1:].contiguous()
    
    # Setup loss function ignoring padding tokens
    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    vocab_size = logits.shape[-1]
    
    # Compute loss
    lm_loss = loss_fct(
        shifted_prediction_scores.view(-1, vocab_size),
        labels.view(-1)
    )
    
    return lm_loss.item()