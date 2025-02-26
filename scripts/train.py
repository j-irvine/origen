#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training script for OriGen models.

This script handles the training of the OriGen language model for generating plasmid
replicon sequences. It uses a transformer architecture (RoFormer) and handles custom
tokenization of biological sequences including bacterial host species names,
Rep protein sequences, and oriV sequences.
"""

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    RoFormerConfig
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from datasets import load_dataset

from origen.utils import calculate_model_size
from origen.tokenizers import tokenize_dataset

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration and architecture selection.
    
    This class handles the core model architecture decisions for OriGen,
    including model type selection and key hyperparameters.
    """

    model_size: int = field(
        metadata={"help": "Hidden dimension size for the model"},
    )
    num_attention_heads: Optional[int] = field(
        default=None,
        metadata={"help": "Number of attention heads"},
    )
    num_hidden_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of transformer layers"},
    )
    attention_probs_dropout_prob: Optional[float] = field(
        default=None,
        metadata={"help": "Dropout probability for attention heads"},
    )
    hidden_dropout_prob: Optional[float] = field(
        default=None,
        metadata={"help": "Dropout probability for hidden layers"},
    )


@dataclass
class DataTrainingArguments:
    """Arguments for data loading and preprocessing.
    
    This class handles the configuration of training data, including
    dataset selection, tokenization, and sequence processing options.
    """

    dataset_name: str = field(
        metadata={"help": "HuggingFace dataset name containing replicon sequences"},
    )
    tokenizer_name: str = field(
        metadata={"help": "HuggingFace tokenizer name to use"},
    )
    max_seq_len: int = field(
        metadata={"help": "Maximum sequence length for training"},
    )
    species_col_name: Optional[str] = field(
        default=None,
        metadata={"help": "Column name for bacterial host species"},
    )
    rep_col_name: Optional[str] = field(
        default=None,
        metadata={"help": ("Column name for the Rep protein representation. " +
                           "Expects \"rep_seq\" for full rep genes, otherwise a tokenized representation, such as pfamid")},
    )
    use_windows: Optional[bool] = field(
        default=False,
        metadata={"help": "Use sliding windows for sequences longer than max_seq_len"},
    )


def main():
    # Parse arguments
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Data loading
    logger.info(f"Loading dataset {data_args.dataset_name}...")
    raw_datasets = load_dataset(data_args.dataset_name)

    # Tokenization
    logger.info(f"Tokenizing with {data_args.tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        data_args.tokenizer_name, trust_remote_code=True
    )
    tokenized_datasets = {
        split: tokenize_dataset(
            raw_datasets[split], 
            tokenizer, 
            data_args.max_seq_len,
            species_col_name=data_args.species_col_name,
            rep_col_name=data_args.rep_col_name,
            windows=data_args.use_windows
        )
        for split in ['train', 'validation']
    }

    # Model
    logger.info("Initializing model...")
    config = RoFormerConfig(
        vocab_size=len(tokenizer.get_vocab()),
        hidden_size=model_args.model_size,
        attention_probs_dropout_prob=model_args.attention_probs_dropout_prob,
        hidden_dropout_prob=model_args.hidden_dropout_prob,
        num_attention_heads=model_args.num_attention_heads,
        num_hidden_layers=model_args.num_hidden_layers,
        is_decoder=True
    )
    model = AutoModelForCausalLM.from_config(config)

    logger.info(f"Model size: {sum(t.numel() for t in model.parameters())/1000**2:.1f}M parameters")

    # Initialize trainer
    logger.info("Training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'] if training_args.do_train else None,
        eval_dataset=tokenized_datasets['validation'] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Training
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()