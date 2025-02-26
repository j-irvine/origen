#!/usr/bin/env python
"""
Model evaluation script for OriGen.

This script evaluates trained OriGen models by computing various metrics including:
- Overall accuracy and cross-entropy
- Performance breakdowns by data source (PLSDB vs DoriC)
- Per-species performance analysis
- Token-type analysis (nucleotide vs non-nucleotide tokens)

The evaluation results are compiled into an HTML report for easy visualization
and analysis.
"""

import argparse
import base64
from io import BytesIO

import pandas as pd
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)

from origen.tokenizers import tokenize_dataset
from origen.inference import model_inference
from origen.metrics import (
    compute_cross_entropy,
    compute_overall_accuracy,
    compute_accuracy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class EvalReport:
    """Generate an HTML evaluation report for an OriGen model.
    
    This class analyzes model performance across different data slices and metrics,
    generating a comprehensive HTML report. It includes breakdowns by data source
    (PLSDB vs DoriC), species-level analysis, and token-type analysis.
    
    Attributes:
        df: Full evaluation dataset
        plsdb_df: PLSDB subset of evaluation data
        doric_df: DoriC subset of evaluation data
        model_name: Name/path of the evaluated model
        input_ids: Token IDs of input sequences
        logits: Model output logits
        tokenizer: Model tokenizer
        html_content: Generated HTML report content
    """
    
    def __init__(
        self, 
        df: pd.DataFrame,
        input_ids: list,
        logits: torch.Tensor,
        model_name: str,
        tokenizer: AutoTokenizer
    ) -> None:
        """Initialize evaluation report generator.
        
        Args:
            df: Evaluation dataset with 'source' column indicating PLSDB/DoriC
            input_ids: List of token ID sequences
            logits: Model predictions for each sequence
            model_name: Name or path of evaluated model
            tokenizer: Associated tokenizer
        """
        self.df = df
        self.plsdb_df = df[df["source"] == "plsdb"]
        self.doric_df = df[df["source"] == "doric"]
        self.model_name = model_name
        self.input_ids = input_ids
        self.logits = logits
        self.tokenizer = tokenizer
        
        # Pre-compute sliced data if available
        if len(self.plsdb_df) > 0:
            self.plsdb_input_ids, self.plsdb_logits = self._slice_inputs_by_df(self.plsdb_df)
        if len(self.doric_df) > 0:
            self.doric_input_ids, self.doric_logits = self._slice_inputs_by_df(self.doric_df)
            
        # Initialize HTML report
        self.html_content = f"""
        <h2>Model Evaluation Report</h2>
        <p>This report shows the metrics for the validation set of model <b>{self.model_name}</b>.</p>
        """

    def _convert_table_to_html(self, table: list, header: list) -> str:
        """Convert a data table to HTML format."""
        df = pd.DataFrame(table, columns=header)
        return df.to_html(index=False, border=1)

    def _slice_inputs_by_df(self, df: pd.DataFrame) -> tuple[list, torch.Tensor]:
        """Extract input_ids and logits for a subset of data."""
        df_indexes = df.index.tolist()
        df_input_ids = [self.input_ids[i] for i in df_indexes]
        df_logits = self.logits[df_indexes]
        return df_input_ids, df_logits

    def _slice_metric(self, df: pd.DataFrame, slice_col_name: str, topn: int = 10) -> list:
        """Compute metrics for data slices based on column values.
        
        Args:
            df: DataFrame to analyze
            slice_col_name: Column to use for slicing (e.g., 'species')
            topn: Number of top categories to analyze (None for all)
            
        Returns:
            List of tuples with (slice_name, count, cross_entropy, accuracy)
        """
        if topn:
            all_slices = df[slice_col_name].value_counts()[:topn].keys().tolist()
        else:
            all_slices = df[slice_col_name].unique()
            
        all_slices_input_ids, all_slices_logits = self._slice_inputs_by_df(df)
        metric_table = []
        
        # Compute overall metrics
        accuracy = compute_overall_accuracy(all_slices_input_ids, all_slices_logits, self.tokenizer)
        cross_entropy = compute_cross_entropy(all_slices_input_ids, all_slices_logits, self.tokenizer)
        metric_table.append(("Overall", len(df), cross_entropy, accuracy))
        
        # Compute per-slice metrics
        for col_slice in all_slices:
            slice_df = df[df[slice_col_name] == col_slice]
            slice_input_ids, slice_logits = self._slice_inputs_by_df(slice_df)
            slice_accuracy = compute_overall_accuracy(slice_input_ids, slice_logits, self.tokenizer)
            slice_cross_entropy = compute_cross_entropy(slice_input_ids, slice_logits, self.tokenizer)
            metric_table.append((col_slice, len(slice_df), slice_cross_entropy, slice_accuracy))
            
        return metric_table

    def add_overall_metric_table(self) -> None:
        """Add overall performance metrics to the report."""
        overall_cross_entropy = compute_cross_entropy(self.input_ids, self.logits, self.tokenizer)
        overall_accuracy = compute_overall_accuracy(self.input_ids, self.logits, self.tokenizer)
        metric_table = [("Overall", overall_cross_entropy, overall_accuracy)]
        
        # Add PLSDB metrics if available
        if len(self.plsdb_df) > 0:
            plsdb_cross_entropy = compute_cross_entropy(self.plsdb_input_ids, self.plsdb_logits, self.tokenizer)
            plsdb_accuracy = compute_overall_accuracy(self.plsdb_input_ids, self.plsdb_logits, self.tokenizer)
            metric_table.append(("PLSDB", plsdb_cross_entropy, plsdb_accuracy))
            
        # Add DoriC metrics if available
        if len(self.doric_df) > 0:
            doric_cross_entropy = compute_cross_entropy(self.doric_input_ids, self.doric_logits, self.tokenizer)
            doric_accuracy = compute_overall_accuracy(self.doric_input_ids, self.doric_logits, self.tokenizer)
            metric_table.append(("DoriC", doric_cross_entropy, doric_accuracy))
            
        overall_metric_table_html = self._convert_table_to_html(
            metric_table, ["Data Source", "Cross Entropy", "Accuracy"]
        )
        self.html_content += f"<h3>Overall Metrics</h3>{overall_metric_table_html}"

    def add_accuracy_histogram(self) -> None:
        """Add histogram of per-sequence accuracies to the report."""
        plt.figure(figsize=(10, 6))
        
        if len(self.plsdb_df) > 0 and len(self.doric_df) > 0:
            # Split by data source if both are present
            plsdb_accuracies = compute_accuracy(self.plsdb_input_ids, self.plsdb_logits, self.tokenizer)[1]
            doric_accuracies = compute_accuracy(self.doric_input_ids, self.doric_logits, self.tokenizer)[1]
            plt.hist(plsdb_accuracies, bins=20, label='PLSDB')
            plt.hist(doric_accuracies, bins=20, label='DoriC')
            plt.legend()
        else:
            # Single histogram for combined data
            accuracies = compute_accuracy(self.input_ids, self.logits, self.tokenizer)[1]
            plt.hist(accuracies, bins=20)
            
        plt.xlabel('Sequence Accuracy')
        plt.ylabel('Frequency')
        
        # Convert plot to base64 for HTML embedding
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()
        
        self.html_content += f"""
        <h3>Accuracy Distribution</h3>
        <img src="data:image/png;base64,{img_str}" alt="Accuracy histogram" width="600">
        """

    def _get_slice_table_html(self, df: pd.DataFrame, col_name: str, topn: int = 5) -> str:
        """Generate HTML table for metrics sliced by a column."""
        slice_table = self._slice_metric(df, col_name, topn=topn)
        slice_table_html = self._convert_table_to_html(
            slice_table, 
            ["Slice", "Count", "Cross Entropy", "Accuracy"]
        )
        return f"""
        <h4>Performance by {col_name}</h4>
        {slice_table_html}
        """
    
    def add_slice_metrics(self) -> None:
        """Add performance metrics for different data slices to the report."""
        if len(self.plsdb_df) > 0 and len(self.doric_df) > 0:
            self.html_content += "<h3>PLSDB Metrics</h3>"
            self.html_content += self._get_slice_table_html(self.plsdb_df, 'species')
            self.html_content += self._get_nt_table_html(self.plsdb_df)
            self.html_content += "<h3>DoriC Metrics</h3>"
            self.html_content += self._get_slice_table_html(self.doric_df, 'species')
            self.html_content += self._get_nt_table_html(self.doric_df)
        else:
            self.html_content += "<h3>Slice Metrics</h3>"
            self.html_content += self._get_slice_table_html(self.df, 'species')
            self.html_content += self._get_nt_table_html(self.df)

    def _nt_slice_metric(self, df: pd.DataFrame) -> list:
        """Compute metrics separately for nucleotide and non-nucleotide tokens."""
        input_ids, logits = self._slice_inputs_by_df(df)
        nucleotides = set("acgt")
        
        # Initialize containers for nucleotide and non-nucleotide sequences
        nt_data = {"input_ids": [], "logits": [], "count": 0}
        non_nt_data = {"input_ids": [], "logits": [], "count": 0}
        
        # Process each sequence
        for i, seq_ids in enumerate(input_ids):
            # Find where nucleotides start
            nt_start_idx = None
            for j, tok_id in enumerate(seq_ids):
                if self.tokenizer.decode(tok_id) in nucleotides:
                    nt_start_idx = j
                    break
                    
            seq_len = len(seq_ids)
            if nt_start_idx is None:
                # No nucleotides found
                nt_data["input_ids"].append([self.tokenizer.pad_token_id] * seq_len)
                nt_data["logits"].append(torch.zeros(seq_len, logits.shape[2]))
                non_nt_data["input_ids"].append(seq_ids)
                non_nt_data["logits"].append(logits[i])
                non_nt_data["count"] += seq_len
            else:
                # Split sequence at nucleotide boundary
                nt_ids = seq_ids[nt_start_idx:] + [self.tokenizer.pad_token_id] * nt_start_idx
                non_nt_ids = seq_ids[:nt_start_idx] + [self.tokenizer.pad_token_id] * (seq_len - nt_start_idx)
                
                # Split logits similarly
                nt_log = torch.cat((
                    logits[i][nt_start_idx:],
                    torch.zeros(nt_start_idx, logits.shape[2])
                ))
                non_nt_log = torch.cat((
                    logits[i][:nt_start_idx],
                    torch.zeros(seq_len - nt_start_idx, logits.shape[2])
                ))
                
                # Store split sequences
                nt_data["input_ids"].append(nt_ids)
                nt_data["logits"].append(nt_log)
                non_nt_data["input_ids"].append(non_nt_ids)
                non_nt_data["logits"].append(non_nt_log)
                
                # Update counts
                nt_data["count"] += sum(1 for id in nt_ids if id != self.tokenizer.pad_token_id)
                non_nt_data["count"] += nt_start_idx

        # Compute metrics
        results = []
        for data, label in [(nt_data, "Nucleotides"), (non_nt_data, "Non-nucleotides")]:
            logits_tensor = torch.stack(data["logits"])
            avg_count = int(data["count"] / len(input_ids))
            cross_ent = compute_cross_entropy(data["input_ids"], logits_tensor, self.tokenizer)
            accuracy = compute_overall_accuracy(data["input_ids"], logits_tensor, self.tokenizer)
            results.append((label, avg_count, cross_ent, accuracy))
            
        return results

    def _get_nt_table_html(self, df: pd.DataFrame) -> str:
        """Generate HTML table for nucleotide/non-nucleotide metrics."""
        nt_table = self._nt_slice_metric(df)
        return f"""
        <h4>Token Type Analysis</h4>
        {self._convert_table_to_html(
            nt_table,
            ["Token type", "Avg tokens per seq", "Cross Entropy", "Accuracy"]
        )}
        """

    def create_html_report(self) -> str:
        """Generate complete HTML report with all metrics and visualizations."""
        self.add_overall_metric_table()
        self.add_accuracy_histogram()
        self.add_slice_metrics()
        return self.html_content


def main():
    """Run model evaluation and generate HTML report."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate OriGen model performance across different metrics and data slices.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument('--dataset', type=str, required=True, help="HuggingFace dataset name containing validation data")
    parser.add_argument('--tokenizer', type=str, required=True, help="Name/path of tokenizer to use")
    parser.add_argument('--max_seq_len', type=int, required=True, help="Maximum sequence length for tokenization")
    parser.add_argument('--rep_col', type=str, required=True, help="Column name for Rep protein representation. Use 'pfamid_fast' or 'pfamid' for domain tokens, or 'rep_seq' for full protein sequence")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save evaluation report (without .html extension)")
    args = parser.parse_args()

    # Configure pandas display
    pd.options.display.float_format = '{:.3f}'.format

    # Load and process data
    logger.info(f"Loading dataset: {args.dataset}")
    val_dataset = load_dataset(args.dataset, trust_remote_code=True)["validation"]
    val_df = val_dataset.to_pandas()

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tokenized_ds_eval = tokenize_dataset(
        val_dataset,
        tokenizer,
        args.max_seq_len,
        species_col_name="species",
        rep_col_name=args.rep_col
    )
    eval_input_ids = tokenized_ds_eval["input_ids"]

    # Run model inference
    logger.info(f"Loading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    eval_logits = model_inference(model, tokenized_ds_eval, data_collator)

    # Generate and save report
    logger.info("Generating evaluation report...")
    eval_report = EvalReport(
        df=val_df,
        input_ids=eval_input_ids,
        logits=eval_logits,
        model_name=args.model_path,
        tokenizer=tokenizer
    )
    html_report = eval_report.create_html_report()

    output_file = f"{args.output_path}.html"
    logger.info(f"Saving report to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_report)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()