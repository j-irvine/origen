#!/bin/bash
# BLAST search script for finding closest matches between generated and training sequences.
# Usage: ./blast_search.sh <input_sequences> <training_fasta> <output_file>

set -e  # Exit on error

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_sequences> <training_fasta> <output_file>"
    exit 1
fi

QUERY_FILE=$1
DB_FASTA=$2
BLAST_OUTPUT=$3
DB_NAME="train_orivs.db"
NUM_THREADS=4

# Create BLAST database
echo "Creating BLAST database..."
makeblastdb -in ${DB_FASTA} \
    -dbtype nucl \
    -out ${DB_NAME}

# Process input sequences and prepare FASTA
echo "Processing input sequences..."
awk '{ 
    seq = $0
    # Remove [END] or [] at the end
    sub(/\[END\]$/, "", seq)
    sub(/\[\]$/, "", seq)
    # Remove spaces and convert to uppercase
    gsub(/ /, "", seq)
    gsub(/[A-Z]/, "", seq)
    print ">seq" NR "\n" toupper(seq)  
}' ${QUERY_FILE} > queries.fasta

# Run BLAST
echo "Running BLAST..."
blastn \
    -query queries.fasta \
    -db ${DB_NAME} \
    -out ${BLAST_OUTPUT}.tmp \
    -num_threads ${NUM_THREADS} \
    -outfmt "6 qseqid sseqid pident qcovs length mismatch gapopen qstart qend sstart send evalue bitscore" \
    -evalue 1e-5

# Process results to keep only best hit per query
echo "Processing results to keep best hits only..."
awk '
    {
        if ($13 > max[$1]) {
            max[$1] = $13
            best[$1] = $0
        }
    }
    END {
        for (query in best) {
            print best[query]
        }
    }
' ${BLAST_OUTPUT}.tmp | sort -t'q' -k2 -n > ${BLAST_OUTPUT}.sorted

# Add header
echo -e "query_id\tsubject_id\tpident\tqcov\tlength\tmismatches\tgap\tqstart\tqend\tsstart\tsend\tevalue\tbit_score" > ${BLAST_OUTPUT}
cat ${BLAST_OUTPUT}.sorted >> ${BLAST_OUTPUT}

# Clean up
rm queries.fasta ${BLAST_OUTPUT}.tmp ${BLAST_OUTPUT}.sorted ${DB_NAME}.*

echo "Analysis complete. Results are in ${BLAST_OUTPUT}"