#!/bin/bash
# Meta Evaluation Inference Shell Script
# Runs metajudge_infer.py for checklist matching evaluation

# ============================================================================
# Configuration
# ============================================================================

# Input/Output files
INPUT_FILE="${INPUT_FILE:-./helpsteer3_test_1000.jsonl}"

# Model being evaluated (used to find corresponding checklist field, e.g., gpt-5-2025-08-07-critic)
MODEL_BE_EVALUATED="${MODEL_BE_EVALUATED:-model-low_deceptive_alignment}"

# Evaluator model configuration
API_KEY="${API_KEY:-${OPENAI_API_KEY}}"
API_BASE="${API_BASE:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
EVAL_MODEL="${EVAL_MODEL:-gpt-4o}"

# Output file path
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_BE_EVALUATED}-${EVAL_MODEL}-meta_eval.jsonl"

# Generation parameters
MAX_TOKENS=8192
TEMPERATURE=0.6
MAX_RETRIES=5
CONCURRENT_REQUESTS=50
RATE_LIMIT_DELAY=1.0
BATCH_SIZE=100

# ============================================================================
# Run Inference
# ============================================================================

echo "=== Meta Evaluation Inference ==="
echo "Input file: $INPUT_FILE"
echo "Model being evaluated: $MODEL_BE_EVALUATED"
echo "Evaluator model: $EVAL_MODEL"
echo "Output file: $OUTPUT_FILE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run inference
python "$SCRIPT_DIR/metajudge_infer.py" \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --model-be-evaluated "$MODEL_BE_EVALUATED" \
    --api-key "$API_KEY" \
    --api-base "$API_BASE" \
    --model "$EVAL_MODEL" \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --max-retries $MAX_RETRIES \
    --concurrent-requests $CONCURRENT_REQUESTS \
    --rate-limit-delay $RATE_LIMIT_DELAY \
    --batch-size $BATCH_SIZE \
    --log-file "${OUTPUT_DIR}/meta_eval_${MODEL_BE_EVALUATED}.log"

echo ""
echo "=== Inference Complete ==="
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "Next, run analysis script to compute Recall/Precision:"
echo "  python metajudge_analysis.py --input-file $OUTPUT_FILE --model-be-evaluated $MODEL_BE_EVALUATED"
