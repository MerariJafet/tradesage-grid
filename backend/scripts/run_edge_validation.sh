#!/bin/bash
# Edge Validation Pipeline
# Ejecuta todo el flujo de validación de Edge
# 
# Usage: ./run_edge_validation.sh

set -e

WORKSPACE="/Users/merari/Desktop/bot de scalping"
DATA_PATH="$WORKSPACE/data/real_binance_ws_pilot"
OUTPUT_DIR="$WORKSPACE/data/edge_validation"
PYTHON="$WORKSPACE/.venv/bin/python"

echo "========================================="
echo "EDGE VALIDATION PIPELINE"
echo "========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Symbols to process
SYMBOLS=("BTCUSDT" "ETHUSDT" "BNBUSDT")

for SYMBOL in "${SYMBOLS[@]}"; do
    echo ""
    echo "========================================="
    echo "Processing $SYMBOL"
    echo "========================================="
    echo ""
    
    # Step 1: Compute features
    echo "[1/2] Computing features..."
    "$PYTHON" backend/scripts/compute_features.py \
        --symbol "$SYMBOL" \
        --data-path "$DATA_PATH" \
        --output "$OUTPUT_DIR/${SYMBOL}_features.parquet"
    
    echo ""
    
    # Step 2: Validate edge
    echo "[2/2] Validating edge..."
    "$PYTHON" backend/scripts/edge_probe.py \
        --features "$OUTPUT_DIR/${SYMBOL}_features.parquet" \
        --symbol "$SYMBOL" \
        --output "$OUTPUT_DIR/${SYMBOL}_edge_results.json"
    
    echo ""
    echo "✓ $SYMBOL completed"
    echo ""
done

# Generate consolidated report
echo ""
echo "========================================="
echo "Generating consolidated report..."
echo "========================================="

"$PYTHON" -c "
import json
import pandas as pd
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

print()
print('='*70)
print('EDGE VALIDATION - CONSOLIDATED RESULTS')
print('='*70)
print()

decisions = []

for symbol in symbols:
    result_file = output_dir / f'{symbol}_edge_results.json'
    
    if result_file.exists():
        with open(result_file) as f:
            results = json.load(f)
        
        verdict = results.get('verdict', {})
        
        print(f'{symbol}:')
        print(f'  Decision: {verdict.get(\"decision\", \"N/A\")}')
        print(f'  Confidence: {verdict.get(\"confidence\", \"N/A\")}')
        print(f'  Signals: {verdict.get(\"signals\", 0)}/4')
        print(f'  Reason: {verdict.get(\"reason\", \"N/A\")}')
        print()
        
        decisions.append(verdict.get('decision', 'NO-GO'))

# Overall decision
go_count = sum(1 for d in decisions if 'GO' in d and 'NO-GO' not in d)
print('='*70)
print(f'GO votes: {go_count}/{len(decisions)}')
print()

if go_count >= 2:
    print('OVERALL DECISION: ✅ GO')
    print('Recommendation: Proceed with 90-day collection')
elif go_count == 1:
    print('OVERALL DECISION: 🟡 CONDITIONAL')
    print('Recommendation: Review individual results, consider extended pilot')
else:
    print('OVERALL DECISION: ❌ NO-GO')
    print('Recommendation: Do not proceed with 90-day collection')
    print('Action: Review feature engineering or pivot to alternative approach')

print('='*70)
"

echo ""
echo "========================================="
echo "EDGE VALIDATION COMPLETE"
echo "========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files generated:"
echo "  - {SYMBOL}_features.parquet (features datasets)"
echo "  - {SYMBOL}_edge_results.json (validation results)"
echo ""
