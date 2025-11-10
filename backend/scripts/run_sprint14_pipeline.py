#!/usr/bin/env python3
"""
Sprint 14 Pipeline - Complete Execution

Runs entire Sprint 14 workflow:
1. Build datasets V2 (with real microstructure)
2. Train ensemble V2 (with Optuna)
3. Evaluate walk-forward V2
4. Generate comprehensive report

Usage:
    python run_sprint14_pipeline.py --symbols BTCUSDT --quick
    
    --quick: Use 5 Optuna trials (fast testing)
    --full: Use 30 Optuna trials (production)
"""
import argparse
import sys
import subprocess
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run shell command and print output."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed with code {result.returncode}")
        return False
    
    print(f"\n✅ {description} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run Sprint 14 complete pipeline')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
    parser.add_argument('--quick', action='store_true', help='Quick mode (5 trials)')
    parser.add_argument('--full', action='store_true', help='Full mode (30 trials)')
    parser.add_argument('--skip-build', action='store_true', help='Skip dataset building')
    parser.add_argument('--skip-train', action='store_true', help='Skip training')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation')
    
    args = parser.parse_args()
    
    # Determine trial count
    if args.quick:
        n_trials = 5
        mode_name = "QUICK"
    elif args.full:
        n_trials = 30
        mode_name = "FULL"
    else:
        n_trials = 10
        mode_name = "STANDARD"
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    python_exe = sys.executable
    scripts_dir = base_dir / "backend" / "scripts"
    
    symbols_str = ' '.join(args.symbols)
    
    print(f"\n{'#'*60}")
    print(f"# SPRINT 14 PIPELINE - {mode_name} MODE")
    print(f"{'#'*60}")
    print(f"Symbols: {symbols_str}")
    print(f"Optuna trials: {n_trials}")
    print(f"Start time: {datetime.now()}")
    
    # Step 1: Build datasets
    if not args.skip_build:
        cmd = [
            python_exe,
            str(scripts_dir / "build_dataset_v2.py"),
            "--symbols", *args.symbols,
            "--span", "12m",
            "--horizon", "60",
            "--mode", "quantile"
        ]
        
        if not run_command(cmd, "Step 1: Build Datasets V2"):
            return 1
    else:
        print("\n⏭ Skipping dataset building")
    
    # Step 2: Train ensemble
    if not args.skip_train:
        cmd = [
            python_exe,
            str(scripts_dir / "train_ensemble_v2.py"),
            "--symbols", *args.symbols,
            "--trials", str(n_trials)
        ]
        
        if not run_command(cmd, "Step 2: Train Ensemble V2 with Optuna"):
            return 1
    else:
        print("\n⏭ Skipping training")
    
    # Step 3: Evaluate walk-forward
    if not args.skip_eval:
        cmd = [
            python_exe,
            str(scripts_dir / "eval_walk_forward_v2.py"),
            "--symbols", *args.symbols,
            "--latency", "1bar",
            "--slippage", "3bps"
        ]
        
        if not run_command(cmd, "Step 3: Evaluate Walk-Forward V2"):
            return 1
    else:
        print("\n⏭ Skipping evaluation")
    
    # Done
    print(f"\n{'#'*60}")
    print(f"# SPRINT 14 PIPELINE COMPLETE")
    print(f"{'#'*60}")
    print(f"End time: {datetime.now()}")
    print(f"\nResults:")
    print(f"  - QA Metrics: qa_metrics_wf_v2.json")
    print(f"  - Buckets: buckets_wf_v2.json")
    print(f"  - PnL Log: pnl_log_wf_v2.txt")
    print(f"  - Models: models/ensemble_v2/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
