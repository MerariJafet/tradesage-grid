#!/usr/bin/env python3
"""
Train ensemble model (XGBoost + LightGBM + CatBoost) with probability calibration.

Steps:
1. Load dataset splits from build_dataset_v1.py
2. Train base learners: XGBoost, LightGBM, CatBoost
3. Calibrate probabilities using CalibratedClassifierCV
4. Create soft voting ensemble
5. Save models per symbol and walk-forward window

Usage:
    python train_ensemble_v1.py --symbols BTCUSDT ETHUSDT BNBUSDT
"""
import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


def import_ml_libraries():
    """Import ML libraries with proper error handling."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("ERROR: xgboost not installed. Run: pip install xgboost")
        sys.exit(1)
    
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("ERROR: lightgbm not installed. Run: pip install lightgbm")
        sys.exit(1)
    
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("ERROR: catboost not installed. Run: pip install catboost")
        sys.exit(1)
    
    return XGBClassifier, LGBMClassifier, CatBoostClassifier


def get_feature_columns(df):
    """Extract feature column names."""
    exclude = ['timestamp', 'y', 'y_sign', 'future_return', 'open', 'high', 'low', 'close', 'volume']
    features = [c for c in df.columns if c not in exclude]
    return features


def create_base_models():
    """Create base models with conservative hyperparameters to avoid overfitting."""
    XGBClassifier, LGBMClassifier, CatBoostClassifier = import_ml_libraries()
    
    # XGBoost: Conservative settings
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    # LightGBM: Fast and regularized
    lgbm = LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # CatBoost: Silent and robust
    catboost = CatBoostClassifier(
        iterations=100,
        depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_data_in_leaf=20,
        random_state=42,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False
    )
    
    return xgb, lgbm, catboost


def train_ensemble_for_window(X_train, y_train, X_val, y_val, window_id):
    """
    Train ensemble for a single walk-forward window.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data for calibration
        window_id: int, window identifier
        
    Returns:
        dict: Trained models and metrics
    """
    print(f"\n   Window {window_id}:")
    print(f"      Train samples: {len(X_train):,}, Positive: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
    print(f"      Val samples: {len(X_val):,}, Positive: {y_val.sum():,} ({y_val.mean()*100:.1f}%)")
    
    # Create base models
    xgb, lgbm, catboost = create_base_models()
    
    # Train base models
    print("      Training base models...")
    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)
    catboost.fit(X_train, y_train)
    
    # Calibrate probabilities using validation set
    print("      Calibrating probabilities...")
    xgb_cal = CalibratedClassifierCV(xgb, cv='prefit', method='isotonic')
    lgbm_cal = CalibratedClassifierCV(lgbm, cv='prefit', method='isotonic')
    catboost_cal = CalibratedClassifierCV(catboost, cv='prefit', method='isotonic')
    
    xgb_cal.fit(X_val, y_val)
    lgbm_cal.fit(X_val, y_val)
    catboost_cal.fit(X_val, y_val)
    
    # Create voting ensemble with calibrated models
    # Note: VotingClassifier doesn't need fitting when estimators are already fitted
    # We'll manually combine predictions instead
    
    # Evaluate on validation set (using soft voting)
    y_pred_xgb = xgb_cal.predict_proba(X_val)[:, 1]
    y_pred_lgbm = lgbm_cal.predict_proba(X_val)[:, 1]
    y_pred_catboost = catboost_cal.predict_proba(X_val)[:, 1]
    
    y_pred_proba = (y_pred_xgb + y_pred_lgbm + y_pred_catboost) / 3
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        'window': window_id,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'val_auc': roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5,
        'val_precision': precision_score(y_val, y_pred, zero_division=0),
        'val_recall': recall_score(y_val, y_pred, zero_division=0)
    }
    
    print(f"      Validation AUC: {metrics['val_auc']:.3f}, "
          f"Precision: {metrics['val_precision']:.3f}, "
          f"Recall: {metrics['val_recall']:.3f}")
    
    # Return ensemble as a dictionary with models
    ensemble = {
        'xgb': xgb_cal,
        'lgbm': lgbm_cal,
        'catboost': catboost_cal
    }
    
    return {
        'ensemble': ensemble,
        'metrics': metrics
    }


def train_symbol(symbol, data_dir, output_dir):
    """
    Train ensemble models for all walk-forward windows of a symbol.
    
    Args:
        symbol: str, e.g., 'BTCUSDT'
        data_dir: Path, directory with parquet files
        output_dir: Path, output directory for models
        
    Returns:
        dict: Training metrics for all windows
    """
    print(f"\n{'='*60}")
    print(f"Training ensemble for {symbol}")
    print(f"{'='*60}")
    
    # Load full dataset
    full_path = data_dir / f"{symbol}_full.parquet"
    if not full_path.exists():
        print(f"   ERROR: Dataset not found: {full_path}")
        return None
    
    df = pd.read_parquet(full_path)
    print(f"Loaded dataset: {len(df):,} rows")
    
    # Load splits metadata
    splits_path = data_dir / f"{symbol}_splits.csv"
    splits_df = pd.read_csv(splits_path)
    print(f"Loaded {len(splits_df)} walk-forward splits")
    
    # Get features
    features = get_feature_columns(df)
    print(f"Features: {len(features)}")
    
    # Train for each window
    all_metrics = []
    models = []
    
    for idx, row in splits_df.iterrows():
        window_id = row['window']
        train_start = row['train_start_idx']
        train_end = row['train_end_idx']
        test_start = row['test_start_idx']
        test_end = row['test_end_idx']
        
        # Split train into train/val for calibration (80/20)
        val_size = int((train_end - train_start) * 0.2)
        val_start = train_end - val_size
        
        X_train = df.iloc[train_start:val_start][features].values
        y_train = df.iloc[train_start:val_start]['y'].values
        
        X_val = df.iloc[val_start:train_end][features].values
        y_val = df.iloc[val_start:train_end]['y'].values
        
        # Train ensemble
        result = train_ensemble_for_window(X_train, y_train, X_val, y_val, window_id)
        
        all_metrics.append(result['metrics'])
        models.append({
            'window': window_id,
            'ensemble': result['ensemble'],
            'features': features
        })
    
    # Save models
    symbol_dir = output_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    
    for model_info in models:
        model_path = symbol_dir / f"ensemble_window_{model_info['window']}.pkl"
        joblib.dump({
            'ensemble': model_info['ensemble'],
            'features': model_info['features']
        }, model_path)
    
    print(f"\nSaved {len(models)} models to {symbol_dir}")
    
    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = symbol_dir / "training_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    
    print("\nTraining Metrics Summary:")
    print(metrics_df.describe())
    
    return {
        'symbol': symbol,
        'windows': len(models),
        'metrics': all_metrics
    }


def main():
    parser = argparse.ArgumentParser(description='Train ensemble models with calibration')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
                        help='Symbols to train')
    
    args = parser.parse_args()
    
    # Directories
    data_dir = Path(__file__).parent.parent.parent / "data" / "datasets"
    output_dir = Path(__file__).parent.parent.parent / "models" / "ensemble_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# Ensemble Training v1 - XGB/LGBM/CatBoost + Calibration")
    print(f"{'#'*60}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    
    # Train each symbol
    all_results = []
    
    for symbol in args.symbols:
        try:
            result = train_symbol(symbol, data_dir, output_dir)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"ERROR training {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save overall summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("Ensemble training complete!")
    print(f"{'='*60}")
    print(f"Trained {len(all_results)} symbols")
    print(f"Models saved to: {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
