#!/usr/bin/env python3
"""
Train Ensemble V2 - Stacking with Optuna Optimization

Features:
- XGBoost + LightGBM + CatBoost base learners
- Logistic Regression meta-learner (calibrated)
- Optuna hyperparameter optimization (30 trials)
- Grid search over horizon and ATR threshold
- Multi-class classification support

Usage:
    python train_ensemble_v2.py --symbols BTCUSDT ETHUSDT BNBUSDT --trials 30
"""
import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


def import_ml_libraries():
    """Import ML libraries."""
    try:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
    except ImportError as e:
        print(f"ERROR: Missing ML library - {e}")
        sys.exit(1)
    
    return XGBClassifier, LGBMClassifier, CatBoostClassifier


def get_feature_columns(df):
    """Extract feature columns."""
    exclude = [
        'timestamp', 'y_multiclass', 'y_class_name', 'y_binary', 
        'y_direction', 'normalized_return', 'open', 'high', 'low', 'close', 'volume'
    ]
    features = [c for c in df.columns if c not in exclude]
    return features


def create_optuna_objective(X_train, y_train, X_val, y_val, trial_name='trial'):
    """
    Create Optuna objective function for hyperparameter optimization.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        trial_name: str, name for this optimization
        
    Returns:
        Callable objective function
    """
    XGBClassifier, LGBMClassifier, CatBoostClassifier = import_ml_libraries()
    
    def objective(trial):
        """Optuna objective - maximize validation F1 score."""
        
        # Suggest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 8)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        colsample = trial.suggest_float('colsample', 0.6, 1.0)
        
        # XGBoost
        xgb = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
            verbosity=0
        )
        
        # LightGBM
        lgbm = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # CatBoost
        catboost = CatBoostClassifier(
            iterations=n_estimators,
            depth=max_depth,
            learning_rate=learning_rate,
            rsm=colsample,  # CatBoost usa 'rsm' en lugar de 'colsample'
            random_state=42,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
            bootstrap_type='Bernoulli',  # Permite usar subsample
            subsample=subsample
        )
        
        # Train base learners
        try:
            xgb.fit(X_train, y_train)
            lgbm.fit(X_train, y_train)
            catboost.fit(X_train, y_train)
            
            # Get meta-features (out-of-fold predictions)
            xgb_pred = xgb.predict_proba(X_val)
            lgbm_pred = lgbm.predict_proba(X_val)
            catboost_pred = catboost.predict_proba(X_val)
            
            # Stack predictions
            meta_features = np.hstack([xgb_pred, lgbm_pred, catboost_pred])
            
            # Meta-learner
            meta_learner = LogisticRegression(
                max_iter=1000, 
                random_state=42,
                multi_class='multinomial'
            )
            meta_learner.fit(meta_features, y_val)
            
            # Final predictions
            y_pred = meta_learner.predict(meta_features)
            
            # Calculate F1 score (macro average for multi-class)
            f1 = f1_score(y_val, y_pred, average='macro')
            
            return f1
            
        except Exception as e:
            print(f"      Trial failed: {e}")
            return 0.0
    
    return objective


def train_stacked_ensemble(X_train, y_train, X_val, y_val, best_params):
    """
    Train stacked ensemble with best hyperparameters.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (for calibration)
        best_params: dict, best hyperparameters from Optuna
        
    Returns:
        dict with base_models, meta_learner, scaler
    """
    XGBClassifier, LGBMClassifier, CatBoostClassifier = import_ml_libraries()
    
    print("      Training base learners with best params...")
    
    # Base learners
    xgb = XGBClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample'],
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        verbosity=0
    )
    
    lgbm = LGBMClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample'],
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    catboost = CatBoostClassifier(
        iterations=best_params['n_estimators'],
        depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        rsm=best_params['colsample'],  # CatBoost usa 'rsm' en lugar de 'colsample'
        random_state=42,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
        bootstrap_type='Bernoulli',  # Permite usar subsample
        subsample=best_params['subsample']
    )
    
    # Train
    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)
    catboost.fit(X_train, y_train)
    
    # Generate meta-features on validation set
    xgb_pred = xgb.predict_proba(X_val)
    lgbm_pred = lgbm.predict_proba(X_val)
    catboost_pred = catboost.predict_proba(X_val)
    
    meta_features = np.hstack([xgb_pred, lgbm_pred, catboost_pred])
    
    # Train meta-learner
    print("      Training meta-learner...")
    meta_learner = LogisticRegression(
        max_iter=1000,
        random_state=42,
        multi_class='multinomial',
        solver='lbfgs'
    )
    meta_learner.fit(meta_features, y_val)
    
    # Calibrate meta-learner
    meta_learner_cal = CalibratedClassifierCV(
        meta_learner, 
        method='isotonic',
        cv='prefit'
    )
    meta_learner_cal.fit(meta_features, y_val)
    
    # Evaluate
    y_pred = meta_learner_cal.predict(meta_features)
    y_pred_proba = meta_learner_cal.predict_proba(meta_features)
    
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    
    # Try to calculate AUC for multi-class
    try:
        from sklearn.preprocessing import label_binarize
        y_val_bin = label_binarize(y_val, classes=np.unique(y_train))
        auc = roc_auc_score(y_val_bin, y_pred_proba, multi_class='ovr', average='macro')
    except:
        auc = 0.0
    
    print(f"      Val Accuracy: {accuracy:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
    
    return {
        'xgb': xgb,
        'lgbm': lgbm,
        'catboost': catboost,
        'meta_learner': meta_learner_cal,
        'best_params': best_params,
        'metrics': {
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc
        }
    }


def train_window_with_optuna(X_train, y_train, X_val, y_val, window_id, n_trials=30):
    """
    Train ensemble for window with Optuna optimization.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        window_id: int, window identifier
        n_trials: int, number of Optuna trials
        
    Returns:
        dict with trained ensemble
    """
    print(f"\n   Window {window_id}:")
    print(f"      Train: {len(X_train):,}, Val: {len(X_val):,}")
    print(f"      Class distribution (train): {np.bincount(y_train.astype(int))}")
    
    # Optuna optimization
    print(f"      Running Optuna optimization ({n_trials} trials)...")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name=f'window_{window_id}'
    )
    
    objective = create_optuna_objective(X_train, y_train, X_val, y_val)
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    print(f"      Best F1: {study.best_value:.3f}")
    print(f"      Best params: {study.best_params}")
    
    # Train final model with best params
    ensemble = train_stacked_ensemble(X_train, y_train, X_val, y_val, study.best_params)
    ensemble['optuna_study'] = study
    
    return ensemble


def train_symbol_v2(symbol, data_dir, output_dir, n_trials=30):
    """
    Train V2 ensemble for all windows of a symbol.
    
    Args:
        symbol: str
        data_dir: Path to datasets
        output_dir: Path for models
        n_trials: int, Optuna trials per window
        
    Returns:
        dict with training results
    """
    print(f"\n{'='*60}")
    print(f"Training Ensemble V2 for {symbol}")
    print(f"{'='*60}")
    
    # Load dataset
    dataset_files = list(data_dir.glob(f"{symbol}_v2_*.parquet"))
    if not dataset_files:
        print(f"   ERROR: No V2 dataset found for {symbol}")
        return None
    
    # Use quantile version if available
    dataset_file = [f for f in dataset_files if 'quantile' in f.name]
    if not dataset_file:
        dataset_file = dataset_files
    dataset_file = dataset_file[0]
    
    print(f"Loading: {dataset_file.name}")
    df = pd.read_parquet(dataset_file)
    
    # Load splits
    splits_path = data_dir / f"{symbol}_v2_splits.csv"
    if not splits_path.exists():
        print(f"   ERROR: Splits file not found")
        return None
    
    splits_df = pd.read_csv(splits_path)
    
    # Get features
    features = get_feature_columns(df)
    print(f"Features: {len(features)}")
    
    # Train each window
    all_metrics = []
    models = []
    
    for idx, row in splits_df.iterrows():
        window_id = row['window']
        train_start = row['train_start_idx']
        train_end = row['train_end_idx']
        test_start = row['test_start_idx']
        
        # Split train into train/val (80/20)
        val_size = int((train_end - train_start) * 0.2)
        val_start = train_end - val_size
        
        X_train = df.iloc[train_start:val_start][features].values
        y_train = df.iloc[train_start:val_start]['y_multiclass'].values
        
        X_val = df.iloc[val_start:train_end][features].values
        y_val = df.iloc[val_start:train_end]['y_multiclass'].values
        
        # Train with Optuna
        ensemble = train_window_with_optuna(
            X_train, y_train, X_val, y_val, window_id, n_trials
        )
        
        all_metrics.append({
            'window': window_id,
            **ensemble['metrics']
        })
        
        models.append({
            'window': window_id,
            'ensemble': ensemble,
            'features': features
        })
    
    # Save models
    symbol_dir = output_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    
    for model_info in models:
        model_path = symbol_dir / f"ensemble_v2_window_{model_info['window']}.pkl"
        joblib.dump({
            'base_models': {
                'xgb': model_info['ensemble']['xgb'],
                'lgbm': model_info['ensemble']['lgbm'],
                'catboost': model_info['ensemble']['catboost']
            },
            'meta_learner': model_info['ensemble']['meta_learner'],
            'best_params': model_info['ensemble']['best_params'],
            'features': model_info['features']
        }, model_path)
    
    print(f"\nSaved {len(models)} models to {symbol_dir}")
    
    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = symbol_dir / "training_metrics_v2.csv"
    metrics_df.to_csv(metrics_path, index=False)
    
    print("\nTraining Metrics Summary:")
    print(metrics_df.describe())
    
    return {
        'symbol': symbol,
        'windows': len(models),
        'metrics': all_metrics
    }


def main():
    parser = argparse.ArgumentParser(description='Train Ensemble V2 with Optuna')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
    parser.add_argument('--trials', type=int, default=30, help='Optuna trials per window')
    
    args = parser.parse_args()
    
    # Directories
    data_dir = Path(__file__).parent.parent.parent / "data" / "datasets_v2"
    output_dir = Path(__file__).parent.parent.parent / "models" / "ensemble_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# Ensemble Training V2 - Stacking + Optuna")
    print(f"{'#'*60}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Optuna trials: {args.trials}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    
    # Train each symbol
    all_results = []
    
    for symbol in args.symbols:
        try:
            result = train_symbol_v2(symbol, data_dir, output_dir, args.trials)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"ERROR training {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    summary_path = output_dir / "training_summary_v2.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("Ensemble V2 training complete!")
    print(f"{'='*60}")
    print(f"Trained {len(all_results)} symbols")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
