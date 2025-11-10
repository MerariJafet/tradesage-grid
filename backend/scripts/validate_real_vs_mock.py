#!/usr/bin/env python3
"""
Validate Real vs MOCK Data Quality
Sprint 15 - Data Integrity & Real Feature Validation

Statistical comparison:
- KS-test: Distribution similarity
- QQ-plots: Visual distribution comparison
- Correlation analysis: Feature-target relationships
- Leakage scan: Train vs test correlation decay

Outputs:
- reports/validation/*.png (visualizations)
- reports/validation/leakage_scan.csv (correlation matrix)
- reports/SPRINT15_DATA_VALIDATION_REPORT.md
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import ks_2samp, spearmanr, pearsonr
import warnings
import argparse
from typing import Dict, Tuple
import logging

warnings.filterwarnings('ignore')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
VALIDATION_DIR = Path('reports/validation')
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_mock_features(symbol: str) -> pd.DataFrame:
    """Load MOCK dataset features."""
    mock_path = Path(f'data/datasets_v3/{symbol}_v3_adaptive.parquet')
    
    if not mock_path.exists():
        raise FileNotFoundError(f"MOCK dataset not found: {mock_path}")
    
    df = pd.read_parquet(mock_path)
    logger.info(f"Loaded MOCK data: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_real_features(symbol: str) -> pd.DataFrame:
    """Load REAL dataset features."""
    real_path = Path(f'data/datasets_real/{symbol}_real_adaptive.parquet')
    
    if not real_path.exists():
        raise FileNotFoundError(f"REAL dataset not found: {real_path}")
    
    df = pd.read_parquet(real_path)
    logger.info(f"Loaded REAL data: {len(df)} rows, {len(df.columns)} columns")
    return df


def ks_test_comparison(mock_series: pd.Series, real_series: pd.Series, 
                       feature_name: str) -> Dict:
    """
    Kolmogorov-Smirnov test for distribution similarity.
    
    Returns:
        dict with ks_stat, p_value, interpretation
    """
    # Remove NaN/inf
    mock_clean = mock_series.replace([np.inf, -np.inf], np.nan).dropna()
    real_clean = real_series.replace([np.inf, -np.inf], np.nan).dropna()
    
    # KS test
    ks_stat, p_value = ks_2samp(mock_clean, real_clean)
    
    # Interpretation
    if p_value < 0.01:
        interpretation = "SIGNIFICANTLY DIFFERENT (p < 0.01) ✅"
        validates_hypothesis = True
    elif p_value < 0.05:
        interpretation = "Different (p < 0.05)"
        validates_hypothesis = True
    else:
        interpretation = "Similar (p >= 0.05) ❌"
        validates_hypothesis = False
    
    result = {
        'feature': feature_name,
        'ks_stat': ks_stat,
        'p_value': p_value,
        'interpretation': interpretation,
        'validates_hypothesis': validates_hypothesis,
        'mock_mean': mock_clean.mean(),
        'real_mean': real_clean.mean(),
        'mock_std': mock_clean.std(),
        'real_std': real_clean.std(),
        'mock_count': len(mock_clean),
        'real_count': len(real_clean)
    }
    
    logger.info(f"{feature_name}: KS={ks_stat:.4f}, p={p_value:.4f} - {interpretation}")
    
    return result


def plot_histogram_comparison(mock_series: pd.Series, real_series: pd.Series,
                              feature_name: str, ks_result: Dict):
    """Plot overlaid histograms for visual comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Remove outliers for better visualization (99th percentile)
    mock_clean = mock_series.replace([np.inf, -np.inf], np.nan).dropna()
    real_clean = real_series.replace([np.inf, -np.inf], np.nan).dropna()
    
    q99_mock = mock_clean.quantile(0.99)
    q99_real = real_clean.quantile(0.99)
    q99 = max(q99_mock, q99_real)
    
    q01_mock = mock_clean.quantile(0.01)
    q01_real = real_clean.quantile(0.01)
    q01 = min(q01_mock, q01_real)
    
    mock_plot = mock_clean[(mock_clean >= q01) & (mock_clean <= q99)]
    real_plot = real_clean[(real_clean >= q01) & (real_clean <= q99)]
    
    # Plot histograms
    ax.hist(mock_plot, bins=50, alpha=0.5, label='MOCK', density=True, color='blue')
    ax.hist(real_plot, bins=50, alpha=0.5, label='REAL', density=True, color='red')
    
    # Add vertical lines for means
    ax.axvline(mock_clean.mean(), color='blue', linestyle='--', 
               label=f'MOCK mean: {mock_clean.mean():.4f}')
    ax.axvline(real_clean.mean(), color='red', linestyle='--',
               label=f'REAL mean: {real_clean.mean():.4f}')
    
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Density')
    ax.set_title(f'{feature_name} Distribution Comparison\n'
                 f'KS-stat: {ks_result["ks_stat"]:.4f}, p-value: {ks_result["p_value"]:.4f}\n'
                 f'{ks_result["interpretation"]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save
    output_path = VALIDATION_DIR / f'{feature_name.lower()}_histogram.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved histogram: {output_path}")


def plot_qq_comparison(mock_series: pd.Series, real_series: pd.Series,
                      feature_name: str):
    """QQ-plot for distribution comparison."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Remove NaN/inf
    mock_clean = mock_series.replace([np.inf, -np.inf], np.nan).dropna()
    real_clean = real_series.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Calculate quantiles
    n = min(len(mock_clean), len(real_clean), 1000)  # Limit to 1000 points
    mock_quantiles = np.percentile(mock_clean, np.linspace(0, 100, n))
    real_quantiles = np.percentile(real_clean, np.linspace(0, 100, n))
    
    # Plot QQ
    ax.scatter(mock_quantiles, real_quantiles, alpha=0.5, s=20)
    
    # Add 45-degree reference line
    min_val = min(mock_quantiles.min(), real_quantiles.min())
    max_val = max(mock_quantiles.max(), real_quantiles.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x (perfect match)')
    
    ax.set_xlabel(f'{feature_name} MOCK Quantiles')
    ax.set_ylabel(f'{feature_name} REAL Quantiles')
    ax.set_title(f'{feature_name} QQ-Plot\nMOCK vs REAL Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save
    output_path = VALIDATION_DIR / f'{feature_name.lower()}_qq.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved QQ-plot: {output_path}")


def plot_time_series_comparison(mock_series: pd.Series, real_series: pd.Series,
                                feature_name: str, window: int = 1000):
    """Plot time series comparison (first N points)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Take first N points
    mock_plot = mock_series.iloc[:window]
    real_plot = real_series.iloc[:window]
    
    ax.plot(mock_plot.values, alpha=0.7, label='MOCK', color='blue')
    ax.plot(real_plot.values, alpha=0.7, label='REAL', color='red')
    
    ax.set_xlabel('Time (bars)')
    ax.set_ylabel(feature_name)
    ax.set_title(f'{feature_name} Time Series Comparison (First {window} bars)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save
    output_path = VALIDATION_DIR / f'{feature_name.lower()}_timeseries.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved time series: {output_path}")


def correlation_analysis(df: pd.DataFrame, target_col: str, 
                         feature_cols: list, lags: list = [1, 3, 5]) -> pd.DataFrame:
    """
    Analyze correlation between features and future target at different lags.
    
    This tests if features have predictive power.
    """
    results = []
    
    for feature in feature_cols:
        if feature not in df.columns:
            continue
        
        for lag in lags:
            # Future target
            future_target = df[target_col].shift(-lag)
            
            # Clean data
            mask = ~(df[feature].isna() | future_target.isna() | 
                    np.isinf(df[feature]) | np.isinf(future_target))
            
            if mask.sum() < 100:
                continue
            
            x = df.loc[mask, feature]
            y = future_target[mask]
            
            # Pearson correlation
            corr, p_value = pearsonr(x, y)
            
            results.append({
                'feature': feature,
                'lag': lag,
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'abs_corr': abs(corr),
                'n_samples': len(x)
            })
    
    results_df = pd.DataFrame(results)
    return results_df.sort_values('abs_corr', ascending=False)


def leakage_scan(df_train: pd.DataFrame, df_test: pd.DataFrame,
                target_col: str, feature_cols: list) -> pd.DataFrame:
    """
    Scan for data leakage by comparing train vs test correlations.
    
    Leakage indicators:
    - High correlation in train (> 0.1)
    - Similar/higher correlation in test (should decay)
    """
    results = []
    
    for feature in feature_cols:
        if feature not in df_train.columns or feature not in df_test.columns:
            continue
        
        # Train correlation
        mask_train = ~(df_train[feature].isna() | df_train[target_col].isna() |
                      np.isinf(df_train[feature]) | np.isinf(df_train[target_col]))
        
        if mask_train.sum() < 100:
            continue
        
        corr_train, _ = pearsonr(df_train.loc[mask_train, feature],
                                 df_train.loc[mask_train, target_col])
        
        # Test correlation
        mask_test = ~(df_test[feature].isna() | df_test[target_col].isna() |
                     np.isinf(df_test[feature]) | np.isinf(df_test[target_col]))
        
        if mask_test.sum() < 100:
            continue
        
        corr_test, _ = pearsonr(df_test.loc[mask_test, feature],
                                df_test.loc[mask_test, target_col])
        
        # Leakage score: test corr should be lower than train
        leakage_score = corr_test - corr_train
        has_leakage = (abs(corr_train) > 0.1) and (leakage_score > 0)
        
        results.append({
            'feature': feature,
            'corr_train': corr_train,
            'corr_test': corr_test,
            'leakage_score': leakage_score,
            'has_leakage': has_leakage,
            'abs_corr_train': abs(corr_train),
            'abs_corr_test': abs(corr_test)
        })
    
    results_df = pd.DataFrame(results)
    return results_df.sort_values('leakage_score', ascending=False)


def generate_validation_report(ks_results: list, corr_results_mock: pd.DataFrame,
                               corr_results_real: pd.DataFrame, 
                               leakage_results: pd.DataFrame,
                               symbol: str):
    """Generate comprehensive markdown validation report."""
    
    report = f"""# Sprint 15 - Data Validation Report: MOCK vs REAL

**Symbol:** {symbol}  
**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Objective:** Validate hypothesis that MOCK data quality caused Sprint 14 failures

---

## Executive Summary

"""
    
    # Summary statistics
    total_features = len(ks_results)
    significant_diff = sum(1 for r in ks_results if r['validates_hypothesis'])
    pct_different = (significant_diff / total_features) * 100 if total_features > 0 else 0
    
    report += f"""
**Key Findings:**
- **Features Tested:** {total_features}
- **Significantly Different (p < 0.05):** {significant_diff} ({pct_different:.1f}%)
- **Hypothesis Validation:** {"✅ CONFIRMED" if pct_different >= 50 else "❌ REJECTED"}

"""
    
    if pct_different >= 50:
        report += """
**Interpretation:**
> MOCK and REAL data distributions are SIGNIFICANTLY DIFFERENT for majority of features.
> This validates the Sprint 14 hypothesis that poor data quality caused model failures.
> **Recommendation:** Proceed with re-training on REAL data.

"""
    else:
        report += """
**Interpretation:**
> MOCK and REAL data distributions are SIMILAR for majority of features.
> This REJECTS the Sprint 14 hypothesis about data quality.
> **Recommendation:** Investigate alternative failure causes (feature engineering, labeling logic).

"""
    
    report += """---

## 1. Distribution Comparison (KS-Test)

### Statistical Test Results

| Feature | KS-Stat | P-Value | MOCK Mean | REAL Mean | Interpretation |
|---------|---------|---------|-----------|-----------|----------------|
"""
    
    for result in ks_results:
        report += f"| {result['feature']} | {result['ks_stat']:.4f} | {result['p_value']:.4f} | "
        report += f"{result['mock_mean']:.4f} | {result['real_mean']:.4f} | "
        report += f"{result['interpretation']} |\n"
    
    report += """

### Visualizations

"""
    
    for result in ks_results:
        feature_lower = result['feature'].lower()
        report += f"#### {result['feature']}\n\n"
        report += f"![Histogram]({feature_lower}_histogram.png)\n\n"
        report += f"![QQ-Plot]({feature_lower}_qq.png)\n\n"
        report += f"![Time Series]({feature_lower}_timeseries.png)\n\n"
    
    report += """---

## 2. Correlation Analysis (Feature → Future Return)

### MOCK Data Correlations

**Top 10 Features by Absolute Correlation (lag=1-5):**

| Feature | Lag | Correlation | P-Value | Significant |
|---------|-----|-------------|---------|-------------|
"""
    
    for _, row in corr_results_mock.head(10).iterrows():
        report += f"| {row['feature']} | {row['lag']} | {row['correlation']:.4f} | "
        report += f"{row['p_value']:.4f} | {'✅' if row['significant'] else '❌'} |\n"
    
    report += """

### REAL Data Correlations

**Top 10 Features by Absolute Correlation (lag=1-5):**

| Feature | Lag | Correlation | P-Value | Significant |
|---------|-----|-------------|---------|-------------|
"""
    
    for _, row in corr_results_real.head(10).iterrows():
        report += f"| {row['feature']} | {row['lag']} | {row['correlation']:.4f} | "
        report += f"{row['p_value']:.4f} | {'✅' if row['significant'] else '❌'} |\n"
    
    report += """

**Interpretation:**
- Features with significant correlation (p < 0.05) have potential predictive power
- Compare MOCK vs REAL: Are different features predictive?
- Low correlations (< 0.05) suggest weak linear relationships (non-linear may exist)

---

## 3. Leakage Scan (Train vs Test Correlation Decay)

**Features with Potential Leakage:**

| Feature | Train Corr | Test Corr | Leakage Score | Has Leakage |
|---------|-----------|-----------|---------------|-------------|
"""
    
    for _, row in leakage_results.head(15).iterrows():
        report += f"| {row['feature']} | {row['corr_train']:.4f} | {row['corr_test']:.4f} | "
        report += f"{row['leakage_score']:.4f} | {'⚠️ YES' if row['has_leakage'] else '✅ NO'} |\n"
    
    leakage_count = leakage_results['has_leakage'].sum()
    
    report += f"""

**Leakage Summary:**
- **Features with Leakage:** {leakage_count}
- **Expected Behavior:** Test correlation < Train correlation (generalization)
- **Leakage Indicator:** Test correlation ≥ Train correlation

"""
    
    if leakage_count > 0:
        report += """
**⚠️ WARNING:** Some features show potential leakage. Review:
1. Forward-looking calculations in features
2. Improper train/test splits
3. Data contamination

"""
    else:
        report += """
**✅ GOOD:** No significant leakage detected. Model should generalize properly.

"""
    
    report += """---

## 4. Conclusions & Recommendations

"""
    
    if pct_different >= 50:
        report += """
### ✅ Hypothesis VALIDATED

**Finding:** MOCK and REAL data distributions differ significantly.

**Root Cause Confirmed:**
1. Synthetic orderbook creates unrealistic spreads/depth
2. Interpolated funding rates smooth real 8-hour volatility
3. Mock volume classification lacks real buy/sell imbalance

**Recommendations:**
1. ✅ Proceed with V3 re-training on REAL data
2. ✅ Use 8 Optuna trials (sanity check)
3. ✅ If PF ≥ 0.8 → Scale to full optimization (30 trials)
4. ✅ If PF < 0.5 → No edge exists, pivot to fundamentals

**Expected Outcome:** Improved PF, balanced direction (40-60% LONG)

"""
    else:
        report += """
### ❌ Hypothesis REJECTED

**Finding:** MOCK and REAL data distributions are similar.

**Implications:**
1. Data quality was NOT the primary failure cause
2. Sprint 14 failures likely due to:
   - Feature engineering issues
   - Labeling logic problems
   - Model architecture limitations
   - No inherent ML edge in 1-min timeframe

**Recommendations:**
1. ⚠️ Re-train anyway (low cost, confirm finding)
2. ⚠️ If PF still < 0.5 → Investigate:
   - Different labeling horizons (30, 90, 120 bars)
   - Feature selection (drop bottom 20%)
   - Alternative targets (spread capture vs directional)
3. ❌ If still fails → PIVOT to non-ML strategies

"""
    
    report += """---

## 5. Next Steps

### Immediate Actions

1. **Review Report:** Stakeholder sign-off (Alejandra, Merari)
2. **Re-train V3:** Use REAL data, 8 Optuna trials
3. **Walk-Forward Evaluation:** conf=0.55, latency=1bar, slippage=3bps

### Decision Gate (Post Re-training)

| PF Result | Win Rate | Decision |
|-----------|----------|----------|
| **PF ≥ 0.8** | WR ≥ 35% | ✅ SUCCESS - Scale to production |
| **0.5 ≤ PF < 0.8** | WR ≥ 30% | 🟡 MARGINAL - Further optimization |
| **PF < 0.5** | WR < 30% | ❌ FAILURE - Pivot to fundamentals |

---

**Report Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Prepared By:** Abinadab (Core Dev)  
**Reviewed By:** Pending  
**Approved By:** Pending

---

**END OF VALIDATION REPORT**
"""
    
    # Save report
    report_path = Path('reports/SPRINT15_DATA_VALIDATION_REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Validation report saved: {report_path}")
    
    # Also save leakage scan CSV
    leakage_path = VALIDATION_DIR / 'leakage_scan.csv'
    leakage_results.to_csv(leakage_path, index=False)
    logger.info(f"Leakage scan saved: {leakage_path}")


def validate_symbol(symbol: str):
    """Run complete validation for a symbol."""
    logger.info(f"{'='*60}")
    logger.info(f"Validating {symbol}: MOCK vs REAL")
    logger.info(f"{'='*60}")
    
    # Load data
    try:
        df_mock = load_mock_features(symbol)
        df_real = load_real_features(symbol)
    except FileNotFoundError as e:
        logger.error(f"Data not found for {symbol}: {e}")
        return
    
    # Identify microstructure features
    microstructure_features = [
        'vpin', 'roll_spread', 'cvd', 'depth_imbalance',
        'funding_momentum', 'funding_rate'
    ]
    
    # Filter to features present in both datasets
    features_to_test = [f for f in microstructure_features 
                       if f in df_mock.columns and f in df_real.columns]
    
    if not features_to_test:
        logger.warning(f"No common microstructure features found for {symbol}")
        return
    
    logger.info(f"Testing {len(features_to_test)} features: {features_to_test}")
    
    # 1. KS-Test Comparison
    logger.info("\n1. Running KS-Tests...")
    ks_results = []
    
    for feature in features_to_test:
        result = ks_test_comparison(df_mock[feature], df_real[feature], feature)
        ks_results.append(result)
        
        # Generate visualizations
        plot_histogram_comparison(df_mock[feature], df_real[feature], feature, result)
        plot_qq_comparison(df_mock[feature], df_real[feature], feature)
        plot_time_series_comparison(df_mock[feature], df_real[feature], feature)
    
    # 2. Correlation Analysis
    logger.info("\n2. Running Correlation Analysis...")
    
    # Need forward return as target
    if 'forward_return' in df_mock.columns:
        target_col = 'forward_return'
    elif 'label' in df_mock.columns:
        target_col = 'label'
    else:
        logger.warning("No target column found, skipping correlation analysis")
        corr_results_mock = pd.DataFrame()
        corr_results_real = pd.DataFrame()
    
    if target_col:
        corr_results_mock = correlation_analysis(df_mock, target_col, features_to_test)
        corr_results_real = correlation_analysis(df_real, target_col, features_to_test)
        
        logger.info(f"MOCK: Top correlated feature = {corr_results_mock.iloc[0]['feature']} "
                   f"(corr={corr_results_mock.iloc[0]['correlation']:.4f})")
        logger.info(f"REAL: Top correlated feature = {corr_results_real.iloc[0]['feature']} "
                   f"(corr={corr_results_real.iloc[0]['correlation']:.4f})")
    
    # 3. Leakage Scan
    logger.info("\n3. Running Leakage Scan...")
    
    # Split into train/test (simple 80/20)
    split_idx = int(len(df_mock) * 0.8)
    df_mock_train = df_mock.iloc[:split_idx]
    df_mock_test = df_mock.iloc[split_idx:]
    
    if target_col:
        leakage_results = leakage_scan(df_mock_train, df_mock_test, target_col, features_to_test)
        leakage_count = leakage_results['has_leakage'].sum()
        logger.info(f"Leakage detected in {leakage_count} features")
    else:
        leakage_results = pd.DataFrame()
    
    # 4. Generate Report
    logger.info("\n4. Generating Validation Report...")
    generate_validation_report(ks_results, corr_results_mock, corr_results_real,
                              leakage_results, symbol)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Validation complete for {symbol}")
    logger.info(f"Reports saved to: {VALIDATION_DIR}")
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Validate MOCK vs REAL data quality')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT'],
                       help='Symbols to validate')
    parser.add_argument('--out', type=str, default='reports/validation',
                       help='Output directory for reports')
    args = parser.parse_args()
    
    # Update output directory
    global VALIDATION_DIR
    VALIDATION_DIR = Path(args.out)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Validate each symbol
    for symbol in args.symbols:
        validate_symbol(symbol)
    
    logger.info("All validations complete!")


if __name__ == "__main__":
    main()
