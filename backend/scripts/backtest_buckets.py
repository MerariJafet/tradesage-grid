"""Bucket analysis for TradeSage aggregated backtests."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import pandas as pd


def analyze_buckets(trades_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, Dict[str, float]] = {}

    if trades_df.empty:
        return buckets

    trades_df = trades_df.copy()
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'], errors='coerce')
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'], errors='coerce')

    trades_df['hour'] = trades_df['entry_time'].dt.hour
    buckets['hora'] = trades_df.groupby('hour')['pnl'].mean().dropna().to_dict()

    if 'entry_atr' in trades_df.columns:
        trades_df['atr'] = pd.to_numeric(trades_df['entry_atr'], errors='coerce')
        trades_df['vol_bucket'] = pd.qcut(trades_df['atr'].fillna(0), q=3, labels=['bajo', 'medio', 'alto'])
        buckets['volatilidad'] = trades_df.groupby('vol_bucket')['pnl'].mean().dropna().to_dict()

    if 'regime' in trades_df.columns:
        buckets['regimen'] = trades_df.groupby('regime')['pnl'].mean().dropna().to_dict()

    trades_df['impetus'] = trades_df.apply(_derive_impetus, axis=1)
    buckets['impetus'] = trades_df.groupby('impetus')['pnl'].mean().dropna().to_dict()

    if 'reason' in trades_df.columns:
        buckets['salida'] = trades_df.groupby('reason')['pnl'].mean().dropna().to_dict()

    if 'symbol' in trades_df.columns:
        buckets['activo'] = trades_df.groupby('symbol')['pnl'].mean().dropna().to_dict()

    return buckets


def _derive_impetus(row) -> str:
    regime = row.get('regime')
    if regime == 'tendencial':
        return 'momentum'
    if regime == 'lateral':
        return 'range_play'
    side = row.get('side')
    return f"side_{side.lower()}" if isinstance(side, str) else 'desconocido'


def main():
    parser = argparse.ArgumentParser(description="Bucket analysis for edge backtests")
    parser.add_argument('--input', required=True, help='Path to pnl_log.txt file')
    parser.add_argument('--output', default='reports/buckets_analysis.txt', help='Output report file')
    args = parser.parse_args()

    pnl_path = Path(args.input)
    if not pnl_path.exists():
        raise FileNotFoundError(f"PnL log not found: {pnl_path}")

    trades_df = pd.read_csv(pnl_path)
    buckets = analyze_buckets(trades_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text_lines = ["Buckets Analysis"]
    for bucket, stats in buckets.items():
        text_lines.append(f"\n[{bucket}]")
        for key, value in sorted(stats.items()):
            text_lines.append(f"{key}: {value:.4f}")

    output_path.write_text("\n".join(text_lines))

    json_path = output_path.with_suffix('.json')
    json_path.write_text(json.dumps(buckets, indent=2, default=str))

    print(f"Buckets analysis saved to {output_path}")


if __name__ == '__main__':
    main()
