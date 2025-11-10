import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Backtesting Script for Aggregated Signals V2

Este script ejecuta backtesting de señales agregadas usando SignalAggregatorV2
con weighting dinámico, filtros y resolución inteligente de conflictos.
"""

import argparse
import asyncio
import logging
import math
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

import pandas as pd
import numpy as np

from app.core.orders import OrderManager
from app.core.strategies.signal_aggregator import SignalAggregatorV2
from app.core.strategies.momentum_scalping import MomentumScalpingStrategy
from app.core.strategies.mean_reversion import MeanReversionStrategy
from app.backtest.backtest_engine import BacktestEngine
from app.core.indicators.indicator_manager import IndicatorManager
from app.core.strategies_edge import detect_regime, filter_session_liquidity
from app.core.microstructure_signals import microstructure_signal
from app.core.strategies.position_sizer import PositionSizer
from app.core.strategies.signal_validator import SignalValidator
from app.backtest.data_loader import BinanceHistoricalDataLoader
from app.backtest.performance_analytics import PerformanceAnalytics
from app.utils.logger import get_logger

logger = get_logger("backtest_aggregated_signals")


class AggregatedSignalsBacktester:
    """
    Backtester especializado para señales agregadas V2

    Características:
    - Ejecuta múltiples estrategias en paralelo
    - Agrega señales usando SignalAggregatorV2
    - Aplica filtros de volatilidad y sesión
    - Mide rendimiento de señales agregadas vs individuales
    """

    def __init__(self, config: Dict):
        self.config = config
        self.data_loader = BinanceHistoricalDataLoader()
        self.backtest_engine = BacktestEngine()
        self.analytics = PerformanceAnalytics()
        self.order_manager = OrderManager()  # Gestor de órdenes avanzadas
        self.calibrated = config.get("calibration_mode", False)
        self.edge_mode = config.get("edge_mode", False)
        self.buckets_analysis = config.get("buckets_analysis", False)
        self.multi_symbol_mode = config.get("multi_symbol_mode", False)
        self._pnl_log_written = False

        # Configuración de estrategias
        self.strategies_config = {
            "MomentumScalping": {
                "class": MomentumScalpingStrategy,
                "params": config.get("momentum_params", {}),
                "enabled": True
            },
            "MeanReversion": {
                "class": MeanReversionStrategy,
                "params": config.get("mean_reversion_params", {}),
                "enabled": True
            }
        }

        enabled_filters = config.get("enabled_strategies") or []
        if enabled_filters:
            enabled_set = {name for name in enabled_filters}
            for strategy_name, strategy_details in self.strategies_config.items():
                strategy_details["enabled"] = strategy_name in enabled_set

        if config.get("disable_mean_reversion", False) and "MeanReversion" in self.strategies_config:
            self.strategies_config["MeanReversion"]["enabled"] = False

        # Inicializar agregador
        self.aggregator = SignalAggregatorV2(
            strategies_config=self.strategies_config,
            min_confidence=config.get("min_confidence", 0.6),
            require_confirmation=config.get("require_confirmation", False),
            single_signal_override=config.get("single_signal_override")
        )

        # Configurar filtros
        self.aggregator.configure_filters(
            volatility_enabled=config.get("volatility_filter", True),
            session_enabled=config.get("session_filter", True),
            liquidity_enabled=config.get("liquidity_filter", False),
            max_volatility=config.get("max_volatility_threshold", 0.05),
            session_start_time=time(8, 0) if self.edge_mode else None,
            session_end_time=time(16, 0) if self.edge_mode else None,
            volume_threshold=1.2 if self.edge_mode else None
        )

        self.backtest_engine.edge_mode = self.edge_mode
        self.enriched_data: Optional[pd.DataFrame] = None

        # Resultados
        self.results = {}
        self.pnl_history: List[Dict[str, Any]] = []

    def _augment_data_with_edge_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Compute ATR, ADX, bandwidth, volume SMA, and regime labels."""

        indicator_manager = IndicatorManager()
        indicator_manager.initialize_symbol(symbol)

        atr_values: List[Optional[float]] = []
        adx_values: List[Optional[float]] = []
        bandwidth_values: List[Optional[float]] = []
        regimes: List[str] = []

        for timestamp, row in data.iterrows():
            bar_dict = {
                'timestamp': timestamp,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }

            indicator_manager.update_with_bar(symbol, bar_dict)
            indicator_snapshot = indicator_manager.get_all_values(symbol)

            atr = indicator_snapshot.get('atr')
            adx = indicator_snapshot.get('adx')
            bandwidth = indicator_snapshot.get('bb_bandwidth')

            atr_values.append(atr)
            adx_values.append(adx)
            bandwidth_values.append(bandwidth)
            regimes.append(detect_regime(bar_dict, adx, bandwidth))

        enriched = data.copy()
        enriched['atr'] = atr_values
        enriched['adx'] = adx_values
        enriched['bb_bandwidth'] = bandwidth_values
        enriched['atr_pct'] = enriched['atr'] / enriched['close']
        enriched['volume_sma_20'] = enriched['volume'].rolling(window=20, min_periods=1).mean()
        enriched['regime'] = regimes
        if 'funding_rate' not in enriched.columns:
            enriched['funding_rate'] = 0.0

        return enriched

    async def run_backtest(self, symbol: str, start_date: str, end_date: str,
                          timeframe: str = "5m") -> Dict:
        """
        Ejecutar backtesting completo de señales agregadas

        Args:
            symbol: Símbolo a testear
            start_date: Fecha inicio (YYYY-MM-DD)
            end_date: Fecha fin (YYYY-MM-DD)
            timeframe: Timeframe (5m, 15m, 1h)

        Returns:
            Diccionario con resultados completos
        """
        logger.info("starting_aggregated_signals_backtest",
                   symbol=symbol, start_date=start_date, end_date=end_date)

        if os.path.exists('error_log_optimized.txt'):
            os.remove('error_log_optimized.txt')

        try:
            # 1. Cargar datos
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            async with self.data_loader as loader:
                bars = await loader.download_klines(
                    symbol=symbol,
                    interval=timeframe,
                    start_date=start_dt,
                    end_date=end_dt
                )

            if not bars:
                raise ValueError(f"No data available for {symbol}")

            data = self.data_loader.bars_to_dataframe(bars)

            # Ensure index is datetime for downstream processing
            if not np.issubdtype(data.index.dtype, np.datetime64):
                data.index = pd.to_datetime(data.index, unit='ms', errors='coerce')

            data = data[~data.index.isna()].sort_index()

            if 'timestamp' in data.columns:
                data = data.drop(columns=['timestamp'])

            logger.info("data_loaded", symbol=symbol, rows=len(data), start=data.index.min(), end=data.index.max())

            if self.edge_mode:
                data = self._augment_data_with_edge_features(data, symbol)

            self.enriched_data = data

            # 2. Generar señales individuales
            individual_signals = await self._generate_individual_signals(data, symbol)

            # 3. Generar señales agregadas
            aggregated_signals = await self._generate_aggregated_signals(data, symbol, individual_signals)

            # 4. Ejecutar backtests
            results = await self._run_backtests(data, individual_signals, aggregated_signals, symbol)

            # 5. Guardar métricas de QA resumidas y log de PnL
            summary_metrics = self._extract_summary_metrics(results)
            # Always refresh the PnL-enabled QA artifact for downstream consumers
            self._write_summary_metrics(summary_metrics, symbol, start_date, end_date)
            if self.calibrated:
                self._write_summary_metrics(
                    summary_metrics,
                    symbol,
                    start_date,
                    end_date,
                    file_name="reports/qa_metrics_calibrated.json"
                )
                self._write_summary_metrics(
                    summary_metrics,
                    symbol,
                    start_date,
                    end_date,
                    file_name="reports/qa_metrics_calibrated_v2.json"
                )
                self._write_summary_metrics(
                    summary_metrics,
                    symbol,
                    start_date,
                    end_date,
                    file_name="reports/qa_metrics_calibrated_v3.json"
                )
                self._write_summary_metrics(
                    summary_metrics,
                    symbol,
                    start_date,
                    end_date,
                    file_name="reports/qa_metrics_calibrated_v4.json"
                )
                self._write_calibration_log(summary_metrics, results, symbol, start_date, end_date)
            append_mode = self.multi_symbol_mode
            include_header = not append_mode or not self._pnl_log_written
            self._write_pnl_log(symbol, start_date, end_date, append=append_mode, include_header=include_header)
            if append_mode:
                self._pnl_log_written = True

            if self.buckets_analysis and not self.multi_symbol_mode:
                self._run_bucket_analysis()

            # 6. Generar reporte completo (sin trades detallados)
            report = await self._generate_report(results, symbol, start_date, end_date)

            self.results[symbol] = report

            logger.info("backtest_completed", symbol=symbol, total_trades=report.get("total_trades", 0))

            return report

        except Exception as e:
            logger.error("backtest_failed", symbol=symbol, error=str(e))
            raise

    async def _generate_individual_signals(self, data: pd.DataFrame, symbol: str) -> Dict[str, List]:
        """Generar señales de cada estrategia individualmente"""
        logger.info("starting_individual_signal_generation", symbol=symbol, data_rows=len(data))
        individual_signals: Dict[str, List] = {}
        volume_sma_series = data['volume_sma_20'] if self.edge_mode and 'volume_sma_20' in data.columns else None
        regime_series = data['regime'] if self.edge_mode and 'regime' in data.columns else None

        for strategy_name, strategy_config in self.strategies_config.items():
            if not strategy_config["enabled"]:
                continue

            logger.info("processing_strategy", strategy=strategy_name)

            try:
                strategy_class = strategy_config["class"]
                params = strategy_config["params"]

                # Crear componentes necesarios para la estrategia
                indicator_manager = IndicatorManager()
                indicator_manager.initialize_symbol(symbol)
                position_sizer = PositionSizer(account_balance=10000.0)
                signal_validator = SignalValidator(indicator_manager=indicator_manager)

                # Instanciar estrategia con dependencias
                strategy = strategy_class(
                    symbol=symbol,
                    indicator_manager=indicator_manager,
                    position_sizer=position_sizer,
                    signal_validator=signal_validator,
                    **params
                )

                if self.config.get("session_filter"):
                    strategy.session_filter_enabled = True
                    strategy.session_start_time = time(9, 0)
                    strategy.session_end_time = time(18, 0)

                signals: List = []
                signals_attempted = 0

                for idx, (_, row) in enumerate(data.iterrows()):
                    bar_dict = {
                        'timestamp': row.name,
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    }

                    indicator_manager.update_with_bar(symbol, bar_dict)

                    if idx < strategy.min_periods:
                        continue

                    if self.edge_mode:
                        volume_sma_value = volume_sma_series.iloc[idx] if volume_sma_series is not None else None
                        if not filter_session_liquidity(bar_dict, volume_sma_value):
                            continue

                    signals_attempted += 1

                    market_data = {
                        'bar': bar_dict,
                        'indicators': {},
                        'orderbook': self._create_mock_orderbook(bar_dict)
                    }

                    if self.edge_mode:
                        market_data['edge_context'] = {
                            'regime': regime_series.iloc[idx] if regime_series is not None else None,
                            'volume_sma': volume_sma_series.iloc[idx] if volume_sma_series is not None else None
                        }

                    signal = await strategy.generate_signal(market_data)
                    if signal:
                        signal.timestamp = pd.to_datetime(row.name).to_pydatetime()
                        signals.append(signal)

                logger.info(
                    "signal_generation_attempted",
                    strategy=strategy_name,
                    signals_attempted=signals_attempted,
                    signals_generated=len(signals)
                )

                individual_signals[strategy_name] = signals

                logger.info(
                    "individual_signals_generated",
                    strategy=strategy_name,
                    signals=len(signals),
                    total_signals_across_all=str(sum(len(s) for s in individual_signals.values()))
                )

            except Exception as e:
                logger.error(
                    "strategy_signal_generation_failed",
                    strategy=strategy_name,
                    error=str(e)
                )
                individual_signals[strategy_name] = []

        return individual_signals

    def _create_mock_orderbook(self, bar: dict) -> dict:
        """
        Create mock orderbook data for backtesting validation.
        Simulates realistic market conditions that will pass signal validation.
        """
        current_price = bar['close']

        spread_pct = 0.0001  # 0.01%
        spread_amount = current_price * spread_pct

        best_bid = current_price - (spread_amount / 2)
        best_ask = current_price + (spread_amount / 2)

        bid_levels = []
        ask_levels = []

        for i in range(10):
            price = best_bid - (i * 0.01)
            volume = 50.0 / (i + 1)
            bid_levels.append([price, volume])

        for i in range(10):
            price = best_ask + (i * 0.01)
            volume = 50.0 / (i + 1)
            ask_levels.append([price, volume])

        return {
            'bids': bid_levels,
            'asks': ask_levels,
            'timestamp': bar['timestamp']
        }

    async def _generate_aggregated_signals(self, data: pd.DataFrame, symbol: str,
                                           individual_signals: Dict) -> List:
        """Generar señales agregadas usando SignalAggregatorV2 y crear órdenes avanzadas"""
        aggregated_signals: List = []
        order_ids_created: List[str] = []

        indicator_manager = IndicatorManager()
        indicator_manager.initialize_symbol(symbol)

        volume_sma_series = data['volume_sma_20'] if self.edge_mode and 'volume_sma_20' in data.columns else None
        regime_series = data['regime'] if self.edge_mode and 'regime' in data.columns else None
        adx_series = data['adx'] if self.edge_mode and 'adx' in data.columns else None
        bandwidth_series = data['bb_bandwidth'] if self.edge_mode and 'bb_bandwidth' in data.columns else None
        atr_series = data['atr'] if self.edge_mode and 'atr' in data.columns else None
        funding_series = data['funding_rate'] if self.edge_mode and 'funding_rate' in data.columns else None

        for _, row in data.iterrows():
            bar_dict = {
                'timestamp': row.name,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }
            indicator_manager.update_with_bar(symbol, bar_dict)

        all_individual_signals: List = []
        for strategy_signals in individual_signals.values():
            all_individual_signals.extend(strategy_signals)

        logger.info("total_individual_signals_collected", count=len(all_individual_signals))

        all_individual_signals.sort(key=lambda s: pd.to_datetime(s.timestamp))

        signals_by_bar: Dict[str, List] = {}
        for signal in all_individual_signals:
            signal_time = pd.to_datetime(signal.timestamp)
            bar_timestamp = signal_time.floor('min')
            bar_key = bar_timestamp.isoformat()

            signals_by_bar.setdefault(bar_key, []).append(signal)

        logger.info("signals_grouped_by_bar", bars=len(signals_by_bar), total_signals=len(all_individual_signals))

        for bar_timestamp_str, bar_signals in signals_by_bar.items():
            try:
                bar_timestamp = pd.to_datetime(bar_timestamp_str)
                closest_idx = data.index.get_indexer([bar_timestamp], method='nearest')[0]
                if closest_idx < 0:
                    continue

                market_data_window = data.iloc[max(0, closest_idx-50):closest_idx+1]
                current_row = data.iloc[closest_idx]
                bar_dict = {
                    'timestamp': bar_timestamp,
                    'open': current_row['open'],
                    'high': current_row['high'],
                    'low': current_row['low'],
                    'close': current_row['close'],
                    'volume': current_row['volume']
                }

                if self.edge_mode:
                    volume_sma_value = volume_sma_series.iloc[closest_idx] if volume_sma_series is not None else None
                    if not filter_session_liquidity(bar_dict, volume_sma_value):
                        self.aggregator.clear_signals(symbol)
                        continue

                    regime = regime_series.iloc[closest_idx] if regime_series is not None else None
                    adx_value = adx_series.iloc[closest_idx] if adx_series is not None else None
                    bandwidth_value = bandwidth_series.iloc[closest_idx] if bandwidth_series is not None else None
                    atr_value = atr_series.iloc[closest_idx] if atr_series is not None else None
                    funding_value = funding_series.iloc[closest_idx] if funding_series is not None else 0.0
                else:
                    regime = None
                    adx_value = None
                    bandwidth_value = None
                    atr_value = None
                    funding_value = 0.0

                for signal in bar_signals:
                    self.aggregator.add_signal(signal)

                agg_signal = self.aggregator.aggregate_signals(
                    symbol=symbol,
                    market_data=market_data_window,
                    timestamp=bar_timestamp.to_pydatetime()
                )

                if agg_signal:
                    if self.edge_mode:
                        indicator_snapshot = agg_signal.indicators or {}
                        if adx_value is not None:
                            indicator_snapshot.setdefault('adx', float(adx_value))
                        if bandwidth_value is not None:
                            indicator_snapshot.setdefault('bb_bandwidth', float(bandwidth_value))
                        if atr_value is not None:
                            indicator_snapshot.setdefault('atr', float(atr_value))
                        agg_signal.indicators = indicator_snapshot

                        agg_signal.reason += f" | regime={regime}"

                        estimated_bid = float(bar_dict['volume']) * (0.55 if bar_dict['close'] >= bar_dict['open'] else 0.45)
                        estimated_ask = max(float(bar_dict['volume']) - estimated_bid, 1e-6)
                        micro_bias = microstructure_signal(bar_dict, estimated_bid, estimated_ask, funding_value)
                        if micro_bias:
                            agg_signal.action = micro_bias.upper()
                            agg_signal.reason += f" | micro_bias={micro_bias}"

                    aggregated_signals.append(agg_signal)

                    order_ids = self.aggregator.create_advanced_orders(
                        agg_signal,
                        self.order_manager,
                        market_data_window,
                        self.config.get("advanced_orders_params", {})
                    )
                    order_ids_created.extend(order_ids)

                    logger.debug(
                        "advanced_orders_created_for_signal",
                        signal_confidence=agg_signal.confidence,
                        orders_created=len(order_ids),
                        signals_in_bar=len(bar_signals)
                    )

                self.aggregator.clear_signals(symbol)

            except Exception as e:
                logger.error(
                    "error_processing_signal_group",
                    error=str(e),
                    bar_timestamp=bar_timestamp_str,
                    signals_in_group=len(bar_signals)
                )

                error_line = (
                    f"{datetime.utcnow().isoformat()} | error_processing_signal_group | "
                    f"bar_timestamp={bar_timestamp_str} | signals_in_group={len(bar_signals)} | error={str(e)}\n"
                )
                try:
                    with open('error_log_optimized.txt', 'a') as error_file:
                        error_file.write(error_line)
                except OSError:
                    pass

        logger.info(
            "aggregated_signals_and_orders_generated",
            symbol=symbol,
            signals=len(aggregated_signals),
            total_orders_created=len(order_ids_created),
            bars_processed=len(signals_by_bar)
        )

        return aggregated_signals

    async def _run_backtests(self, data: pd.DataFrame, individual_signals: Dict,
                           aggregated_signals: List, symbol: str) -> Dict:
        """Ejecutar backtests para señales individuales y agregadas"""

        results = {
            "individual": {},
            "aggregated": {},
            "comparison": {}
        }

        # Backtest señales individuales
        for strategy_name, signals in individual_signals.items():
            if not signals:
                continue

            try:
                total_trades = len(signals)

                backtest_result = {
                    "total_trades": total_trades,
                    "win_rate": 0.0,  # No podemos calcular sin ejecución real
                    "total_pnl": 0.0,
                    "final_balance": self.config.get("initial_balance", 10000.0),
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "signals_analyzed": total_trades
                }

                results["individual"][strategy_name] = backtest_result

                logger.debug("individual_backtest_completed",
                           strategy=strategy_name,
                           trades=total_trades)

            except Exception as e:
                logger.error("individual_backtest_failed",
                           strategy=strategy_name, error=str(e))

        # Backtest señales agregadas
        try:
            if aggregated_signals:
                backtest_outcome = await self.backtest_engine.run_aggregated_signals_backtest(
                    data=data,
                    aggregated_signals=[s.dict() if hasattr(s, "dict") else s for s in aggregated_signals],
                    symbol=symbol,
                    risk_per_trade_pct=self.config.get("risk_per_trade_pct", 0.02),
                    order_manager=self.order_manager,
                    force_fill_simulation=self.config.get("force_fill_simulation", False),
                    dynamic_tp_sl=self.config.get("dynamic_tp_sl", False),
                    atr_stop_multiplier=self.config.get("atr_stop_multiplier", 1.8),
                    tp_sl_ratio=self.config.get("tp_sl_ratio", 2.0)
                )

                agg_metrics = backtest_outcome.get("metrics", {}) or {}
                self.pnl_history = backtest_outcome.get("pnl_history", [])
                executed_trades = backtest_outcome.get("trades", [])

                total_signals = len(aggregated_signals)
                filled_trades = len(executed_trades)
                fill_rate_pct = (filled_trades / total_signals * 100) if total_signals else 0.0

                results["aggregated"] = {
                    "total_trades": agg_metrics.get("total_trades", 0),
                    "total_signals": total_signals,
                    "filled_trades": filled_trades,
                    "fill_rate_pct": fill_rate_pct,
                    "win_rate": agg_metrics.get("win_rate", 0.0),
                    "win_rate_pct": agg_metrics.get("win_rate", 0.0) * 100,
                    "total_pnl": agg_metrics.get("total_pnl", 0.0),
                    "final_balance": agg_metrics.get("final_balance", self.config.get("initial_balance", 10000.0)),
                    "sharpe_ratio": agg_metrics.get("sharpe_ratio", 0.0),
                    "max_drawdown": agg_metrics.get("max_drawdown", 0.0),
                    "profit_factor": agg_metrics.get("profit_factor", 0.0),
                    "expectancy": agg_metrics.get("expectancy", 0.0),
                    "slippage_pct": self.backtest_engine.slippage_pct
                }

                logger.info(
                    "aggregated_backtest_completed",
                    symbol=symbol,
                    trades=results["aggregated"].get("total_trades", 0)
                )
        except Exception as e:
            logger.error("aggregated_backtest_failed", symbol=symbol, error=str(e))

        # Comparación
        results["comparison"] = self._compare_results(results)

        return results

    def _compare_results(self, results: Dict) -> Dict:
        """Comparar rendimiento de estrategias individuales vs agregadas"""
        comparison = {
            "best_individual_strategy": None,
            "aggregated_vs_best": {},
            "improvement_metrics": {}
        }

        if not results["individual"]:
            return comparison

        # Encontrar mejor estrategia individual
        best_strategy = None
        best_sharpe = -float('inf')

        for strategy_name, result in results["individual"].items():
            if result.get("sharpe_ratio", -float('inf')) > best_sharpe:
                best_sharpe = result["sharpe_ratio"]
                best_strategy = strategy_name

        comparison["best_individual_strategy"] = best_strategy

        # Comparar agregada vs mejor individual
        if results["aggregated"] and best_strategy:
            agg_result = results["aggregated"]
            best_ind_result = results["individual"][best_strategy]

            comparison["aggregated_vs_best"] = {
                "sharpe_ratio_diff": agg_result.get("sharpe_ratio", 0) - best_ind_result.get("sharpe_ratio", 0),
                "profit_factor_diff": agg_result.get("profit_factor", 1) - best_ind_result.get("profit_factor", 1),
                "win_rate_diff": agg_result.get("win_rate", 0) - best_ind_result.get("win_rate", 0),
                "expectancy_diff": agg_result.get("expectancy", 0) - best_ind_result.get("expectancy", 0),
                "max_drawdown_diff": best_ind_result.get("max_drawdown", 0) - agg_result.get("max_drawdown", 0)  # Menor es mejor
            }

        # Métricas de mejora
        if results["aggregated"]:
            agg_result = results["aggregated"]
            comparison["improvement_metrics"] = {
                "sharpe_ratio": agg_result.get("sharpe_ratio", 0),
                "profit_factor": agg_result.get("profit_factor", 1),
                "win_rate": agg_result.get("win_rate", 0),
                "expectancy": agg_result.get("expectancy", 0),
                "max_drawdown": agg_result.get("max_drawdown", 0),
                "total_trades": agg_result.get("total_trades", 0)
            }

        return comparison

    async def _generate_report(self, results: Dict, symbol: str,
                             start_date: str, end_date: str) -> Dict:
        """Generar reporte completo de backtesting"""

        summary_metrics = self._extract_summary_metrics(results)

        report = {
            "symbol": symbol,
            "period": {"start": start_date, "end": end_date},
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": summary_metrics,
            "results": results,
            "aggregated_signal_count": results.get("aggregated", {}).get("total_trades", 0),
            "individual_strategy_trade_counts": {
                name: details.get("total_trades", 0)
                for name, details in results.get("individual", {}).items()
            },
            "notes": "Detailed trade breakdown omitted to keep QA artifacts lightweight"
        }

        logger.info("summary_report_generated", symbol=symbol, metrics=summary_metrics)

        return report

    def _extract_summary_metrics(self, results: Dict) -> Dict[str, Any]:
        """Extraer métricas de resumen necesarias para QA."""
        aggregated = results.get("aggregated", {}) or {}

        return {
            "sharpe_ratio": aggregated.get("sharpe_ratio", 0.0),
            "profit_factor": aggregated.get("profit_factor", 0.0),
            "max_drawdown": aggregated.get("max_drawdown", 0.0),
            "expectancy": aggregated.get("expectancy", 0.0),
            "fill_rate_pct": aggregated.get("fill_rate_pct", 0.0),
            "slippage_pct": aggregated.get("slippage_pct", 0.0),
            "total_trades": aggregated.get("total_trades", 0),
            "win_rate": aggregated.get("win_rate", 0.0),
            "win_rate_pct": aggregated.get("win_rate", 0.0) * 100
        }

    def _write_summary_metrics(self, summary: Dict[str, Any], symbol: str,
                               start_date: str, end_date: str,
                               file_name: Optional[str] = None) -> None:
        """Persistir métricas de resumen en el archivo indicado."""
        os.makedirs('reports', exist_ok=True)

        payload = {
            "symbol": symbol,
            "period": {"start": start_date, "end": end_date},
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": summary
        }

        summary_file = file_name or "reports/qa_metrics_pnl_enabled.json"
        with open(summary_file, 'w') as f:
            json.dump(payload, f, indent=2)

        logger.info("summary_metrics_saved", file=summary_file)

    def _write_pnl_log(self, symbol: str, start_date: str, end_date: str,
                       append: bool = False, include_header: bool = True) -> None:
        """Persistir historial de PnL en reports/pnl_log.txt."""

        os.makedirs('reports', exist_ok=True)
        log_path = Path("reports/pnl_log.txt")
        mode = 'a' if append else 'w'

        def _format_optional(value: Any, precision: int = 6) -> str:
            if value is None:
                return ''
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return ''
            if math.isnan(numeric) or math.isinf(numeric):
                return ''
            return f"{numeric:.{precision}f}"

        def _to_float(value: Any, default: float = 0.0) -> float:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return default
            if math.isnan(numeric) or math.isinf(numeric):
                return default
            return numeric

        try:
            with log_path.open(mode) as handle:
                header = (
                    "symbol,trade_id,side,entry_time,exit_time,entry_price,exit_price,"
                    "pnl,pnl_pct,reason,entry_atr,entry_adx,entry_bandwidth,regime\n"
                )
                if include_header:
                    handle.write(header)

                if not self.pnl_history:
                    line = (
                        f"{symbol},N/A,N/A,{start_date},{end_date},0,0,0,0,no_trades_executed,,,,\n"
                    )
                    handle.write(line)
                    return

                for trade in self.pnl_history:
                    entry_ts = (
                        datetime.utcfromtimestamp(trade['entry_timestamp'] / 1000).isoformat()
                        if trade.get('entry_timestamp') else ''
                    )
                    exit_ts = (
                        datetime.utcfromtimestamp(trade['exit_timestamp'] / 1000).isoformat()
                        if trade.get('exit_timestamp') else ''
                    )

                    trade_symbol = trade.get('symbol') or symbol
                    entry_price = _to_float(trade.get('entry_price'))
                    exit_price = _to_float(trade.get('exit_price'))
                    pnl_value = _to_float(trade.get('pnl'))
                    pnl_pct = _to_float(trade.get('pnl_pct'))
                    entry_atr = _format_optional(trade.get('entry_atr'))
                    entry_adx = _format_optional(trade.get('entry_adx'))
                    entry_bandwidth = _format_optional(trade.get('entry_bandwidth'))
                    regime = trade.get('regime') or ''

                    handle.write(
                        (
                            f"{trade_symbol},{trade.get('trade_id','')},{trade.get('side','')},"
                            f"{entry_ts},{exit_ts},{entry_price:.2f},{exit_price:.2f},"
                            f"{pnl_value:.2f},{pnl_pct:.4f},{trade.get('reason','')},"
                            f"{entry_atr},{entry_adx},{entry_bandwidth},{regime}\n"
                        )
                    )

            logger.info("pnl_log_saved", file=str(log_path), entries=len(self.pnl_history))
        except OSError as exc:
            logger.error("pnl_log_write_failed", file=str(log_path), error=str(exc))

    def _run_bucket_analysis(self, input_path: str = "reports/pnl_log.txt",
                             output_path: str = "reports/buckets_analysis.txt") -> None:
        """Generate bucket analysis artifacts from the PnL log."""

        try:
            from scripts import backtest_buckets
        except ImportError as exc:
            logger.error("bucket_analysis_import_failed", error=str(exc))
            return

        pnl_file = Path(input_path)
        if not pnl_file.exists():
            logger.warning("bucket_analysis_skipped", reason="pnl_log_missing", file=str(pnl_file))
            return

        try:
            trades_df = pd.read_csv(pnl_file)
        except Exception as exc:
            logger.error("bucket_analysis_read_failed", file=str(pnl_file), error=str(exc))
            return

        buckets = backtest_buckets.analyze_buckets(trades_df)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        lines = ["Buckets Analysis"]
        for bucket, stats in buckets.items():
            lines.append(f"\n[{bucket}]")
            for key, value in sorted(stats.items()):
                lines.append(f"{key}: {value:.4f}")

        output_file.write_text("\n".join(lines))
        json_path = output_file.with_suffix('.json')
        json_path.write_text(json.dumps(buckets, indent=2, default=str))

        logger.info("bucket_analysis_completed", file=str(output_file), buckets=len(buckets))

    def _write_calibration_log(self, summary: Dict[str, Any], results: Dict[str, Any],
                               symbol: str, start_date: str, end_date: str) -> None:
        """Persistir ajustes y métricas de calibración."""

        os.makedirs('reports', exist_ok=True)
        log_path = "reports/calibration_log.txt"

        aggregated = results.get("aggregated", {}) or {}
        momentum_params = self.config.get("momentum_params", {})
        mean_rev_params = self.config.get("mean_reversion_params", {})

        total_trades = aggregated.get("total_trades", 0)
        win_rate_pct = aggregated.get("win_rate", 0.0) * 100
        profit_factor = aggregated.get("profit_factor", 0.0)
        sharpe_ratio = aggregated.get("sharpe_ratio", 0.0)
        expectancy = aggregated.get("expectancy", 0.0)
        fill_rate_pct = aggregated.get("fill_rate_pct", 0.0)

        total_exits = len(self.pnl_history)
        stop_loss_exits = sum(1 for trade in self.pnl_history if trade.get("reason") == "stop_loss")
        stop_loss_ratio_pct = (stop_loss_exits / total_exits * 100) if total_exits else 0.0

        lines = [
            f"Calibration Run @ {datetime.utcnow().isoformat()}",
            f"Symbol: {symbol} | Period: {start_date} -> {end_date}",
            "",
            "Strategy Parameters:",
            f"  MomentumScalping: {json.dumps(momentum_params, sort_keys=True)}",
            f"  MeanReversion: {json.dumps(mean_rev_params, sort_keys=True)}",
            "",
            "Aggregated Performance:",
            f"  Total Trades: {total_trades}",
            f"  Win Rate %: {win_rate_pct:.2f}",
            f"  Profit Factor: {profit_factor:.2f}",
            f"  Sharpe Ratio: {sharpe_ratio:.2f}",
            f"  Expectancy: {expectancy:.2f}",
            f"  Fill Rate %: {fill_rate_pct:.2f}",
            f"  Stop-Loss Exit %: {stop_loss_ratio_pct:.2f}",
            "",
            "Summary Metrics:",
            json.dumps(summary, indent=2, default=str)
        ]

        try:
            with open(log_path, 'w') as logfile:
                logfile.write("\n".join(lines) + "\n")
            logger.info("calibration_log_saved", file=log_path)
        except OSError as exc:
            logger.error("calibration_log_write_failed", file=log_path, error=str(exc))

    async def run_multiple_symbols(self, symbols: List[str], start_date: str,
                                 end_date: str, timeframe: str = "5m") -> Dict:
        """Ejecutar backtesting para múltiples símbolos"""

        all_results = {}

        if self.multi_symbol_mode:
            log_path = Path("reports/pnl_log.txt")
            if log_path.exists():
                log_path.unlink()
            self._pnl_log_written = False

        for symbol in symbols:
            try:
                result = await self.run_backtest(symbol, start_date, end_date, timeframe)
                all_results[symbol] = result

                # Pequeña pausa para no sobrecargar
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error("symbol_backtest_failed", symbol=symbol, error=str(e))
                all_results[symbol] = {"error": str(e)}

        # Generar reporte consolidado
        consolidated_report = {
            "symbols_tested": symbols,
            "period": {"start": start_date, "end": end_date},
            "timestamp": datetime.utcnow().isoformat(),
            "results": all_results,
            "summary": self._generate_consolidated_summary(all_results)
        }

        # Guardar reporte consolidado
        consolidated_file = f"reports/consolidated_aggregated_backtest_{start_date}_{end_date}.json"
        with open(consolidated_file, 'w') as f:
            json.dump(consolidated_report, f, indent=2, default=str)

        logger.info("consolidated_report_saved", file=consolidated_file)

        if self.buckets_analysis:
            self._run_bucket_analysis()

        return consolidated_report

    def _generate_consolidated_summary(self, all_results: Dict) -> Dict:
        """Generar resumen consolidado de todos los símbolos"""

        summary = {
            "total_symbols": len(all_results),
            "successful_tests": 0,
            "failed_tests": 0,
            "average_metrics": {},
            "best_performing_symbol": None,
            "worst_performing_symbol": None
        }

        successful_results = []
        best_sharpe = -float('inf')
        worst_sharpe = float('inf')

        for symbol, result in all_results.items():
            if "error" not in result:
                summary["successful_tests"] += 1
                successful_results.append(result)

                # Tracking best/worst
                sharpe = result.get("results", {}).get("aggregated", {}).get("sharpe_ratio", -float('inf'))
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    summary["best_performing_symbol"] = symbol
                if sharpe < worst_sharpe:
                    worst_sharpe = sharpe
                    summary["worst_performing_symbol"] = symbol
            else:
                summary["failed_tests"] += 1

        # Calcular promedios
        if successful_results:
            metrics_to_average = ["sharpe_ratio", "profit_factor", "win_rate", "expectancy", "max_drawdown"]

            for metric in metrics_to_average:
                values = [
                    r.get("results", {}).get("aggregated", {}).get(metric, 0)
                    for r in successful_results
                    if r.get("results", {}).get("aggregated", {}).get(metric) is not None
                ]

                if values:
                    summary["average_metrics"][metric] = sum(values) / len(values)

        return summary


def _period_to_days(period: str) -> int:
    normalized = period.strip().lower()

    if normalized.endswith("months"):
        value = normalized[:-6]
        multiplier = 30
    elif normalized.endswith("month"):
        value = normalized[:-5]
        multiplier = 30
    elif normalized.endswith("days"):
        value = normalized[:-4]
        multiplier = 1
    elif normalized.endswith("day"):
        value = normalized[:-3]
        multiplier = 1
    else:
        raise ValueError(f"Unsupported period format: {period}")

    value = value.strip()
    if not value.isdigit():
        raise ValueError(f"Unsupported period format: {period}")

    return int(value) * multiplier


def _resolve_backtest_dates(period: Optional[str],
                            start_date: Optional[str],
                            end_date: Optional[str]) -> tuple[str, str]:
    resolved_start = start_date
    resolved_end = end_date
    end_dt = None

    if resolved_end:
        end_dt = datetime.strptime(resolved_end, "%Y-%m-%d").date()

    if period:
        days = _period_to_days(period)

        if end_dt is None:
            end_dt = datetime.utcnow().date()
            resolved_end = end_dt.strftime("%Y-%m-%d")

        start_dt = end_dt - timedelta(days=days)
        resolved_start = resolved_start or start_dt.strftime("%Y-%m-%d")

    if resolved_end is None:
        end_dt = datetime.utcnow().date()
        resolved_end = end_dt.strftime("%Y-%m-%d")
    else:
        if end_dt is None:
            end_dt = datetime.strptime(resolved_end, "%Y-%m-%d").date()

    if resolved_start is None:
        fallback_start = end_dt - timedelta(days=90)
        resolved_start = fallback_start.strftime("%Y-%m-%d")

    return resolved_start, resolved_end


async def main(symbol: str = "BTCUSDT", symbols: Optional[List[str]] = None,
               start_date: str = "2025-01-01", end_date: str = "2025-09-30",
               timeframe: str = "1m", calibrated: bool = False,
               force_fill_simulation: bool = False,
               dynamic_tp_sl: bool = False, optimized_filters: bool = False,
               relaxed_filters: bool = False, edge_mode: bool = False,
               buckets_analysis: bool = False, tp_sl_ratio: float = 2.0,
               strategy: Optional[str] = None,
               disable_mean_reversion: bool = False):
    """Función principal para ejecutar backtesting"""

    # Ensure local environment variables point to the host Redis when running locally
    # Docker compose uses REDIS_HOST=redis for container networking; for local runs
    # we'll prefer localhost if the redis hostname isn't resolvable.
    os.environ.setdefault('REDIS_HOST', os.environ.get('REDIS_HOST', 'redis-1'))
    os.environ.setdefault('REDIS_PORT', os.environ.get('REDIS_PORT', '6379'))

    symbol_list = [s.strip().upper() for s in (symbols or []) if s and s.strip()]
    target_symbols = symbol_list if symbol_list else [symbol.upper()]
    primary_symbol = target_symbols[0]
    multi_symbol_run = len(target_symbols) > 1

    # Configuración del backtest
    config = {
        "min_confidence": 0.6,  # User's requirement
        "volatility_filter": True,  # Enable volatility filter
        "session_filter": False,
        "liquidity_filter": False,
        "max_volatility_threshold": 0.05,  # ATR > 5%
        "momentum_params": {
            "macd_fast": 8, "macd_slow": 21, "macd_signal": 5,  # Faster MACD for scalping
            "adx_period": 14, "adx_threshold": 25,
            "supertrend_period": 10, "supertrend_multiplier": 3.0,
            "rsi_period": 14, "rsi_momentum": 50.0,
            "rsi_overbought": 55.0, "rsi_oversold": 45.0,
            "volume_sma_period": 20, "volume_multiplier": 1.5,
            "atr_period": 10, "stop_multiplier": 1.0,
            "atr_take_profit_multiplier": 2.0,
            "profit_target_pct": 0.005, "stop_loss_pct": 0.003
        },
        "mean_reversion_params": {
            "rsi_period": 7, "rsi_overbought": 75, "rsi_oversold": 25,
            "bb_period": 10, "bb_std": 1.5,
            "volume_multiplier": 1.5, "atr_stop_multiplier": 1.5,
            "profit_target_atr_multiplier": 2.0
        },
        "advanced_orders_params": {
            "limit_offset": 0.001,  # 0.1% (within 0.05-0.2% range)
            "iceberg_visible": 0.05,  # 0.05 BTC
            "trailing_atr": 1.2,  # 1.0-1.5x ATR
            "stop_adaptive": True  # volatility_based
        },
        "risk_per_trade_pct": 0.02,
        "calibration_mode": calibrated,
        "require_confirmation": False,
        "single_signal_override": None,
        "force_fill_simulation": force_fill_simulation,
        "dynamic_tp_sl": dynamic_tp_sl,
        "optimized_filters": optimized_filters,
        "relaxed_filters": relaxed_filters,
        "atr_stop_multiplier": 1.8,
        "tp_sl_ratio": tp_sl_ratio,
        "enabled_strategies": None,
        "disable_mean_reversion": disable_mean_reversion
    }

    if strategy:
        selected_strategies = [s.strip() for s in strategy.split(',') if s.strip()]
        if selected_strategies:
            config["enabled_strategies"] = selected_strategies

    if calibrated:
        config["momentum_params"].update({
            "rsi_period": 5,
            "rsi_overbought": 68.0,
            "rsi_oversold": 32.0,
            "adx_threshold": 30.0,
            "bollinger_window": 20,
            "stop_multiplier": 2.0,
            "atr_take_profit_multiplier": 3.0,
            "profit_target_pct": 0.0,  # rely on ATR-based TP
            "stop_loss_pct": 0.0
        })
        config["mean_reversion_params"].update({
            "atr_stop_multiplier": 2.0,
            "profit_target_atr_multiplier": 3.5,
            "rsi_overbought": 80.0,
            "rsi_oversold": 20.0
        })
        config["risk_per_trade_pct"] = 0.01
        config["min_confidence"] = 0.85
        config["require_confirmation"] = True
        config["single_signal_override"] = 0.9
        config["session_filter"] = True

    if optimized_filters:
        config["min_confidence"] = max(config.get("min_confidence", 0.6), 0.7)
        config["require_confirmation"] = True
        # Allow exceptionally strong single signals to bypass double-confirmation
        existing_override = config.get("single_signal_override") or 0.0
        config["single_signal_override"] = max(existing_override, 0.88)

    if relaxed_filters:
        config["min_confidence"] = min(config.get("min_confidence", 0.6), 0.55)
        config["require_confirmation"] = False
        existing_override = config.get("single_signal_override") or 0.0
        config["single_signal_override"] = max(existing_override, 0.75)

    if edge_mode:
        config["edge_mode"] = True
        config["session_filter"] = True
        config["liquidity_filter"] = True
    else:
        config["edge_mode"] = False

    config["buckets_analysis"] = buckets_analysis
    config["multi_symbol_mode"] = multi_symbol_run

    # Inicializar backtester
    backtester = AggregatedSignalsBacktester(config)

    logger.info("starting_backtest_execution",
               symbols=", ".join(target_symbols), start=start_date, end=end_date)

    try:
        if multi_symbol_run:
            consolidated = await backtester.run_multiple_symbols(target_symbols, start_date, end_date, timeframe)

            print("\n=== CONSOLIDATED BACKTEST RESULTS ===")
            print(f"Symbols: {', '.join(target_symbols)}")
            print(f"Period: {start_date} to {end_date}")
            print(f"Timeframe: {timeframe}")

            summary = consolidated.get("summary", {})
            print("\nAverage Metrics:")
            for metric, value in summary.get("average_metrics", {}).items():
                print(f"  {metric}: {value:.4f}")

            print(f"\nSuccessful Tests: {summary.get('successful_tests', 0)}")
            print(f"Failed Tests: {summary.get('failed_tests', 0)}")
            if summary.get("best_performing_symbol"):
                print(f"Best Performer: {summary['best_performing_symbol']}")
            if summary.get("worst_performing_symbol"):
                print(f"Worst Performer: {summary['worst_performing_symbol']}")

            print("\n✅ Multi-symbol backtest completed successfully!")
        else:
            results = await backtester.run_backtest(primary_symbol, start_date, end_date, timeframe)

            print("\n=== BACKTEST RESULTS ===")
            print(f"Symbol: {primary_symbol}")
            print(f"Period: {start_date} to {end_date}")
            print(f"Timeframe: {timeframe}")

            if results.get("results", {}).get("aggregated"):
                agg_results = results["results"]["aggregated"]
                print("\nAggregated Signals Performance:")
                profit_factor = agg_results.get('profit_factor', 0.0)
                profit_factor_display = "∞" if math.isinf(profit_factor) else f"{profit_factor:.2f}"
                max_drawdown = agg_results.get('max_drawdown', 0.0)
                print(f"  Total Signals: {agg_results.get('total_signals', 0)}")
                print(f"  Filled Trades: {agg_results.get('filled_trades', 0)}")
                print(f"  Total Trades: {agg_results.get('total_trades', 0)}")
                print(f"  Fill Rate: {agg_results.get('fill_rate_pct', 0.0):.2f}%")
                print(f"  Win Rate: {agg_results.get('win_rate', 0):.1%}")
                print(f"  Profit Factor: {profit_factor_display}")
                print(f"  Sharpe Ratio: {agg_results.get('sharpe_ratio', 0):.2f}")
                print(f"  Expectancy: ${agg_results.get('expectancy', 0):.2f}")
                print(f"  Total PnL: ${agg_results.get('total_pnl', 0.0):.2f}")
                print(f"  Final Balance: ${agg_results.get('final_balance', backtester.config.get('initial_balance', 10000.0)):.2f}")
                print(f"  Max Drawdown: {max_drawdown:.2%}")

            portfolio_summary = backtester.order_manager.get_portfolio_summary()
            print("\nAdvanced Orders Summary:")
            print(f"  Total Orders Created: {portfolio_summary['total_orders']}")
            print(f"  Active Orders: {portfolio_summary['active_orders']}")
            print(f"  Filled Orders: {portfolio_summary['filled_orders']}")
            print(f"  Cancelled Orders: {portfolio_summary['cancelled_orders']}")

            print("\n✅ Backtest with Advanced Orders completed successfully!")
    except Exception as e:
        logger.error("backtest_execution_failed", error=str(e))
        print(f"❌ Backtest failed: {str(e)}")


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Aggregated signals backtest runner")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair to backtest")
    parser.add_argument("--start-date", dest="start_date", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", dest="end_date", help="End date YYYY-MM-DD")
    parser.add_argument("--period", help="Relative period (e.g., 3months, 90days)")
    parser.add_argument("--timeframe", default="1m", help="Kline interval")
    parser.add_argument("--calibrated", action="store_true", help="Use calibrated strategy parameters")
    parser.add_argument("--force-fill", dest="force_fill_simulation", action="store_true",
                        help="Force OHLC-based order fill simulation")
    parser.add_argument("--force-fill-simulation", dest="force_fill_simulation", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--dynamic-tp-sl", action="store_true",
                        help="Enable ATR-driven take-profit / stop-loss overrides")
    parser.add_argument("--optimized-filters", action="store_true",
                        help="Enforce stricter confidence and confirmation requirements")
    parser.add_argument("--relaxed-filters", action="store_true",
                        help="Loosen confirmation/threshold requirements for testing")
    parser.add_argument("--edge-mode", action="store_true",
                        help="Enable edge-mode enrichment, filters, and trade metadata")
    parser.add_argument("--symbols",
                        help="Comma-separated list of symbols to backtest (overrides --symbol)" )
    parser.add_argument("--buckets-analysis", dest="buckets_analysis", action="store_true",
                        help="Run buckets analysis after backtest completion")
    parser.add_argument("--tp-sl-ratio", type=float, default=2.0,
                        help="ATR-based take-profit to stop-loss ratio")
    parser.add_argument("--strategy", help="Comma-separated list of strategies to enable")
    parser.add_argument("--no-mean-reversion", action="store_true",
                        help="Disable the mean reversion strategy")

    args = parser.parse_args()

    try:
        resolved_start, resolved_end = _resolve_backtest_dates(args.period, args.start_date, args.end_date)
    except ValueError as exc:
        logger.error("invalid_period_parameter", error=str(exc))
        raise SystemExit(str(exc)) from exc

    symbols_list = None
    if args.symbols:
        symbols_list = [s.strip() for s in args.symbols.split(',') if s.strip()]

    asyncio.run(main(
        symbol=args.symbol,
        symbols=symbols_list,
        start_date=resolved_start,
        end_date=resolved_end,
        timeframe=args.timeframe,
        calibrated=args.calibrated,
        force_fill_simulation=args.force_fill_simulation,
        dynamic_tp_sl=args.dynamic_tp_sl,
        optimized_filters=args.optimized_filters,
        relaxed_filters=args.relaxed_filters,
        edge_mode=args.edge_mode,
        buckets_analysis=args.buckets_analysis,
        tp_sl_ratio=args.tp_sl_ratio,
        strategy=args.strategy,
        disable_mean_reversion=args.no_mean_reversion
    ))