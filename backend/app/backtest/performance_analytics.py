# backend/app/backtest/performance_analytics.py

from typing import List, Dict, Optional
import pandas as pd
import json
from datetime import datetime
from app.backtest.models import BacktestResult, BacktestTrade
from app.utils.logger import get_logger

logger = get_logger("performance_analytics")

class PerformanceAnalytics:
    """
    Análisis avanzado de performance de backtesting

    Métricas calculadas:
    - Ratios de Sharpe, Sortino, Calmar
    - Maximum Adverse Excursion (MAE)
    - Maximum Favorable Excursion (MFE)
    - Expectancy
    - Recovery Factor
    - Profit per Trade
    - Trade Distribution Analysis
    """

    def __init__(self):
        self.logger = get_logger("performance_analytics")

    def analyze(self, result: BacktestResult) -> Dict:
        """
        Análisis completo de resultados

        Args:
            result: BacktestResult del backtest

        Returns:
            Dict con análisis detallado
        """

        self.logger.info(
            "analyzing_performance",
            strategy=result.strategy_name,
            total_trades=result.total_trades
        )

        if result.total_trades == 0:
            return self._empty_analysis(result)

        analysis = {
            "summary": self._create_summary(result),
            "returns_analysis": self._analyze_returns(result),
            "risk_analysis": self._analyze_risk(result),
            "trade_distribution": self._analyze_trade_distribution(result),
            "time_analysis": self._analyze_time_patterns(result),
            "expectancy": self._calculate_expectancy(result),
            "ratios": self._calculate_ratios(result),
            "equity_curve": self._generate_equity_curve(result)
        }

        self.logger.info(
            "analysis_complete",
            expectancy=analysis["expectancy"]["expectancy"],
            sharpe=analysis["ratios"]["sharpe_ratio"]
        )

        return analysis

    def _empty_analysis(self, result: BacktestResult) -> Dict:
        """Análisis vacío para cuando no hay trades"""
        return {
            "summary": {
                "strategy": result.strategy_name,
                "symbol": result.symbol,
                "period": f"{result.start_date} to {result.end_date}",
                "total_trades": 0,
                "message": "No trades executed during backtest period"
            }
        }

    def _create_summary(self, result: BacktestResult) -> Dict:
        """Crear resumen ejecutivo"""

        roi_pct = ((result.final_balance - result.initial_balance) / result.initial_balance) * 100
        days = (result.end_date - result.start_date).days
        annualized_return = (roi_pct / days * 365) if days > 0 else 0

        return {
            "strategy": result.strategy_name,
            "symbol": result.symbol,
            "period": f"{result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}",
            "duration_days": days,
            "initial_balance": result.initial_balance,
            "final_balance": result.final_balance,
            "total_pnl": result.total_pnl,
            "roi_pct": roi_pct,
            "annualized_return": annualized_return,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "max_drawdown_pct": result.max_drawdown_pct
        }

    def _analyze_returns(self, result: BacktestResult) -> Dict:
        """Análisis de retornos"""

        returns = [trade.pnl for trade in result.trades]
        returns_pct = [trade.pnl_pct for trade in result.trades]

        # Statistics
        avg_return = sum(returns) / len(returns) if returns else 0
        median_return = sorted(returns)[len(returns) // 2] if returns else 0

        # Percentiles
        sorted_returns = sorted(returns)
        p25 = sorted_returns[len(returns) // 4] if returns else 0
        p75 = sorted_returns[3 * len(returns) // 4] if returns else 0

        # Winning vs Losing
        winning = [r for r in returns if r > 0]
        losing = [r for r in returns if r < 0]

        return {
            "avg_return": avg_return,
            "median_return": median_return,
            "std_deviation": self._std_dev(returns),
            "percentile_25": p25,
            "percentile_75": p75,
            "total_wins": sum(winning),
            "total_losses": abs(sum(losing)),
            "avg_win": sum(winning) / len(winning) if winning else 0,
            "avg_loss": abs(sum(losing)) / len(losing) if losing else 0,
            "win_loss_ratio": (sum(winning) / abs(sum(losing))) if losing else 0,
            "largest_win": max(returns) if returns else 0,
            "largest_loss": min(returns) if returns else 0
        }

    def _analyze_risk(self, result: BacktestResult) -> Dict:
        """Análisis de riesgo"""

        returns = [trade.pnl for trade in result.trades]

        # Value at Risk (VaR) - 95% confidence
        sorted_returns = sorted(returns)
        var_95_index = int(len(returns) * 0.05)
        var_95 = sorted_returns[var_95_index] if var_95_index < len(returns) else 0

        # Conditional VaR (CVaR) - average of worst 5%
        worst_5pct = sorted_returns[:var_95_index] if var_95_index > 0 else []
        cvar_95 = sum(worst_5pct) / len(worst_5pct) if worst_5pct else 0

        # Ulcer Index (measure of downside volatility)
        ulcer_index = self._calculate_ulcer_index(result)

        return {
            "max_drawdown": result.max_drawdown,
            "max_drawdown_pct": result.max_drawdown_pct,
            "value_at_risk_95": var_95,
            "conditional_var_95": cvar_95,
            "ulcer_index": ulcer_index,
            "downside_deviation": self._downside_deviation(returns),
            "max_consecutive_losses": result.max_consecutive_losses,
            "recovery_factor": abs(result.total_pnl / result.max_drawdown) if result.max_drawdown != 0 else 0
        }

    def _analyze_trade_distribution(self, result: BacktestResult) -> Dict:
        """Análisis de distribución de trades"""

        returns = [trade.pnl for trade in result.trades]

        # Create bins for distribution
        bins = [-float('inf'), -500, -200, -100, -50, 0, 50, 100, 200, 500, float('inf')]
        distribution = {
            "< -500": 0,
            "-500 to -200": 0,
            "-200 to -100": 0,
            "-100 to -50": 0,
            "-50 to 0": 0,
            "0 to 50": 0,
            "50 to 100": 0,
            "100 to 200": 0,
            "200 to 500": 0,
            "> 500": 0
        }

        labels = list(distribution.keys())

        for ret in returns:
            for i in range(len(bins) - 1):
                if bins[i] <= ret < bins[i + 1]:
                    distribution[labels[i]] += 1
                    break

        return {
            "distribution": distribution,
            "skewness": self._calculate_skewness(returns),
            "kurtosis": self._calculate_kurtosis(returns)
        }

    def _analyze_time_patterns(self, result: BacktestResult) -> Dict:
        """Análisis de patrones temporales"""

        # Trade duration analysis
        durations = []
        for trade in result.trades:
            if trade.exit_timestamp:
                duration = (trade.exit_timestamp - trade.entry_timestamp) / (1000 * 60)  # minutes
                durations.append(duration)

        # Hour of day analysis (when trades were opened)
        hours = {}
        for trade in result.trades:
            dt = datetime.fromtimestamp(trade.entry_timestamp / 1000)
            hour = dt.hour
            if hour not in hours:
                hours[hour] = {"count": 0, "total_pnl": 0}
            hours[hour]["count"] += 1
            hours[hour]["total_pnl"] += trade.pnl

        return {
            "avg_duration_minutes": sum(durations) / len(durations) if durations else 0,
            "min_duration_minutes": min(durations) if durations else 0,
            "max_duration_minutes": max(durations) if durations else 0,
            "trades_by_hour": hours
        }

    def _calculate_expectancy(self, result: BacktestResult) -> Dict:
        """Calcular expectancy (ganancia esperada por trade)"""

        win_rate = result.win_rate / 100
        avg_win = result.avg_win
        avg_loss = abs(result.avg_loss)

        # Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Expectancy per dollar risked
        expectancy_ratio = expectancy / avg_loss if avg_loss > 0 else 0

        return {
            "expectancy": expectancy,
            "expectancy_ratio": expectancy_ratio,
            "interpretation": self._interpret_expectancy(expectancy)
        }

    def _interpret_expectancy(self, expectancy: float) -> str:
        """Interpretar expectancy"""
        if expectancy > 100:
            return "Excellent - Strong positive expectancy"
        elif expectancy > 50:
            return "Very Good - Good positive expectancy"
        elif expectancy > 0:
            return "Acceptable - Positive expectancy"
        elif expectancy > -50:
            return "Poor - Small negative expectancy"
        else:
            return "Very Poor - Strong negative expectancy"

    def _calculate_ratios(self, result: BacktestResult) -> Dict:
        """Calcular ratios de performance"""

        # Calmar Ratio = Annualized Return / Max Drawdown
        days = (result.end_date - result.start_date).days
        annualized_return = (result.total_pnl_pct / days * 365) if days > 0 else 0
        calmar_ratio = annualized_return / result.max_drawdown_pct if result.max_drawdown_pct > 0 else 0

        # Sterling Ratio
        sterling_ratio = annualized_return / (result.max_drawdown_pct + 10) if result.max_drawdown_pct > 0 else 0

        return {
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "sterling_ratio": sterling_ratio,
            "profit_to_drawdown": abs(result.total_pnl / result.max_drawdown) if result.max_drawdown > 0 else 0
        }

    def _generate_equity_curve(self, result: BacktestResult) -> List[Dict]:
        """Generar equity curve"""

        equity_curve = []
        running_balance = result.initial_balance

        for trade in result.trades:
            running_balance += trade.pnl

            equity_curve.append({
                "timestamp": trade.exit_timestamp,
                "datetime": datetime.fromtimestamp(trade.exit_timestamp / 1000).isoformat() if trade.exit_timestamp else None,
                "balance": running_balance,
                "pnl": trade.pnl,
                "trade_number": len(equity_curve) + 1
            })

        return equity_curve

    # Helper methods
    def _std_dev(self, values: List[float]) -> float:
        """Calcular desviación estándar"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def _downside_deviation(self, returns: List[float]) -> float:
        """Calcular downside deviation (solo retornos negativos)"""
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return 0
        return self._std_dev(negative_returns)

    def _calculate_ulcer_index(self, result: BacktestResult) -> float:
        """Calcular Ulcer Index (drawdown volatility)"""
        # Simplified calculation
        if result.max_drawdown_pct == 0:
            return 0
        return (result.max_drawdown_pct ** 2) ** 0.5

    def _calculate_skewness(self, values: List[float]) -> float:
        """Calcular skewness (asimetría)"""
        if len(values) < 3:
            return 0

        mean = sum(values) / len(values)
        std = self._std_dev(values)

        if std == 0:
            return 0

        n = len(values)
        skew = sum(((x - mean) / std) ** 3 for x in values) * n / ((n - 1) * (n - 2))
        return skew

    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calcular kurtosis (peso de las colas)"""
        if len(values) < 4:
            return 0

        mean = sum(values) / len(values)
        std = self._std_dev(values)

        if std == 0:
            return 0

        n = len(values)
        kurt = sum(((x - mean) / std) ** 4 for x in values) / n - 3
        return kurt

    def save_report(
        self,
        analysis: Dict,
        filename: str = None
    ):
        """Guardar reporte en JSON"""

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy = analysis.get("summary", {}).get("strategy", "unknown")
            filename = f"reports/backtest_{strategy}_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        self.logger.info("report_saved", filename=filename)

        return filename

    def print_report(self, analysis: Dict):
        """Imprimir reporte formateado"""

        print("\n" + "=" * 80)
        print("PERFORMANCE ANALYSIS REPORT")
        print("=" * 80)

        # Summary
        summary = analysis["summary"]
        print(f"\n📊 Summary:")
        print(f"   Strategy: {summary['strategy']}")
        print(f"   Symbol: {summary['symbol']}")
        print(f"   Period: {summary['period']}")
        print(f"   Duration: {summary['duration_days']} days")

        print(f"\n💰 Returns:")
        print(f"   Initial Balance: ${summary['initial_balance']:,.2f}")
        print(f"   Final Balance: ${summary['final_balance']:,.2f}")
        print(f"   Total P&L: ${summary['total_pnl']:,.2f}")
        print(f"   ROI: {summary['roi_pct']:.2f}%")
        print(f"   Annualized Return: {summary['annualized_return']:.2f}%")

        print(f"\n📈 Trade Statistics:")
        print(f"   Total Trades: {summary['total_trades']}")
        print(f"   Win Rate: {summary['win_rate']:.2f}%")
        print(f"   Profit Factor: {summary['profit_factor']:.2f}")

        # Expectancy
        expectancy = analysis["expectancy"]
        print(f"\n🎯 Expectancy:")
        print(f"   Expectancy: ${expectancy['expectancy']:.2f}")
        print(f"   Expectancy Ratio: {expectancy['expectancy_ratio']:.2f}")
        print(f"   {expectancy['interpretation']}")

        # Ratios
        ratios = analysis["ratios"]
        print(f"\n📉 Risk-Adjusted Returns:")
        print(f"   Sharpe Ratio: {ratios['sharpe_ratio']:.2f}")
        print(f"   Sortino Ratio: {ratios['sortino_ratio']:.2f}")
        print(f"   Calmar Ratio: {ratios['calmar_ratio']:.2f}")

        # Risk
        risk = analysis["risk_analysis"]
        print(f"\n⚠️ Risk Metrics:")
        print(f"   Max Drawdown: ${risk['max_drawdown']:,.2f} ({risk['max_drawdown_pct']:.2f}%)")
        print(f"   Value at Risk (95%): ${risk['value_at_risk_95']:,.2f}")
        print(f"   Recovery Factor: {risk['recovery_factor']:.2f}")

        print("\n" + "=" * 80)