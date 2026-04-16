import json
import os
from datetime import datetime

from binance_client import BinanceFuturesClient
from config import Config


class TradePerformanceAnalyzer:
    def __init__(self, symbols=None, income_limit=500):
        self.symbols = symbols or Config.TOP_20_SYMBOLS
        self.income_limit = income_limit
        self.client = BinanceFuturesClient()
        self.reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(self.reports_dir, exist_ok=True)

    def analyze_symbol(self, symbol):
        records = self.client.get_income_history(limit=self.income_limit, symbol=symbol) or []
        realized = [r for r in records if str(r.get("incomeType", "")).upper() == "REALIZED_PNL"]

        wins = 0
        losses = 0
        breakeven = 0
        total_realized_pnl = 0.0

        for row in realized:
            income = float(row.get("income", 0.0))
            total_realized_pnl += income
            if income > 0:
                wins += 1
            elif income < 0:
                losses += 1
            else:
                breakeven += 1

        trade_count = wins + losses + breakeven
        win_rate = (wins / trade_count) * 100 if trade_count else 0.0
        avg_pnl = total_realized_pnl / trade_count if trade_count else 0.0

        return {
            "symbol": symbol,
            "trade_count": trade_count,
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "win_rate": round(win_rate, 2),
            "total_realized_pnl": round(total_realized_pnl, 6),
            "avg_pnl": round(avg_pnl, 6),
        }

    def build_report(self):
        symbol_stats = [self.analyze_symbol(symbol) for symbol in self.symbols]

        total_trades = sum(s["trade_count"] for s in symbol_stats)
        total_wins = sum(s["wins"] for s in symbol_stats)
        total_losses = sum(s["losses"] for s in symbol_stats)
        total_realized_pnl = sum(s["total_realized_pnl"] for s in symbol_stats)
        overall_win_rate = (total_wins / total_trades) * 100 if total_trades else 0.0

        report = {
            "generated_at_utc": datetime.utcnow().isoformat(),
            "symbols_analyzed": len(symbol_stats),
            "summary": {
                "total_trades": total_trades,
                "total_wins": total_wins,
                "total_losses": total_losses,
                "overall_win_rate": round(overall_win_rate, 2),
                "total_realized_pnl": round(total_realized_pnl, 6),
            },
            "symbols": sorted(
                symbol_stats,
                key=lambda x: (x["total_realized_pnl"], x["win_rate"], x["trade_count"]),
                reverse=True,
            ),
        }
        return report

    def save_report(self, report):
        latest_path = os.path.join(self.reports_dir, "trade_performance_latest.json")
        dated_path = os.path.join(
            self.reports_dir,
            f"trade_performance_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        with open(dated_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return latest_path, dated_path

