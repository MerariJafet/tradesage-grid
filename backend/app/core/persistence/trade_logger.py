"""Trade logging utilities for realtime persistence to CSV and SQLite."""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Optional
import datetime as dt

PERSISTENCE_DIR = Path(__file__).resolve().parent
DB_PATH = PERSISTENCE_DIR / "trades.db"
CSV_PATH = PERSISTENCE_DIR / "trades.csv"


class TradeLogger:
    """Handle dual logging of trades to CSV and SQLite."""

    def __init__(self) -> None:
        PERSISTENCE_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header()
        self._init_db()

    def _ensure_csv_header(self) -> None:
        if CSV_PATH.exists():
            return
        with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "symbol", "price", "qty", "pnl", "exposure"])

    def _init_db(self) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    qty REAL NOT NULL,
                    pnl REAL NOT NULL,
                    exposure REAL NOT NULL
                )
                """
            )
            conn.commit()

    def log_trade(
        self,
        symbol: str,
        price: float,
        qty: float,
        pnl: float,
        exposure: float,
        *,
        timestamp: Optional[str] = None,
    ) -> None:
        ts = timestamp or dt.datetime.now(dt.timezone.utc).isoformat()
        row = (ts, symbol, float(price), float(qty), float(pnl), float(exposure))
        with CSV_PATH.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(row)
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO trades (timestamp, symbol, price, qty, pnl, exposure) VALUES (?, ?, ?, ?, ?, ?)",
                row,
            )
            conn.commit()
        print(
            f"[LOG] {symbol} | price={float(price):.2f} | qty={float(qty):.6f} "
            f"| pnl={float(pnl):.2f} | exposure={float(exposure):.4f}"
        )
