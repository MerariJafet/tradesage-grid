#!/usr/bin/env python3
"""
Test script for Historical Data Loader (Sprint 8.1)
Tests the BinanceHistoricalDataLoader functionality
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.utils.logger import setup_logging, get_logger
from app.backtest.data_loader import BinanceHistoricalDataLoader

logger = get_logger("test_data_loader")

async def test_data_download():
    """Test downloading historical data from Binance"""
    logger.info("Starting Historical Data Loader test")

    # Initialize the data loader
    loader = BinanceHistoricalDataLoader()

    # Test parameters
    symbol = "BTCUSDT"
    interval = "1m"
    days = 7

    # Calculate date range (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    logger.info("Test parameters", symbol=symbol, interval=interval, days=days,
                start_date=start_date.isoformat(), end_date=end_date.isoformat())

    try:
        # Download data
        logger.info("Downloading historical data...")
        data = await loader.download_klines(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )

        logger.info("Download completed", records=len(data))

        # Validate data
        if len(data) == 0:
            logger.error("No data downloaded")
            return False

        # Check data structure
        first_record = data[0]
        expected_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_fields = [field for field in expected_fields if field not in first_record]

        if missing_fields:
            logger.error("Missing fields in data", missing_fields=missing_fields)
            return False

        logger.info("Data structure validation passed")

        # Check timestamp ordering
        timestamps = [record['timestamp'] for record in data]
        if timestamps != sorted(timestamps):
            logger.error("Data not properly sorted by timestamp")
            return False

        logger.info("Timestamp ordering validation passed")

        # Check date range
        first_timestamp = timestamps[0]
        last_timestamp = timestamps[-1]

        if first_timestamp < start_date.timestamp() * 1000:
            logger.warning("First record is before start date")
        if last_timestamp > end_date.timestamp() * 1000:
            logger.warning("Last record is after end date")

        # Save to CSV for inspection
        csv_path = loader.save_to_csv(data, symbol, interval, start_date, end_date)
        logger.info("Data saved to CSV", path=csv_path)

        # Load from CSV to verify persistence
        loaded_data = loader.load_from_csv(csv_path)
        logger.info("Data loaded from CSV", loaded_records=len(loaded_data))

        if len(loaded_data) != len(data):
            logger.error("CSV persistence failed", original=len(data), loaded=len(loaded_data))
            return False

        logger.info("CSV persistence validation passed")

        # Show sample data
        logger.info("Sample records:")
        for i, record in enumerate(data[:3]):
            logger.info(f"Record {i+1}: {record}")

        logger.info("Last record:", record=data[-1])

        # Calculate some basic statistics
        closes = [float(record['close']) for record in data]
        volumes = [float(record['volume']) for record in data]

        stats = {
            'total_records': len(data),
            'price_range': f"{min(closes):.2f} - {max(closes):.2f}",
            'avg_price': f"{sum(closes)/len(closes):.2f}",
            'total_volume': f"{sum(volumes):.2f}",
            'avg_volume': f"{sum(volumes)/len(volumes):.2f}"
        }

        logger.info("Data statistics", **stats)

        logger.info("✅ Historical Data Loader test PASSED")
        return True

    except Exception as e:
        logger.error("Test failed", error=str(e), exc_info=True)
        return False

async def main():
    """Main test function"""
    setup_logging()

    logger.info("=" * 80)
    logger.info("HISTORICAL DATA LOADER TEST (Sprint 8.1)")
    logger.info("=" * 80)

    success = await test_data_download()

    logger.info("=" * 80)
    if success:
        logger.info("🎉 ALL TESTS PASSED - Historical Data Loader is ready!")
        sys.exit(0)
    else:
        logger.error("❌ TESTS FAILED - Check the logs above for details")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())