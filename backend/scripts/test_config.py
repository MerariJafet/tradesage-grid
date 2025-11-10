#!/usr/bin/env python3
"""
Test configuration validation script.
Validates that all required environment variables are set and valid.
"""
import os
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_config():
    """Test that configuration loads correctly."""
    try:
        # Import config
        from app.config import settings

        # Check required settings
        required_settings = [
            'MODE',
            'BINANCE_API_KEY',
            'BINANCE_API_SECRET',
            'BINANCE_TESTNET',
            'POSTGRES_PASSWORD',
            'REDIS_HOST',
            'REDIS_PORT',
            'LOG_LEVEL'
        ]

        missing = []
        for setting in required_settings:
            value = getattr(settings, setting, None)
            if value is None or str(value).strip() == '':
                missing.append(setting)

        if missing:
            print(f"❌ Missing required settings: {', '.join(missing)}")
            return False

        # Validate specific values
        if settings.BINANCE_TESTNET not in [True, False]:
            print("❌ BINANCE_TESTNET must be True or False")
            return False

        if not isinstance(settings.REDIS_PORT, int):
            print("❌ REDIS_PORT must be an integer")
            return False

        print("✅ Configuration validation passed")
        print(f"   Mode: {settings.MODE}")
        print(f"   Testnet: {settings.BINANCE_TESTNET}")
        print(f"   Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        return True

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

if __name__ == "__main__":
    success = test_config()
    sys.exit(0 if success else 1)