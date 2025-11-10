#!/usr/bin/env python3
"""
Test Binance API connection script.
Tests authentication and basic API connectivity.
"""
import sys
import time
import hmac
import hashlib
import urllib.request
import json
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent))

def create_signature(query_string, secret):
    """Create HMAC SHA256 signature for Binance API."""
    return hmac.new(secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

async def test_binance_connection():
    """Test Binance API connection."""
    try:
        from app.config import settings

        # Determine base URL
        base_url = "https://testnet.binance.vision" if settings.BINANCE_TESTNET else "https://api.binance.com"

        # Test 1: Ping
        print("🔄 Testing Binance API ping...")
        try:
            with urllib.request.urlopen(f"{base_url}/api/v3/ping", timeout=10) as response:
                if response.status == 200:
                    print("✅ Ping successful!")
                else:
                    print(f"❌ Ping failed with status {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Ping failed: {e}")
            return False

        # Test 2: Server time
        print("🔄 Testing server time...")
        try:
            with urllib.request.urlopen(f"{base_url}/api/v3/time", timeout=10) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    server_time = data.get('serverTime')
                    local_time = int(time.time() * 1000)
                    diff = abs(local_time - server_time)
                    print(f"✅ Server time: {server_time}")
                    print(f"   Local time: {local_time}")
                    print(f"   Difference: {diff}ms")
                else:
                    print(f"❌ Server time failed with status {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Server time failed: {e}")
            return False

        # Test 3: Account info (authenticated)
        print("🔄 Testing authenticated request (account info)...")
        try:
            # Create timestamp
            timestamp = int(time.time() * 1000)
            query_string = f"timestamp={timestamp}"
            signature = create_signature(query_string, settings.BINANCE_API_SECRET)

            url = f"{base_url}/api/v3/account?{query_string}&signature={signature}"
            req = urllib.request.Request(url)
            req.add_header('X-MBX-APIKEY', settings.BINANCE_API_KEY)

            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    if 'accountType' in data:
                        print("✅ Authentication successful!")
                        print(f"   Account Type: {data.get('accountType')}")
                        print(f"   Can Trade: {data.get('canTrade')}")
                        print(f"   Can Deposit: {data.get('canDeposit')}")
                        return True
                    else:
                        print("❌ Invalid account response")
                        return False
                else:
                    print(f"❌ Authentication failed with status {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False

    except Exception as e:
        print(f"❌ Binance connection error: {e}")
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_binance_connection())
    sys.exit(0 if success else 1)