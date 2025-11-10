#!/bin/bash

# Live monitoring script for Binance collector
# Runs monitor_pilot.py every 30 seconds with screen clearing

cd "/Users/merari/Desktop/bot de scalping"
source .venv/bin/activate

echo "🔴 LIVE MONITORING - Press Ctrl+C to stop"
echo "Refreshing every 30 seconds..."
echo ""

while true; do
    clear
    echo "════════════════════════════════════════════════════════════════════════"
    echo "📊 BINANCE PILOT - LIVE MONITORING"
    echo "════════════════════════════════════════════════════════════════════════"
    echo "🕐 Last Update: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Run monitor
    python backend/scripts/monitor_pilot.py
    
    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo "⏳ Next refresh in 30 seconds... (Ctrl+C to stop)"
    echo "════════════════════════════════════════════════════════════════════════"
    
    sleep 30
done
