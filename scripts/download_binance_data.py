#!/usr/bin/env python3
"""
Script para descargar datos históricos de Binance
Uso: python scripts/download_binance_data.py [SYMBOL] [INTERVAL] [DAYS]
Ejemplo: python scripts/download_binance_data.py BTCUSDT 1m 10
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from binance.client import Client

# Configuración
API_KEY = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_API_SECRET', '')

def download_historical_data(symbol='BTCUSDT', interval='1m', days=10, output_dir='data/raw'):
    """
    Descarga datos históricos de Binance
    
    Args:
        symbol: Par de trading (ej: BTCUSDT, ETHUSDT)
        interval: Intervalo de velas (1m, 5m, 15m, 1h, etc)
        days: Número de días hacia atrás
        output_dir: Directorio de salida
    """
    print(f"📥 Descargando {symbol} - {interval} - últimos {days} días...")
    
    # Inicializar cliente (funciona sin API keys para datos públicos)
    client = Client(api_key=API_KEY, api_secret=API_SECRET)
    
    # Calcular fecha de inicio
    start_date = datetime.now() - timedelta(days=days)
    start_str = start_date.strftime('%Y-%m-%d')
    
    try:
        # Obtener datos históricos
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=f"{start_str} UTC"
        )
        
        # Convertir a DataFrame
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convertir timestamps a datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convertir tipos numéricos
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar CSV
        filename = f"{symbol}_{interval}_{days}d_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        print(f"✅ Datos guardados: {filepath}")
        print(f"📊 Total de registros: {len(df):,}")
        print(f"📅 Rango: {df['open_time'].min()} → {df['open_time'].max()}")
        print(f"💾 Tamaño: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
        
        return filepath
        
    except Exception as e:
        print(f"❌ Error al descargar datos: {e}")
        return None

def main():
    # Argumentos por defecto
    symbol = 'BTCUSDT'
    interval = '1m'
    days = 10
    
    # Parsear argumentos de línea de comandos
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    if len(sys.argv) > 2:
        interval = sys.argv[2]
    if len(sys.argv) > 3:
        days = int(sys.argv[3])
    
    # Descargar datos
    download_historical_data(symbol, interval, days)

if __name__ == '__main__':
    main()
