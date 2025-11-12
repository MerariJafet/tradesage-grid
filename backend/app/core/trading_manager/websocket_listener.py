import json
import time
import websocket


def on_message(ws, message):
    data = json.loads(message)
    print('Price Update:', data.get('p'))


def on_error(ws, error):
    print('Error:', error)


def on_close(ws, close_status_code, close_msg):
    print('Closed connection')


def on_open(ws):
    print('Connected to Binance stream')


if __name__ == '__main__':
    socket = 'wss://stream.binance.com:9443/ws/btcusdt@trade'
    ws = websocket.WebSocketApp(
        socket,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.on_open = on_open
    ws.run_forever()
