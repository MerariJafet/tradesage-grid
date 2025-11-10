export function RecentSignals() {
  // Mock data for recent signals
  const signals = [
    { id: 'SIG001', symbol: 'BTCUSDT', action: 'BUY', confidence: 0.85, time: '14:32:15' },
    { id: 'SIG002', symbol: 'ETHUSDT', action: 'SELL', confidence: 0.72, time: '14:28:42' },
    { id: 'SIG003', symbol: 'BTCUSDT', action: 'BUY', confidence: 0.91, time: '14:25:18' },
  ];

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Recent Signals</h3>
        <span className="text-2xl">📊</span>
      </div>

      <div className="space-y-3">
        {signals.map((signal) => (
          <div key={signal.id} className="flex items-center justify-between p-3 bg-gray-50 rounded">
            <div>
              <div className="font-medium">{signal.symbol}</div>
              <div className="text-sm text-gray-600">{signal.time}</div>
            </div>
            <div className="text-right">
              <div className={`font-semibold ${signal.action === 'BUY' ? 'text-green-600' : 'text-red-600'}`}>
                {signal.action}
              </div>
              <div className="text-sm text-gray-600">
                {(signal.confidence * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}