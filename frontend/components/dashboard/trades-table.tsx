export function TradesTable() {
  // Mock data for recent trades
  const trades = [
    {
      id: 'ORD001',
      symbol: 'BTCUSDT',
      side: 'BUY',
      quantity: 0.016,
      price: 62040.82,
      status: 'FILLED',
      time: '14:32:15'
    },
    {
      id: 'ORD002',
      symbol: 'ETHUSDT',
      side: 'SELL',
      quantity: 0.5,
      price: 3450.25,
      status: 'FILLED',
      time: '14:28:42'
    },
    {
      id: 'ORD003',
      symbol: 'BTCUSDT',
      side: 'BUY',
      quantity: 0.008,
      price: 61980.15,
      status: 'PARTIALLY_FILLED',
      time: '14:25:18'
    },
  ];

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Recent Trades</h3>
        <span className="text-2xl">💼</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b">
              <th className="text-left py-2">Symbol</th>
              <th className="text-left py-2">Side</th>
              <th className="text-right py-2">Quantity</th>
              <th className="text-right py-2">Price</th>
              <th className="text-center py-2">Status</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((trade) => (
              <tr key={trade.id} className="border-b">
                <td className="py-2">{trade.symbol}</td>
                <td className={`py-2 font-medium ${trade.side === 'BUY' ? 'text-green-600' : 'text-red-600'}`}>
                  {trade.side}
                </td>
                <td className="py-2 text-right">{trade.quantity}</td>
                <td className="py-2 text-right">${trade.price.toFixed(2)}</td>
                <td className="py-2 text-center">
                  <span className={`px-2 py-1 rounded text-xs ${
                    trade.status === 'FILLED' ? 'bg-green-100 text-green-800' :
                    trade.status === 'PARTIALLY_FILLED' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {trade.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}