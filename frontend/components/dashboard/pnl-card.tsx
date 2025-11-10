interface PnLData {
  total_pnl: number;
  pnl_percent: number;
  initial_balance: number;
  current_balance: number;
  today_pnl: number;
  open_positions: number;
}

export function PnLCard() {
  // Mock data for now - in real implementation, this would fetch from API
  const pnlData: PnLData = {
    total_pnl: 1250.75,
    pnl_percent: 12.5,
    initial_balance: 10000,
    current_balance: 11250.75,
    today_pnl: 450.25,
    open_positions: 2
  };

  const isPositive = pnlData.total_pnl >= 0;
  const pnlColor = isPositive ? 'text-green-600' : 'text-red-600';
  const pnlIcon = isPositive ? '📈' : '📉';

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">P&L Overview</h3>
        <span className="text-2xl">{pnlIcon}</span>
      </div>

      <div className={`text-3xl font-bold mb-2 ${pnlColor}`}>
        ${pnlData.total_pnl.toFixed(2)}
      </div>

      <div className={`text-lg mb-4 ${pnlColor}`}>
        {isPositive ? '+' : ''}{pnlData.pnl_percent.toFixed(2)}%
      </div>

      <div className="space-y-1 text-sm text-gray-600">
        <p>Balance: ${pnlData.current_balance.toFixed(2)}</p>
        <p>Today: ${pnlData.today_pnl.toFixed(2)}</p>
        <p>Open Positions: {pnlData.open_positions}</p>
      </div>
    </div>
  );
}