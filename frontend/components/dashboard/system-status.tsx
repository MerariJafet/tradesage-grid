interface SystemStatus {
  binance: {
    status: 'connected' | 'disconnected' | 'reconnecting';
    latency_ms: number;
    reconnects: number;
    last_ping: string;
  };
  system: {
    uptime_seconds: number;
    mode: 'paper' | 'live';
    start_time: string;
  };
  database: {
    status: string;
    last_write: string;
  };
}

interface SystemStatusCardProps {
  status: SystemStatus | null;
  isLoading: boolean;
}

export function SystemStatusCard({ status, isLoading }: SystemStatusCardProps) {
  if (isLoading) {
    return (
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
          <div className="h-8 bg-gray-200 rounded w-1/2"></div>
        </div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-2">System Status</h3>
        <p className="text-red-600">Unable to connect to backend</p>
      </div>
    );
  }

  const statusColor = status.binance.status === 'connected' ? 'text-green-600' : 'text-red-600';
  const statusDot = status.binance.status === 'connected' ? '🟢' : '🔴';

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">WebSocket Status</h3>
        <span className="text-2xl">📡</span>
      </div>

      <div className={`text-2xl font-bold mb-2 ${statusColor}`}>
        {statusDot} {status.binance.status}
      </div>

      <div className="space-y-1 text-sm text-gray-600">
        <p>Latency: {status.binance.latency_ms}ms</p>
        <p>Uptime: {Math.floor(status.system.uptime_seconds / 60)}min</p>
        <p>Reconnects: {status.binance.reconnects}</p>
      </div>
    </div>
  );
}