'use client';

import { useEffect, useState } from 'react';
import { SystemStatusCard } from '@/components/dashboard/system-status';
import { PnLCard } from '@/components/dashboard/pnl-card';
import { RecentSignals } from '@/components/dashboard/recent-signals';
import { TradesTable } from '@/components/dashboard/trades-table';
import { RiskStatusCard } from '@/components/dashboard/risk-status-card';
import { KillSwitchButton } from '@/components/dashboard/kill-switch-button';
import { HealthScoreIndicator } from '@/components/dashboard/health-score-indicator';
import { AggregatedSignalsCard } from '@/components/dashboard/aggregated-signals-card';

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

export default function DashboardPage() {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/system/status');
        if (res.ok) {
          const data = await res.json();
          setSystemStatus(data);
        }
      } catch (error) {
        console.error('Failed to fetch system status:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">TradeSage Expert Dashboard</h1>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600">Mode:</span>
          <span className="px-3 py-1 rounded-full bg-blue-100 text-blue-800 text-sm font-medium">
            {systemStatus?.system?.mode === 'paper' ? 'Paper Trading' : 'Live Trading'}
          </span>
        </div>
      </div>

      {/* Status Grid - 4 columnas */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <SystemStatusCard status={systemStatus} isLoading={isLoading} />
        <PnLCard />
        <RiskStatusCard />
        <HealthScoreIndicator />
      </div>

      {/* Kill-Switch Button - Destacado */}
      <div className="max-w-md mx-auto">
        <KillSwitchButton />
      </div>

      {/* Signals y Trades */}
      <div className="grid gap-4 md:grid-cols-3">
        <RecentSignals />
        <AggregatedSignalsCard />
        <TradesTable />
      </div>
    </div>
  );
}