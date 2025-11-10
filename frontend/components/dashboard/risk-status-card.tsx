'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Shield, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

interface RiskStatus {
  status: {
    trading_enabled: boolean;
    kill_switch_active: boolean;
    kill_switch_reason: string | null;
    consecutive_losses: number;
    in_cooldown: boolean;
  };
  period_pnl: {
    daily: {
      pnl: number;
      pnl_pct: number;
    };
  };
  drawdown: {
    current_pct: number;
    in_drawdown: boolean;
  };
  position_limits: {
    open_positions: number;
    max_positions: number;
    exposure_utilization_pct: number;
  };
}

export function RiskStatusCard() {
  const [risk, setRisk] = useState<RiskStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchRisk = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/risk/status');
        const json = await res.json();
        if (json.success) {
          setRisk(json.data);
        }
      } catch (error) {
        console.error('Failed to fetch risk status:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchRisk();
    const interval = setInterval(fetchRisk, 3000);
    return () => clearInterval(interval);
  }, []);

  if (loading) return <Card><CardContent className="pt-6">Loading...</CardContent></Card>;
  if (!risk) return <Card><CardContent className="pt-6">Failed to load</CardContent></Card>;

  const getStatusIcon = () => {
    if (risk.status.kill_switch_active) return <XCircle className="h-5 w-5 text-red-600" />;
    if (risk.status.in_cooldown) return <AlertTriangle className="h-5 w-5 text-yellow-600" />;
    if (!risk.status.trading_enabled) return <AlertTriangle className="h-5 w-5 text-orange-600" />;
    return <CheckCircle className="h-5 w-5 text-green-600" />;
  };

  const getStatusColor = () => {
    if (risk.status.kill_switch_active) return 'text-red-600';
    if (risk.status.in_cooldown) return 'text-yellow-600';
    if (!risk.status.trading_enabled) return 'text-orange-600';
    return 'text-green-600';
  };

  const getStatusText = () => {
    if (risk.status.kill_switch_active) return 'EMERGENCY STOP';
    if (risk.status.in_cooldown) return 'COOLDOWN';
    if (!risk.status.trading_enabled) return 'PAUSED';
    return 'ACTIVE';
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">Risk Manager</CardTitle>
        <Shield className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-2 mb-4">
          {getStatusIcon()}
          <div className={`text-2xl font-bold ${getStatusColor()}`}>
            {getStatusText()}
          </div>
        </div>

        {risk.status.kill_switch_active && risk.status.kill_switch_reason && (
          <div className="mb-4 p-2 bg-red-50 border border-red-200 rounded">
            <p className="text-xs text-red-700 font-medium">
              {risk.status.kill_switch_reason}
            </p>
          </div>
        )}

        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-xs text-muted-foreground">Daily P&L:</span>
            <span className={`text-sm font-medium ${
              risk.period_pnl.daily.pnl >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              ${risk.period_pnl.daily.pnl.toFixed(2)} ({risk.period_pnl.daily.pnl_pct.toFixed(2)}%)
            </span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-xs text-muted-foreground">Drawdown:</span>
            <div className="flex items-center gap-2">
              <span className={`text-sm font-medium ${
                risk.drawdown.current_pct > 5 ? 'text-orange-600' : 'text-muted-foreground'
              }`}>
                {risk.drawdown.current_pct.toFixed(2)}%
              </span>
              {risk.drawdown.in_drawdown && (
                <Badge variant="destructive" className="text-xs">In DD</Badge>
              )}
            </div>
          </div>

          {risk.status.consecutive_losses > 0 && (
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">Consecutive Losses:</span>
              <Badge variant={risk.status.consecutive_losses >= 3 ? 'destructive' : 'secondary'}>
                {risk.status.consecutive_losses}
              </Badge>
            </div>
          )}

          <div className="flex justify-between items-center">
            <span className="text-xs text-muted-foreground">Positions:</span>
            <span className="text-sm">
              {risk.position_limits.open_positions}/{risk.position_limits.max_positions}
            </span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-xs text-muted-foreground">Exposure:</span>
            <span className="text-sm">
              {risk.position_limits.exposure_utilization_pct.toFixed(1)}%
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
