'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { TrendingUp, TrendingDown, Activity } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

interface AggregatedSignal {
  timestamp: string;
  signal: {
    symbol: string;
    action: string;
    confidence: number;
    entry_price: number;
    stop_loss: number;
    take_profit: number;
    quantity: number;
    reason: string;
    source_signals: string[];
  };
  source_signals: Array<{
    strategy_name: string;
    confidence: number;
  }>;
}

export function AggregatedSignalsCard() {
  const [signals, setSignals] = useState<AggregatedSignal[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSignals = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/signals/aggregated?limit=10');
        const json = await res.json();
        if (json.success) {
          setSignals(json.data.signals);
        }
      } catch (error) {
        console.error('Failed to fetch aggregated signals:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchSignals();
    const interval = setInterval(fetchSignals, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Aggregated Signals
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground">Loading...</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Activity className="h-4 w-4" />
          Aggregated Signals
        </CardTitle>
      </CardHeader>
      <CardContent>
        {signals.length === 0 ? (
          <div className="text-sm text-muted-foreground">No recent signals</div>
        ) : (
          <div className="space-y-3">
            {signals.slice(0, 5).map((item, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center gap-3">
                  {item.signal.action === 'BUY' ? (
                    <TrendingUp className="h-5 w-5 text-green-600" />
                  ) : (
                    <TrendingDown className="h-5 w-5 text-red-600" />
                  )}
                  <div>
                    <div className="font-medium text-sm">{item.signal.symbol}</div>
                    <div className="text-xs text-muted-foreground">
                      {item.signal.source_signals.join(' + ')}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">
                      Entry: ${item.signal.entry_price} | SL: ${item.signal.stop_loss}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant={item.signal.action === 'BUY' ? 'default' : 'destructive'}>
                    {item.signal.action}
                  </Badge>
                  <span className="text-sm font-medium">
                    {(item.signal.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}