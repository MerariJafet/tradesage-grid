'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Activity } from 'lucide-react';

interface HealthData {
  health_score: number;
  category: string;
  color: string;
  recommendations: Array<{
    priority: string;
    message: string;
  }>;
}

export function HealthScoreIndicator() {
  const [health, setHealth] = useState<HealthData | null>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/risk/health');
        const json = await res.json();
        if (json.success) {
          setHealth(json.data);
        }
      } catch (error) {
        console.error('Failed to fetch health score:', error);
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  if (!health) return <Card><CardContent className="pt-6">Loading...</CardContent></Card>;

  const getScoreColor = () => {
    if (health.health_score >= 80) return 'text-green-600';
    if (health.health_score >= 60) return 'text-blue-600';
    if (health.health_score >= 40) return 'text-yellow-600';
    if (health.health_score >= 20) return 'text-orange-600';
    return 'text-red-600';
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">System Health</CardTitle>
        <Activity className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <div className="flex items-baseline gap-2 mb-2">
              <div className={`text-3xl font-bold ${getScoreColor()}`}>
                {health.health_score}
              </div>
              <div className="text-sm text-muted-foreground">/ 100</div>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full ${getScoreColor().replace('text-', 'bg-')}`}
                style={{ width: `${health.health_score}%` }}
              />
            </div>
            <div className="text-xs text-muted-foreground mt-1 capitalize">
              {health.category}
            </div>
          </div>

          {health.recommendations.length > 0 && (
            <div className="p-2 bg-muted rounded text-xs">
              <div className="font-medium mb-1">
                {health.recommendations[0].priority === 'critical' && '🔴 Critical'}
                {health.recommendations[0].priority === 'high' && '🟠 High Priority'}
                {health.recommendations[0].priority === 'medium' && '🟡 Medium Priority'}
                {health.recommendations[0].priority === 'low' && '🟢 Info'}
              </div>
              <div className="text-muted-foreground">
                {health.recommendations[0].message}
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
