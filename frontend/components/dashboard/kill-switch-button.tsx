'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { AlertTriangle, Play, Loader2 } from 'lucide-react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';

export function KillSwitchButton() {
  const [isActive, setIsActive] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/risk/status');
        const json = await res.json();
        if (json.success) {
          setIsActive(json.data.status.kill_switch_active);
        }
      } catch (error) {
        console.error('Failed to check kill-switch status:', error);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleActivate = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/risk/kill-switch/activate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: 'Manual emergency stop from dashboard' })
      });
      
      const json = await res.json();
      if (json.success) {
        setIsActive(true);
        alert('Kill-Switch Activated: All trading operations stopped.');
      }
    } catch (error) {
      alert('Failed to activate kill-switch');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/risk/kill-switch/reset', {
        method: 'POST'
      });
      
      const json = await res.json();
      if (json.success) {
        setIsActive(false);
        alert('Trading Resumed: Kill-switch has been reset.');
      }
    } catch (error) {
      alert('Failed to reset kill-switch');
    } finally {
      setLoading(false);
    }
  };

  if (isActive) {
    return (
      <AlertDialog>
        <AlertDialogTrigger asChild>
          <Button variant="default" className="w-full bg-green-600 hover:bg-green-700" disabled={loading}>
            {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
            Resume Trading
          </Button>
        </AlertDialogTrigger>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Resume Trading Operations?</AlertDialogTitle>
            <AlertDialogDescription>
              This will reset the kill-switch and resume all trading operations.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleReset} className="bg-green-600 hover:bg-green-700">
              Resume Trading
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    );
  }

  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button variant="destructive" className="w-full" disabled={loading}>
          {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <AlertTriangle className="mr-2 h-4 w-4" />}
          Emergency Stop
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>⚠️ Activate Emergency Stop?</AlertDialogTitle>
          <AlertDialogDescription>
            This will immediately stop all trading operations. Only use in emergency situations!
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction onClick={handleActivate} className="bg-red-600 hover:bg-red-700">
            Emergency Stop
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
