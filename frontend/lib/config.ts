/**
 * Frontend configuration
 * Centralizes all environment variables and configuration
 */

export const config = {
  // API endpoints
  backendUrl: process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000',
  wsUrl: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000',

  // Feature flags
  debug: process.env.NEXT_PUBLIC_DEBUG === 'true',

  // Refresh intervals (in milliseconds)
  refreshIntervals: {
    status: parseInt(process.env.NEXT_PUBLIC_STATUS_REFRESH_INTERVAL || '5000', 10),
    trades: parseInt(process.env.NEXT_PUBLIC_TRADES_REFRESH_INTERVAL || '3000', 10),
  },

  // API endpoints
  api: {
    systemStatus: '/api/system/status',
    dashboard: '/api/dashboard',
    strategies: '/api/strategies',
    risk: '/api/risk',
    signals: '/api/signals',
    health: '/health',
  },
} as const;

export default config;
