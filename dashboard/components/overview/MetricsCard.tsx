'use client'

import { useRealtimeMetrics } from '@/hooks/useRealtimeData'

function formatNumber(num: number, decimals = 2): string {
  return num.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

function formatPercent(num: number): string {
  return (num * 100).toFixed(2) + '%'
}

interface MetricItemProps {
  label: string
  value: string
  subValue?: string
  positive?: boolean
  negative?: boolean
}

function MetricItem({ label, value, subValue, positive, negative }: MetricItemProps) {
  const valueColor = positive ? 'text-green-400' : negative ? 'text-red-400' : 'text-white'

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="text-sm text-gray-400 mb-1">{label}</div>
      <div className={`text-2xl font-bold ${valueColor}`}>{value}</div>
      {subValue && <div className="text-xs text-gray-500 mt-1">{subValue}</div>}
    </div>
  )
}

export function MetricsCard() {
  const { metrics, loading } = useRealtimeMetrics()

  if (loading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-gray-800 rounded-lg p-4 animate-pulse">
            <div className="h-4 bg-gray-700 rounded w-20 mb-2"></div>
            <div className="h-8 bg-gray-700 rounded w-24"></div>
          </div>
        ))}
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 text-gray-400">
        No metrics data available
      </div>
    )
  }

  const totalPnL = metrics.total_pnl
  const isProfitable = totalPnL >= 0

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <MetricItem
        label="Total PnL"
        value={`$${formatNumber(totalPnL)}`}
        subValue={`Realized: $${formatNumber(metrics.realized_pnl)}`}
        positive={isProfitable}
        negative={!isProfitable}
      />
      <MetricItem
        label="Win Rate"
        value={formatPercent(metrics.win_rate)}
        subValue={`${metrics.total_trades} trades`}
      />
      <MetricItem
        label="Sharpe Ratio"
        value={formatNumber(metrics.sharpe_ratio)}
        positive={metrics.sharpe_ratio > 1}
        negative={metrics.sharpe_ratio < 0}
      />
      <MetricItem
        label="Max Drawdown"
        value={formatPercent(metrics.max_drawdown)}
        negative={metrics.max_drawdown > 0.1}
      />
    </div>
  )
}
