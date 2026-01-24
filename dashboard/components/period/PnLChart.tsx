'use client'

import { MarketSnapshot, Trade } from '@/lib/supabase'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { format } from 'date-fns'

interface PnLChartProps {
  snapshots: MarketSnapshot[]
  trades: Trade[]
}

export function PnLChart({ snapshots, trades }: PnLChartProps) {
  if (snapshots.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 h-64 flex items-center justify-center">
        <div className="text-gray-400">No data available</div>
      </div>
    )
  }

  // Calculate cumulative PnL from trades
  let cumulativePnL = 0
  const pnlByTime = new Map<number, number>()

  trades.forEach((trade) => {
    cumulativePnL += trade.pnl
    const ts = new Date(trade.timestamp).getTime()
    pnlByTime.set(ts, cumulativePnL)
  })

  // Combine with snapshots for continuous line
  const chartData = snapshots.map((snap) => {
    const ts = new Date(snap.timestamp).getTime()
    // Find the most recent PnL at or before this timestamp
    let pnl = 0
    for (const [tradeTs, tradePnl] of pnlByTime) {
      if (tradeTs <= ts) {
        pnl = tradePnl
      }
    }
    return {
      timestamp: ts,
      pnl: snap.period_pnl || pnl,
    }
  })

  const minPnL = Math.min(...chartData.map((d) => d.pnl), 0)
  const maxPnL = Math.max(...chartData.map((d) => d.pnl), 0)
  const yPadding = Math.max(Math.abs(maxPnL - minPnL) * 0.1, 1)

  return (
    <div className="bg-gray-800 rounded-lg p-4 h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="timestamp"
            tickFormatter={(ts) => format(new Date(ts), 'HH:mm')}
            stroke="#9CA3AF"
            fontSize={12}
          />
          <YAxis
            domain={[minPnL - yPadding, maxPnL + yPadding]}
            tickFormatter={(v) => `$${v.toFixed(0)}`}
            stroke="#9CA3AF"
            fontSize={12}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1F2937',
              border: '1px solid #374151',
              borderRadius: '8px',
            }}
            labelFormatter={(ts) => format(new Date(ts), 'HH:mm:ss')}
            formatter={(value: number) => [`$${value.toFixed(2)}`, 'PnL']}
          />
          <ReferenceLine y={0} stroke="#6B7280" strokeDasharray="3 3" />
          <Line
            type="monotone"
            dataKey="pnl"
            stroke="#22C55E"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
