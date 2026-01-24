'use client'

import { useRealtimeEquity } from '@/hooks/useRealtimeData'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { format } from 'date-fns'

export function EquityChart() {
  const { equityHistory, loading } = useRealtimeEquity()

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 h-80 flex items-center justify-center">
        <div className="text-gray-400">Loading equity data...</div>
      </div>
    )
  }

  if (equityHistory.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 h-80 flex items-center justify-center">
        <div className="text-gray-400">No equity data available yet</div>
      </div>
    )
  }

  const chartData = equityHistory.map((snap) => ({
    timestamp: new Date(snap.timestamp).getTime(),
    equity: snap.equity,
    cash: snap.cash,
    positionValue: snap.position_value,
  }))

  const minEquity = Math.min(...chartData.map((d) => d.equity))
  const maxEquity = Math.max(...chartData.map((d) => d.equity))
  const yDomain = [
    Math.floor(minEquity * 0.95),
    Math.ceil(maxEquity * 1.05),
  ]

  return (
    <div className="bg-gray-800 rounded-lg p-4 h-80">
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
            domain={yDomain}
            tickFormatter={(v) => `$${v.toLocaleString()}`}
            stroke="#9CA3AF"
            fontSize={12}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1F2937',
              border: '1px solid #374151',
              borderRadius: '8px',
            }}
            labelFormatter={(ts) => format(new Date(ts), 'MMM dd HH:mm:ss')}
            formatter={(value: number) => [`$${value.toLocaleString()}`, 'Equity']}
          />
          <Line
            type="monotone"
            dataKey="equity"
            stroke="#22C55E"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
