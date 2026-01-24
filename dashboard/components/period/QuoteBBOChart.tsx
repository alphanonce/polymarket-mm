'use client'

import { MarketSnapshot } from '@/lib/supabase'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { format } from 'date-fns'

interface QuoteBBOChartProps {
  snapshots: MarketSnapshot[]
}

export function QuoteBBOChart({ snapshots }: QuoteBBOChartProps) {
  if (snapshots.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 h-64 flex items-center justify-center">
        <div className="text-gray-400">No data available</div>
      </div>
    )
  }

  const chartData = snapshots.map((snap) => ({
    timestamp: new Date(snap.timestamp).getTime(),
    bestBid: snap.best_bid,
    bestAsk: snap.best_ask,
    midPrice: snap.mid_price,
    ourBid: snap.our_bid && snap.our_bid > 0 ? snap.our_bid : null,
    ourAsk: snap.our_ask && snap.our_ask > 0 ? snap.our_ask : null,
  }))

  // Calculate y-axis domain
  const allPrices = chartData.flatMap((d) => [
    d.bestBid,
    d.bestAsk,
    d.ourBid,
    d.ourAsk,
  ].filter((p): p is number => p !== null && p !== undefined && p > 0))

  if (allPrices.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 h-64 flex items-center justify-center">
        <div className="text-gray-400">No price data available</div>
      </div>
    )
  }

  const minPrice = Math.min(...allPrices)
  const maxPrice = Math.max(...allPrices)
  const padding = (maxPrice - minPrice) * 0.1 || 0.01

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
            domain={[minPrice - padding, maxPrice + padding]}
            tickFormatter={(v) => `$${v.toFixed(3)}`}
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
            formatter={(value: number | null) => value ? [`$${value.toFixed(4)}`, ''] : ['-', '']}
          />
          <Legend />
          <Line
            type="stepAfter"
            dataKey="bestBid"
            name="Best Bid"
            stroke="#22C55E"
            strokeWidth={1}
            dot={false}
            opacity={0.7}
          />
          <Line
            type="stepAfter"
            dataKey="bestAsk"
            name="Best Ask"
            stroke="#EF4444"
            strokeWidth={1}
            dot={false}
            opacity={0.7}
          />
          <Line
            type="stepAfter"
            dataKey="ourBid"
            name="Our Bid"
            stroke="#10B981"
            strokeWidth={2}
            dot={false}
            strokeDasharray="5 5"
          />
          <Line
            type="stepAfter"
            dataKey="ourAsk"
            name="Our Ask"
            stroke="#F59E0B"
            strokeWidth={2}
            dot={false}
            strokeDasharray="5 5"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
