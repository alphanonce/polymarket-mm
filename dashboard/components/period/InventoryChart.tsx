'use client'

import { MarketSnapshot } from '@/lib/supabase'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { format } from 'date-fns'

interface InventoryChartProps {
  snapshots: MarketSnapshot[]
}

export function InventoryChart({ snapshots }: InventoryChartProps) {
  if (snapshots.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 h-64 flex items-center justify-center">
        <div className="text-gray-400">No data available</div>
      </div>
    )
  }

  const chartData = snapshots.map((snap) => ({
    timestamp: new Date(snap.timestamp).getTime(),
    inventory: snap.inventory,
    inventoryValue: snap.inventory_value,
  }))

  const maxInventory = Math.max(...chartData.map((d) => Math.abs(d.inventory)), 1)
  const yDomain = [-maxInventory * 1.1, maxInventory * 1.1]

  return (
    <div className="bg-gray-800 rounded-lg p-4 h-64">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="timestamp"
            tickFormatter={(ts) => format(new Date(ts), 'HH:mm')}
            stroke="#9CA3AF"
            fontSize={12}
          />
          <YAxis
            domain={yDomain}
            tickFormatter={(v) => v.toFixed(1)}
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
            formatter={(value: number) => [value.toFixed(2), 'Inventory']}
          />
          <ReferenceLine y={0} stroke="#6B7280" strokeDasharray="3 3" />
          <defs>
            <linearGradient id="inventoryGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#3B82F6" stopOpacity={0.1}/>
            </linearGradient>
          </defs>
          <Area
            type="monotone"
            dataKey="inventory"
            stroke="#3B82F6"
            fillOpacity={1}
            fill="url(#inventoryGradient)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
