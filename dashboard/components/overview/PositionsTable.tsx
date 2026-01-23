'use client'

import { useRealtimePositions } from '@/hooks/useRealtimeData'

function formatNumber(num: number, decimals = 2): string {
  return num.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

export function PositionsTable() {
  const { positions, loading } = useRealtimePositions()

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="animate-pulse space-y-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-12 bg-gray-700 rounded"></div>
          ))}
        </div>
      </div>
    )
  }

  if (positions.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 text-gray-400 text-center">
        No open positions
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden">
      <table className="w-full text-sm">
        <thead className="bg-gray-700">
          <tr>
            <th className="px-4 py-3 text-left text-gray-300 font-medium">Asset</th>
            <th className="px-4 py-3 text-right text-gray-300 font-medium">Side</th>
            <th className="px-4 py-3 text-right text-gray-300 font-medium">Size</th>
            <th className="px-4 py-3 text-right text-gray-300 font-medium">Avg Entry</th>
            <th className="px-4 py-3 text-right text-gray-300 font-medium">Unrealized</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-700">
          {positions.map((position) => {
            const unrealizedPnL = position.unrealized_pnl
            const pnlColor = unrealizedPnL >= 0 ? 'text-green-400' : 'text-red-400'

            return (
              <tr key={position.id} className="hover:bg-gray-750">
                <td className="px-4 py-3">
                  <div className="font-medium">{position.slug || position.asset_id.slice(0, 16)}</div>
                </td>
                <td className="px-4 py-3 text-right">
                  <span className={position.side === 'up' ? 'text-green-400' : 'text-red-400'}>
                    {position.side?.toUpperCase()}
                  </span>
                </td>
                <td className="px-4 py-3 text-right">{formatNumber(position.size)}</td>
                <td className="px-4 py-3 text-right">${formatNumber(position.avg_entry_price)}</td>
                <td className={`px-4 py-3 text-right ${pnlColor}`}>
                  ${formatNumber(unrealizedPnL)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
