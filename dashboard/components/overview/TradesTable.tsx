'use client'

import { useRealtimeTrades } from '@/hooks/useRealtimeData'
import { format } from 'date-fns'

function formatNumber(num: number, decimals = 2): string {
  return num.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

export function TradesTable() {
  const { trades, loading } = useRealtimeTrades(20)

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="animate-pulse space-y-4">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-10 bg-gray-700 rounded"></div>
          ))}
        </div>
      </div>
    )
  }

  if (trades.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 text-gray-400 text-center">
        No trades yet
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden max-h-96 overflow-y-auto">
      <table className="w-full text-sm">
        <thead className="bg-gray-700 sticky top-0">
          <tr>
            <th className="px-4 py-3 text-left text-gray-300 font-medium">Time</th>
            <th className="px-4 py-3 text-left text-gray-300 font-medium">Market</th>
            <th className="px-4 py-3 text-center text-gray-300 font-medium">Side</th>
            <th className="px-4 py-3 text-right text-gray-300 font-medium">Price</th>
            <th className="px-4 py-3 text-right text-gray-300 font-medium">Size</th>
            <th className="px-4 py-3 text-right text-gray-300 font-medium">PnL</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-700">
          {trades.map((trade) => {
            const isBuy = trade.side === 1
            const sideColor = isBuy ? 'text-green-400' : 'text-red-400'
            const pnlColor = trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'

            return (
              <tr key={trade.id} className="hover:bg-gray-750">
                <td className="px-4 py-2 text-gray-400 text-xs">
                  {format(new Date(trade.timestamp), 'HH:mm:ss')}
                </td>
                <td className="px-4 py-2 truncate max-w-32" title={trade.slug}>
                  {trade.slug.slice(0, 20)}
                </td>
                <td className={`px-4 py-2 text-center ${sideColor}`}>
                  {isBuy ? 'BUY' : 'SELL'}
                </td>
                <td className="px-4 py-2 text-right">${formatNumber(trade.price)}</td>
                <td className="px-4 py-2 text-right">{formatNumber(trade.size)}</td>
                <td className={`px-4 py-2 text-right ${pnlColor}`}>
                  ${formatNumber(trade.pnl)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
