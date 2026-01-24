'use client'

import { Market, MarketSnapshot, Trade } from '@/lib/supabase'

interface PeriodSummaryProps {
  market: Market | null
  trades: Trade[]
  snapshots: MarketSnapshot[]
}

function formatNumber(num: number, decimals = 2): string {
  return num.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

export function PeriodSummary({ market, trades, snapshots }: PeriodSummaryProps) {
  // Calculate summary stats
  const totalPnL = trades.reduce((sum, t) => sum + t.pnl, 0)
  const buyTrades = trades.filter((t) => t.side === 1)
  const sellTrades = trades.filter((t) => t.side === -1)

  const avgBuyPrice = buyTrades.length > 0
    ? buyTrades.reduce((sum, t) => sum + t.price, 0) / buyTrades.length
    : 0

  const avgSellPrice = sellTrades.length > 0
    ? sellTrades.reduce((sum, t) => sum + t.price, 0) / sellTrades.length
    : 0

  const totalVolume = trades.reduce((sum, t) => sum + t.size, 0)

  // Get spread stats from snapshots
  const spreads = snapshots.filter((s) => s.spread && s.spread > 0).map((s) => s.spread!)
  const avgSpread = spreads.length > 0 ? spreads.reduce((a, b) => a + b, 0) / spreads.length : 0

  const pnlColor = totalPnL >= 0 ? 'text-green-400' : 'text-red-400'

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="text-sm text-gray-400 mb-1">Period PnL</div>
        <div className={`text-2xl font-bold ${pnlColor}`}>
          ${formatNumber(market?.period_pnl || totalPnL)}
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-4">
        <div className="text-sm text-gray-400 mb-1">Total Trades</div>
        <div className="text-2xl font-bold">{trades.length}</div>
        <div className="text-xs text-gray-500 mt-1">
          {buyTrades.length} buys / {sellTrades.length} sells
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-4">
        <div className="text-sm text-gray-400 mb-1">Avg Prices</div>
        <div className="text-lg">
          <span className="text-green-400">${formatNumber(avgBuyPrice, 4)}</span>
          {' / '}
          <span className="text-red-400">${formatNumber(avgSellPrice, 4)}</span>
        </div>
        <div className="text-xs text-gray-500 mt-1">buy / sell</div>
      </div>

      <div className="bg-gray-800 rounded-lg p-4">
        <div className="text-sm text-gray-400 mb-1">Volume / Avg Spread</div>
        <div className="text-lg font-bold">{formatNumber(totalVolume)}</div>
        <div className="text-xs text-gray-500 mt-1">
          Spread: ${formatNumber(avgSpread, 4)}
        </div>
      </div>
    </div>
  )
}
