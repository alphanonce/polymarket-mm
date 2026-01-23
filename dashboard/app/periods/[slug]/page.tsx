'use client'

import { usePeriodData } from '@/hooks/usePeriodData'
import { PnLChart } from '@/components/period/PnLChart'
import { QuoteBBOChart } from '@/components/period/QuoteBBOChart'
import { InventoryChart } from '@/components/period/InventoryChart'
import { PeriodSummary } from '@/components/period/PeriodSummary'
import Link from 'next/link'
import { use } from 'react'

interface PageProps {
  params: Promise<{ slug: string }>
}

export default function PeriodDetailPage({ params }: PageProps) {
  const { slug } = use(params)
  const { market, snapshots, trades, loading } = usePeriodData(slug)

  if (loading) {
    return (
      <div className="space-y-8">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-gray-700 rounded w-64"></div>
          <div className="h-64 bg-gray-700 rounded"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <Link
            href="/"
            className="text-sm text-blue-400 hover:text-blue-300 mb-2 inline-block"
          >
            &larr; Back to Dashboard
          </Link>
          <h1 className="text-2xl font-bold">{slug}</h1>
          {market && (
            <div className="text-sm text-gray-400 mt-1">
              {market.asset.toUpperCase()} / {market.timeframe}
              {market.status === 'resolved' && market.outcome && (
                <span className={`ml-2 ${market.outcome === 'up' ? 'text-green-400' : 'text-red-400'}`}>
                  Resolved: {market.outcome.toUpperCase()}
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Period Summary */}
      <PeriodSummary market={market} trades={trades} snapshots={snapshots} />

      {/* Charts */}
      <div className="grid gap-8">
        <section>
          <h2 className="text-lg font-semibold mb-4">PnL Over Time</h2>
          <PnLChart snapshots={snapshots} trades={trades} />
        </section>

        <section>
          <h2 className="text-lg font-semibold mb-4">Quote vs Market BBO</h2>
          <QuoteBBOChart snapshots={snapshots} />
        </section>

        <section>
          <h2 className="text-lg font-semibold mb-4">Inventory</h2>
          <InventoryChart snapshots={snapshots} />
        </section>
      </div>

      {/* Trades table */}
      {trades.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold mb-4">Trades in This Period</h2>
          <div className="bg-gray-800 rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-700">
                <tr>
                  <th className="px-4 py-3 text-left text-gray-300 font-medium">Time</th>
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
                      <td className="px-4 py-2 text-gray-400">
                        {new Date(trade.timestamp).toLocaleTimeString()}
                      </td>
                      <td className={`px-4 py-2 text-center ${sideColor}`}>
                        {isBuy ? 'BUY' : 'SELL'}
                      </td>
                      <td className="px-4 py-2 text-right">${trade.price.toFixed(4)}</td>
                      <td className="px-4 py-2 text-right">{trade.size.toFixed(2)}</td>
                      <td className={`px-4 py-2 text-right ${pnlColor}`}>
                        ${trade.pnl.toFixed(2)}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  )
}
