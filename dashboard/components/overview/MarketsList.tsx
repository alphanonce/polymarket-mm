'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { getMarkets, Market } from '@/lib/supabase'
import { format } from 'date-fns'

function formatNumber(num: number, decimals = 2): string {
  return num.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

export function MarketsList() {
  const [markets, setMarkets] = useState<Market[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<'all' | 'active' | 'resolved'>('all')

  useEffect(() => {
    async function fetchMarkets() {
      const data = await getMarkets()
      setMarkets(data)
      setLoading(false)
    }
    fetchMarkets()
  }, [])

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="animate-pulse space-y-4">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-16 bg-gray-700 rounded"></div>
          ))}
        </div>
      </div>
    )
  }

  const filteredMarkets = markets.filter((m) => {
    if (filter === 'all') return true
    return m.status === filter
  })

  return (
    <div className="space-y-4">
      {/* Filter buttons */}
      <div className="flex gap-2">
        {(['all', 'active', 'resolved'] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              filter === f
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      {/* Markets grid */}
      {filteredMarkets.length === 0 ? (
        <div className="bg-gray-800 rounded-lg p-4 text-gray-400 text-center">
          No markets found
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredMarkets.map((market) => {
            const pnlColor = market.period_pnl >= 0 ? 'text-green-400' : 'text-red-400'
            const statusColor =
              market.status === 'active' ? 'bg-green-500' : 'bg-gray-500'

            return (
              <Link
                key={market.id}
                href={`/periods/${market.slug}`}
                className="bg-gray-800 rounded-lg p-4 hover:bg-gray-750 transition-colors block"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="font-medium truncate" title={market.slug}>
                    {market.slug}
                  </div>
                  <span
                    className={`w-2 h-2 rounded-full ${statusColor}`}
                    title={market.status}
                  />
                </div>

                <div className="flex items-center gap-2 text-sm text-gray-400 mb-3">
                  <span>{market.asset.toUpperCase()}</span>
                  <span>/</span>
                  <span>{market.timeframe}</span>
                </div>

                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-400">Period PnL</div>
                  <div className={`font-medium ${pnlColor}`}>
                    ${formatNumber(market.period_pnl)}
                  </div>
                </div>

                {market.outcome && (
                  <div className="mt-2 flex items-center justify-between">
                    <div className="text-sm text-gray-400">Outcome</div>
                    <div
                      className={`font-medium ${
                        market.outcome === 'up' ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {market.outcome.toUpperCase()}
                    </div>
                  </div>
                )}

                <div className="mt-2 text-xs text-gray-500">
                  {format(new Date(market.created_at), 'MMM dd, HH:mm')}
                </div>
              </Link>
            )
          })}
        </div>
      )}
    </div>
  )
}
