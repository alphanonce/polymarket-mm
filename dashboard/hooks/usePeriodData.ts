'use client'

import { useEffect, useState, useCallback } from 'react'
import { supabase, Market, MarketSnapshot, Trade, getMarketBySlug, getMarketSnapshots, getTradesBySlug } from '@/lib/supabase'

export function usePeriodData(slug: string) {
  const [market, setMarket] = useState<Market | null>(null)
  const [snapshots, setSnapshots] = useState<MarketSnapshot[]>([])
  const [trades, setTrades] = useState<Trade[]>([])
  const [loading, setLoading] = useState(true)

  const fetchData = useCallback(async () => {
    const [marketData, snapshotsData, tradesData] = await Promise.all([
      getMarketBySlug(slug),
      getMarketSnapshots(slug),
      getTradesBySlug(slug),
    ])

    setMarket(marketData)
    setSnapshots(snapshotsData)
    setTrades(tradesData)
    setLoading(false)
  }, [slug])

  useEffect(() => {
    fetchData()

    // Subscribe to real-time updates for this market
    const snapshotChannel = supabase
      .channel(`snapshots-${slug}`)
      .on('postgres_changes', {
        event: 'INSERT',
        schema: 'public',
        table: 'market_snapshots',
        filter: `slug=eq.${slug}`,
      }, () => {
        fetchData()
      })
      .subscribe()

    const tradesChannel = supabase
      .channel(`trades-${slug}`)
      .on('postgres_changes', {
        event: 'INSERT',
        schema: 'public',
        table: 'trades',
        filter: `slug=eq.${slug}`,
      }, () => {
        fetchData()
      })
      .subscribe()

    return () => {
      supabase.removeChannel(snapshotChannel)
      supabase.removeChannel(tradesChannel)
    }
  }, [slug, fetchData])

  return { market, snapshots, trades, loading, refresh: fetchData }
}
