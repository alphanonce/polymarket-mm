'use client'

import { useEffect, useState, useCallback } from 'react'
import { supabase, Metrics, Position, Trade, EquitySnapshot, getMetrics, getPositions, getTrades, getEquityHistory } from '@/lib/supabase'

export function useRealtimeMetrics() {
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [loading, setLoading] = useState(true)

  const fetchMetrics = useCallback(async () => {
    const data = await getMetrics()
    setMetrics(data)
    setLoading(false)
  }, [])

  useEffect(() => {
    fetchMetrics()

    // Subscribe to real-time updates
    const channel = supabase
      .channel('metrics-changes')
      .on('postgres_changes', {
        event: '*',
        schema: 'public',
        table: 'metrics',
      }, () => {
        fetchMetrics()
      })
      .subscribe()

    return () => {
      supabase.removeChannel(channel)
    }
  }, [fetchMetrics])

  return { metrics, loading, refresh: fetchMetrics }
}

export function useRealtimePositions() {
  const [positions, setPositions] = useState<Position[]>([])
  const [loading, setLoading] = useState(true)

  const fetchPositions = useCallback(async () => {
    const data = await getPositions()
    setPositions(data)
    setLoading(false)
  }, [])

  useEffect(() => {
    fetchPositions()

    const channel = supabase
      .channel('positions-changes')
      .on('postgres_changes', {
        event: '*',
        schema: 'public',
        table: 'positions',
      }, () => {
        fetchPositions()
      })
      .subscribe()

    return () => {
      supabase.removeChannel(channel)
    }
  }, [fetchPositions])

  return { positions, loading, refresh: fetchPositions }
}

export function useRealtimeTrades(limit = 50) {
  const [trades, setTrades] = useState<Trade[]>([])
  const [loading, setLoading] = useState(true)

  const fetchTrades = useCallback(async () => {
    const data = await getTrades(limit)
    setTrades(data)
    setLoading(false)
  }, [limit])

  useEffect(() => {
    fetchTrades()

    const channel = supabase
      .channel('trades-changes')
      .on('postgres_changes', {
        event: 'INSERT',
        schema: 'public',
        table: 'trades',
      }, () => {
        fetchTrades()
      })
      .subscribe()

    return () => {
      supabase.removeChannel(channel)
    }
  }, [fetchTrades])

  return { trades, loading, refresh: fetchTrades }
}

export function useRealtimeEquity() {
  const [equityHistory, setEquityHistory] = useState<EquitySnapshot[]>([])
  const [loading, setLoading] = useState(true)

  const fetchEquity = useCallback(async () => {
    const data = await getEquityHistory()
    setEquityHistory(data)
    setLoading(false)
  }, [])

  useEffect(() => {
    fetchEquity()

    const channel = supabase
      .channel('equity-changes')
      .on('postgres_changes', {
        event: 'INSERT',
        schema: 'public',
        table: 'equity_snapshots',
      }, () => {
        fetchEquity()
      })
      .subscribe()

    return () => {
      supabase.removeChannel(channel)
    }
  }, [fetchEquity])

  return { equityHistory, loading, refresh: fetchEquity }
}
