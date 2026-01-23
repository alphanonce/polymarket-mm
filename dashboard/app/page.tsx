'use client'

import { EquityChart } from '@/components/overview/EquityChart'
import { PositionsTable } from '@/components/overview/PositionsTable'
import { TradesTable } from '@/components/overview/TradesTable'
import { MetricsCard } from '@/components/overview/MetricsCard'
import { MarketsList } from '@/components/overview/MarketsList'

export default function Home() {
  return (
    <div className="space-y-8">
      {/* Metrics Cards */}
      <MetricsCard />

      {/* Equity Chart */}
      <section>
        <h2 className="text-lg font-semibold mb-4">Equity Curve</h2>
        <EquityChart />
      </section>

      {/* Two-column layout for positions and trades */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <section>
          <h2 className="text-lg font-semibold mb-4">Current Positions</h2>
          <PositionsTable />
        </section>

        <section>
          <h2 className="text-lg font-semibold mb-4">Recent Trades</h2>
          <TradesTable />
        </section>
      </div>

      {/* Markets List */}
      <section>
        <h2 className="text-lg font-semibold mb-4">Markets</h2>
        <MarketsList />
      </section>
    </div>
  )
}
