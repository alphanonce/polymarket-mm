-- Paper Trading Database Schema for Supabase
-- Run this in the Supabase SQL Editor to create the tables

-- Positions (current state)
CREATE TABLE IF NOT EXISTS positions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  asset_id TEXT NOT NULL UNIQUE,
  slug TEXT,
  side TEXT,  -- 'up' or 'down'
  size DECIMAL NOT NULL DEFAULT 0,
  avg_entry_price DECIMAL NOT NULL DEFAULT 0,
  unrealized_pnl DECIMAL DEFAULT 0,
  realized_pnl DECIMAL DEFAULT 0,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trades (history)
CREATE TABLE IF NOT EXISTS trades (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  asset_id TEXT NOT NULL,
  slug TEXT NOT NULL,
  side INTEGER NOT NULL,  -- 1=buy, -1=sell
  price DECIMAL NOT NULL,
  size DECIMAL NOT NULL,
  pnl DECIMAL DEFAULT 0,
  timestamp TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Equity snapshots (for chart)
CREATE TABLE IF NOT EXISTS equity_snapshots (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  equity DECIMAL NOT NULL,
  cash DECIMAL NOT NULL,
  position_value DECIMAL NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Metrics (latest stats - singleton row)
CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY DEFAULT 1,
  total_pnl DECIMAL DEFAULT 0,
  realized_pnl DECIMAL DEFAULT 0,
  unrealized_pnl DECIMAL DEFAULT 0,
  total_trades INTEGER DEFAULT 0,
  win_rate DECIMAL DEFAULT 0,
  sharpe_ratio DECIMAL DEFAULT 0,
  max_drawdown DECIMAL DEFAULT 0,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Markets (active/resolved periods)
CREATE TABLE IF NOT EXISTS markets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  slug TEXT NOT NULL UNIQUE,
  asset TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  status TEXT DEFAULT 'active',  -- 'active', 'resolved'
  outcome TEXT,  -- 'up', 'down' (when resolved)
  start_ts BIGINT,
  end_ts BIGINT,
  period_pnl DECIMAL DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Market snapshots (for per-period charts: quote vs BBO, inventory)
CREATE TABLE IF NOT EXISTS market_snapshots (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  slug TEXT NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  -- Market BBO
  best_bid DECIMAL,
  best_ask DECIMAL,
  mid_price DECIMAL,
  spread DECIMAL,
  -- Our quotes
  our_bid DECIMAL,
  our_ask DECIMAL,
  -- Inventory
  inventory DECIMAL DEFAULT 0,
  inventory_value DECIMAL DEFAULT 0,
  -- PnL at this point
  period_pnl DECIMAL DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_market_snapshots_slug_ts ON market_snapshots(slug, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_slug_ts ON trades(slug, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_equity_snapshots_ts ON equity_snapshots(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_positions_updated ON positions(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_markets_status ON markets(status);
CREATE INDEX IF NOT EXISTS idx_markets_asset_timeframe ON markets(asset, timeframe);

-- Initialize metrics row (singleton pattern)
INSERT INTO metrics (id, total_pnl, realized_pnl, unrealized_pnl, total_trades, win_rate, sharpe_ratio, max_drawdown)
VALUES (1, 0, 0, 0, 0, 0, 0, 0)
ON CONFLICT (id) DO NOTHING;

-- Enable Row Level Security (RLS) for Supabase
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE equity_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE markets ENABLE ROW LEVEL SECURITY;
ALTER TABLE market_snapshots ENABLE ROW LEVEL SECURITY;

-- Create policies for anon/service role access
-- These allow full access - adjust based on your security requirements
CREATE POLICY "Allow all access to positions" ON positions FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all access to trades" ON trades FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all access to equity_snapshots" ON equity_snapshots FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all access to metrics" ON metrics FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all access to markets" ON markets FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all access to market_snapshots" ON market_snapshots FOR ALL USING (true) WITH CHECK (true);

-- Enable realtime for dashboard subscriptions
ALTER PUBLICATION supabase_realtime ADD TABLE positions;
ALTER PUBLICATION supabase_realtime ADD TABLE trades;
ALTER PUBLICATION supabase_realtime ADD TABLE equity_snapshots;
ALTER PUBLICATION supabase_realtime ADD TABLE metrics;
ALTER PUBLICATION supabase_realtime ADD TABLE markets;
ALTER PUBLICATION supabase_realtime ADD TABLE market_snapshots;
