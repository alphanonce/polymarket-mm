package paper

import (
	"sync"
	"time"

	"go.uber.org/zap"
)

// Store provides buffered writes to Supabase with batching
type Store struct {
	client *SupabaseClient
	logger *zap.Logger

	// Buffered data for batch writes
	tradeBuf      []Trade
	snapshotBuf   []MarketSnapshot
	tradeMu       sync.Mutex
	snapshotMu    sync.Mutex

	// Batch settings
	tradeBatchSize    int
	snapshotBatchSize int
	flushInterval     time.Duration

	// Shutdown
	stopCh chan struct{}
	wg     sync.WaitGroup
}

// StoreConfig configures the paper trading store
type StoreConfig struct {
	TradeBatchSize    int
	SnapshotBatchSize int
	FlushInterval     time.Duration
}

// DefaultStoreConfig returns default store configuration
func DefaultStoreConfig() StoreConfig {
	return StoreConfig{
		TradeBatchSize:    50,
		SnapshotBatchSize: 100,
		FlushInterval:     5 * time.Second,
	}
}

// NewStore creates a new paper trading store
func NewStore(client *SupabaseClient, config StoreConfig, logger *zap.Logger) *Store {
	if config.TradeBatchSize == 0 {
		config.TradeBatchSize = 50
	}
	if config.SnapshotBatchSize == 0 {
		config.SnapshotBatchSize = 100
	}
	if config.FlushInterval == 0 {
		config.FlushInterval = 5 * time.Second
	}

	return &Store{
		client:            client,
		logger:            logger,
		tradeBuf:          make([]Trade, 0, config.TradeBatchSize),
		snapshotBuf:       make([]MarketSnapshot, 0, config.SnapshotBatchSize),
		tradeBatchSize:    config.TradeBatchSize,
		snapshotBatchSize: config.SnapshotBatchSize,
		flushInterval:     config.FlushInterval,
		stopCh:            make(chan struct{}),
	}
}

// Start begins the background flush goroutine
func (s *Store) Start() {
	s.wg.Add(1)
	go s.flushLoop()
}

// Stop stops the store and flushes remaining data
func (s *Store) Stop() {
	close(s.stopCh)
	s.wg.Wait()

	// Final flush
	s.flushTrades()
	s.flushSnapshots()
}

func (s *Store) flushLoop() {
	defer s.wg.Done()
	ticker := time.NewTicker(s.flushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.flushTrades()
			s.flushSnapshots()
		case <-s.stopCh:
			return
		}
	}
}

// BufferTrade adds a trade to the buffer
func (s *Store) BufferTrade(trade Trade) {
	s.tradeMu.Lock()
	s.tradeBuf = append(s.tradeBuf, trade)
	shouldFlush := len(s.tradeBuf) >= s.tradeBatchSize
	s.tradeMu.Unlock()

	if shouldFlush {
		go s.flushTrades()
	}
}

// BufferMarketSnapshot adds a market snapshot to the buffer
func (s *Store) BufferMarketSnapshot(snap MarketSnapshot) {
	s.snapshotMu.Lock()
	s.snapshotBuf = append(s.snapshotBuf, snap)
	shouldFlush := len(s.snapshotBuf) >= s.snapshotBatchSize
	s.snapshotMu.Unlock()

	if shouldFlush {
		go s.flushSnapshots()
	}
}

func (s *Store) flushTrades() {
	s.tradeMu.Lock()
	if len(s.tradeBuf) == 0 {
		s.tradeMu.Unlock()
		return
	}
	trades := s.tradeBuf
	s.tradeBuf = make([]Trade, 0, s.tradeBatchSize)
	s.tradeMu.Unlock()

	if err := s.client.InsertTrades(trades); err != nil {
		s.logger.Error("Failed to flush trades", zap.Error(err), zap.Int("count", len(trades)))
		// Re-add to buffer on failure
		s.tradeMu.Lock()
		s.tradeBuf = append(trades, s.tradeBuf...)
		s.tradeMu.Unlock()
	}
}

func (s *Store) flushSnapshots() {
	s.snapshotMu.Lock()
	if len(s.snapshotBuf) == 0 {
		s.snapshotMu.Unlock()
		return
	}
	snapshots := s.snapshotBuf
	s.snapshotBuf = make([]MarketSnapshot, 0, s.snapshotBatchSize)
	s.snapshotMu.Unlock()

	if err := s.client.InsertMarketSnapshots(snapshots); err != nil {
		s.logger.Error("Failed to flush snapshots", zap.Error(err), zap.Int("count", len(snapshots)))
		// Re-add to buffer on failure (if not too many)
		if len(snapshots) < s.snapshotBatchSize*2 {
			s.snapshotMu.Lock()
			s.snapshotBuf = append(snapshots, s.snapshotBuf...)
			s.snapshotMu.Unlock()
		}
	}
}

// InsertTrade inserts a trade immediately (bypasses buffer)
func (s *Store) InsertTrade(trade Trade) error {
	return s.client.InsertTrade(trade)
}

// UpsertPosition upserts a position immediately
func (s *Store) UpsertPosition(pos Position) error {
	return s.client.UpsertPosition(pos)
}

// UpsertPositions upserts multiple positions
func (s *Store) UpsertPositions(positions []Position) error {
	return s.client.UpsertPositions(positions)
}

// InsertEquitySnapshot inserts an equity snapshot immediately
func (s *Store) InsertEquitySnapshot(snap EquitySnapshot) error {
	return s.client.InsertEquitySnapshot(snap)
}

// UpsertMetrics updates metrics immediately
func (s *Store) UpsertMetrics(metrics Metrics) error {
	return s.client.UpsertMetrics(metrics)
}

// UpsertMarket upserts a market record
func (s *Store) UpsertMarket(market Market) error {
	return s.client.UpsertMarket(market)
}

// InsertMarketSnapshot inserts a market snapshot (use BufferMarketSnapshot for batching)
func (s *Store) InsertMarketSnapshot(snap MarketSnapshot) error {
	return s.client.InsertMarketSnapshot(snap)
}

// HealthCheck tests connectivity
func (s *Store) HealthCheck() error {
	return s.client.HealthCheck()
}
