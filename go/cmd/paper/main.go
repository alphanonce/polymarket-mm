// Package main is the entry point for the paper trading data service
package main

import (
	"flag"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/alphanonce/polymarket-mm/internal/aggregator"
	"github.com/alphanonce/polymarket-mm/internal/binance"
	"github.com/alphanonce/polymarket-mm/internal/chainlink"
	"github.com/alphanonce/polymarket-mm/internal/paper"
	"github.com/alphanonce/polymarket-mm/internal/polymarket"
	"github.com/alphanonce/polymarket-mm/internal/shm"
	"go.uber.org/zap"
	"gopkg.in/yaml.v3"
)

// Config represents the paper trading service configuration
type Config struct {
	Polymarket struct {
		WSEndpoint string   `yaml:"ws_endpoint"`
		Markets    []string `yaml:"markets"`
	} `yaml:"polymarket"`

	Binance struct {
		Symbols []string `yaml:"symbols"`
	} `yaml:"binance"`

	Chainlink struct {
		RPCURL string `yaml:"rpc_url"`
	} `yaml:"chainlink"`

	Supabase struct {
		URL    string `yaml:"url"`
		APIKey string `yaml:"api_key"`
	} `yaml:"supabase"`

	Paper struct {
		EquitySnapshotInterval    time.Duration `yaml:"equity_snapshot_interval"`
		MarketSnapshotInterval    time.Duration `yaml:"market_snapshot_interval"`
		MetricsSyncInterval       time.Duration `yaml:"metrics_sync_interval"`
		PositionSyncInterval      time.Duration `yaml:"position_sync_interval"`
		TradeBatchSize            int           `yaml:"trade_batch_size"`
		MarketSnapshotBatchSize   int           `yaml:"market_snapshot_batch_size"`
	} `yaml:"paper"`
}

func main() {
	configPath := flag.String("config", "data/config/paper.yaml", "Path to config file")
	flag.Parse()

	// Initialize logger
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// Load config
	cfg, err := loadConfig(*configPath)
	if err != nil {
		logger.Fatal("Failed to load config", zap.Error(err))
	}

	// Initialize Supabase client
	supabaseClient := paper.NewSupabaseClient(
		cfg.Supabase.URL,
		cfg.Supabase.APIKey,
		logger,
	)

	// Health check
	if err := supabaseClient.HealthCheck(); err != nil {
		logger.Fatal("Supabase health check failed", zap.Error(err))
	}
	logger.Info("Connected to Supabase")

	// Initialize store with batching
	storeConfig := paper.DefaultStoreConfig()
	if cfg.Paper.TradeBatchSize > 0 {
		storeConfig.TradeBatchSize = cfg.Paper.TradeBatchSize
	}
	if cfg.Paper.MarketSnapshotBatchSize > 0 {
		storeConfig.SnapshotBatchSize = cfg.Paper.MarketSnapshotBatchSize
	}
	store := paper.NewStore(supabaseClient, storeConfig, logger)
	store.Start()
	defer store.Stop()

	// Initialize main shared memory writer (for aggregator)
	shmWriter, err := shm.NewWriter()
	if err != nil {
		logger.Fatal("Failed to create SHM writer", zap.Error(err))
	}
	defer shmWriter.Close()

	// Initialize paper SHM writer (for marking trades persisted)
	paperSHMWriter, err := shm.NewPaperWriter()
	if err != nil {
		// Paper SHM might not exist yet if Python hasn't started
		logger.Warn("Paper SHM not available yet, will retry", zap.Error(err))
	} else {
		defer paperSHMWriter.Close()
	}

	// Initialize Polymarket WebSocket client
	polyWS := polymarket.NewWSClient(polymarket.WSClientConfig{
		Endpoint: cfg.Polymarket.WSEndpoint,
		Logger:   logger,
	})
	if err := polyWS.Connect(); err != nil {
		logger.Fatal("Failed to connect to Polymarket WebSocket", zap.Error(err))
	}
	defer polyWS.Close()

	// Subscribe to markets
	for _, market := range cfg.Polymarket.Markets {
		if err := polyWS.SubscribeBook(market); err != nil {
			logger.Error("Failed to subscribe to book", zap.String("market", market), zap.Error(err))
		}
		if err := polyWS.SubscribeTrades(market); err != nil {
			logger.Error("Failed to subscribe to trades", zap.String("market", market), zap.Error(err))
		}
	}

	// Initialize Binance client
	var binanceWS *binance.SpotClient
	if len(cfg.Binance.Symbols) > 0 {
		binanceWS = binance.NewSpotClient(binance.SpotClientConfig{
			Logger: logger,
		})
		if err := binanceWS.Connect(cfg.Binance.Symbols); err != nil {
			logger.Error("Failed to connect to Binance WebSocket", zap.Error(err))
		} else {
			defer binanceWS.Close()
		}
	}

	// Initialize Chainlink reader (optional)
	var chainlinkReader *chainlink.Reader
	if cfg.Chainlink.RPCURL != "" {
		chainlinkReader, err = chainlink.NewReader(chainlink.ReaderConfig{
			RPCURL: cfg.Chainlink.RPCURL,
			Logger: logger,
		})
		if err != nil {
			logger.Error("Failed to create Chainlink reader", zap.Error(err))
		} else {
			defer chainlinkReader.Stop()
			if err := chainlinkReader.AddDefaultFeeds(); err != nil {
				logger.Error("Failed to add default feeds", zap.Error(err))
			}
			chainlinkReader.StartPolling()
		}
	}

	// Initialize aggregator
	agg := aggregator.New(aggregator.Config{
		Logger:          logger,
		SHMWriter:       shmWriter,
		PolyWS:          polyWS,
		BinanceWS:       binanceWS,
		ChainlinkReader: chainlinkReader,
	})
	if err := agg.Start(); err != nil {
		logger.Fatal("Failed to start aggregator", zap.Error(err))
	}
	defer agg.Stop()

	// Start paper trading sync goroutines
	stopCh := make(chan struct{})

	// Sync equity snapshots
	equityInterval := cfg.Paper.EquitySnapshotInterval
	if equityInterval == 0 {
		equityInterval = 60 * time.Second
	}
	go syncEquitySnapshots(paperSHMWriter, store, equityInterval, stopCh, logger)

	// Sync market snapshots
	snapshotInterval := cfg.Paper.MarketSnapshotInterval
	if snapshotInterval == 0 {
		snapshotInterval = 5 * time.Second
	}
	go syncMarketSnapshots(paperSHMWriter, store, snapshotInterval, stopCh, logger)

	// Sync metrics
	metricsInterval := cfg.Paper.MetricsSyncInterval
	if metricsInterval == 0 {
		metricsInterval = 60 * time.Second
	}
	go syncMetrics(paperSHMWriter, store, metricsInterval, stopCh, logger)

	// Sync positions
	positionInterval := cfg.Paper.PositionSyncInterval
	if positionInterval == 0 {
		positionInterval = 10 * time.Second
	}
	go syncPositions(paperSHMWriter, store, positionInterval, stopCh, logger)

	// Watch for pending trades
	go watchTrades(paperSHMWriter, store, stopCh, logger)

	logger.Info("Paper trading service started successfully")

	// Wait for shutdown signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	close(stopCh)
	logger.Info("Shutting down...")
}

func syncEquitySnapshots(reader *shm.PaperWriter, store *paper.Store, interval time.Duration, stopCh <-chan struct{}, logger *zap.Logger) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if reader == nil {
				continue
			}

			layout := reader.Layout()
			snap := paper.EquitySnapshot{
				Equity:        layout.TotalEquity,
				Cash:          layout.Cash,
				PositionValue: layout.PositionValue,
				Timestamp:     time.Now(),
			}

			if snap.Equity > 0 {
				if err := store.InsertEquitySnapshot(snap); err != nil {
					logger.Error("Failed to insert equity snapshot", zap.Error(err))
				}
			}

		case <-stopCh:
			return
		}
	}
}

func syncMarketSnapshots(reader *shm.PaperWriter, store *paper.Store, interval time.Duration, stopCh <-chan struct{}, logger *zap.Logger) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if reader == nil {
				continue
			}

			// Create paper reader for reading quotes
			paperReader, err := shm.NewPaperReader()
			if err != nil {
				continue
			}

			quotes := paperReader.GetQuotes()
			paperReader.Close()

			for _, q := range quotes {
				if q.Slug != "" {
					store.BufferMarketSnapshot(q)
				}
			}

		case <-stopCh:
			return
		}
	}
}

func syncMetrics(reader *shm.PaperWriter, store *paper.Store, interval time.Duration, stopCh <-chan struct{}, logger *zap.Logger) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if reader == nil {
				continue
			}

			// Create paper reader for reading metrics
			paperReader, err := shm.NewPaperReader()
			if err != nil {
				continue
			}

			metrics := paperReader.GetMetrics()
			paperReader.Close()

			if err := store.UpsertMetrics(metrics); err != nil {
				logger.Error("Failed to upsert metrics", zap.Error(err))
			}

		case <-stopCh:
			return
		}
	}
}

func syncPositions(reader *shm.PaperWriter, store *paper.Store, interval time.Duration, stopCh <-chan struct{}, logger *zap.Logger) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if reader == nil {
				continue
			}

			// Create paper reader for reading positions
			paperReader, err := shm.NewPaperReader()
			if err != nil {
				continue
			}

			positions := paperReader.GetPositions()
			paperReader.Close()

			if len(positions) > 0 {
				if err := store.UpsertPositions(positions); err != nil {
					logger.Error("Failed to upsert positions", zap.Error(err))
				}
			}

		case <-stopCh:
			return
		}
	}
}

func watchTrades(writer *shm.PaperWriter, store *paper.Store, stopCh <-chan struct{}, logger *zap.Logger) {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if writer == nil {
				continue
			}

			// Create paper reader for reading trades
			paperReader, err := shm.NewPaperReader()
			if err != nil {
				continue
			}

			trades := paperReader.GetPendingTrades()
			layout := paperReader.Layout()
			currentTail := layout.TradesTail
			paperReader.Close()

			if len(trades) > 0 {
				for _, t := range trades {
					store.BufferTrade(t)
				}

				// Mark trades as persisted
				writer.MarkTradesPersisted(currentTail)
				logger.Debug("Persisted trades", zap.Int("count", len(trades)))
			}

		case <-stopCh:
			return
		}
	}
}

func loadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}

	// Override with environment variables
	if v := os.Getenv("SUPABASE_URL"); v != "" {
		cfg.Supabase.URL = v
	}
	if v := os.Getenv("SUPABASE_KEY"); v != "" {
		cfg.Supabase.APIKey = v
	}
	if v := os.Getenv("POLYGON_RPC_URL"); v != "" {
		cfg.Chainlink.RPCURL = v
	}

	return &cfg, nil
}
