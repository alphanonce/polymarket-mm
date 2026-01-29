// Package main is the entry point for the trading executor
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
	"github.com/alphanonce/polymarket-mm/internal/executor"
	"github.com/alphanonce/polymarket-mm/internal/polymarket"
	"github.com/alphanonce/polymarket-mm/internal/shm"
	"go.uber.org/zap"
	"gopkg.in/yaml.v3"
)

// Config represents the executor configuration
type Config struct {
	Polymarket struct {
		APIKey     string   `yaml:"api_key"`
		APISecret  string   `yaml:"api_secret"`
		Passphrase string   `yaml:"passphrase"`
		Endpoint   string   `yaml:"endpoint"`
		WSEndpoint string   `yaml:"ws_endpoint"`
		Markets    []string `yaml:"markets"`
	} `yaml:"polymarket"`

	// Market discovery settings
	Discovery struct {
		Enabled     bool     `yaml:"enabled"`
		PollIntervalS int    `yaml:"poll_interval_s"`
		Assets      []string `yaml:"assets"`
		Timeframes  []string `yaml:"timeframes"`
	} `yaml:"discovery"`

	Binance struct {
		Symbols []string `yaml:"symbols"`
	} `yaml:"binance"`

	Chainlink struct {
		RPCURL string   `yaml:"rpc_url"`
		Feeds  []string `yaml:"feeds"`
	} `yaml:"chainlink"`

	Risk struct {
		MaxPositionPerAsset float64 `yaml:"max_position_per_asset"`
		MaxTotalExposure    float64 `yaml:"max_total_exposure"`
		MaxOrderSize        float64 `yaml:"max_order_size"`
		MinOrderSize        float64 `yaml:"min_order_size"`
	} `yaml:"risk"`
}

func main() {
	configPath := flag.String("config", "data/config/executor.yaml", "Path to config file")
	flag.Parse()

	// Initialize logger
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// Load config
	cfg, err := loadConfig(*configPath)
	if err != nil {
		logger.Fatal("Failed to load config", zap.Error(err))
	}

	// Initialize shared memory writer
	shmWriter, err := shm.NewWriter()
	if err != nil {
		logger.Fatal("Failed to create SHM writer", zap.Error(err))
	}
	defer shmWriter.Close()

	// Initialize Polymarket client
	polyClient := polymarket.NewClient(polymarket.ClientConfig{
		Endpoint:   cfg.Polymarket.Endpoint,
		APIKey:     cfg.Polymarket.APIKey,
		APISecret:  cfg.Polymarket.APISecret,
		Passphrase: cfg.Polymarket.Passphrase,
		Logger:     logger,
	})

	// Initialize Polymarket WebSocket client
	polyWS := polymarket.NewWSClient(polymarket.WSClientConfig{
		Endpoint: cfg.Polymarket.WSEndpoint,
		Logger:   logger,
	})
	if err := polyWS.Connect(); err != nil {
		logger.Fatal("Failed to connect to Polymarket WebSocket", zap.Error(err))
	}
	defer polyWS.Close()

	// Subscribe to static markets from config
	for _, market := range cfg.Polymarket.Markets {
		if err := polyWS.SubscribeBook(market); err != nil {
			logger.Error("Failed to subscribe to book", zap.String("market", market), zap.Error(err))
		}
		if err := polyWS.SubscribeTrades(market); err != nil {
			logger.Error("Failed to subscribe to trades", zap.String("market", market), zap.Error(err))
		}
	}

	// Initialize market discovery service
	var discovery *polymarket.DiscoveryService
	if cfg.Discovery.Enabled {
		discoveryCfg := polymarket.DiscoveryConfig{
			Logger: logger,
		}
		if cfg.Discovery.PollIntervalS > 0 {
			discoveryCfg.PollInterval = time.Duration(cfg.Discovery.PollIntervalS) * time.Second
		}
		if len(cfg.Discovery.Assets) > 0 {
			discoveryCfg.SupportedAssets = cfg.Discovery.Assets
		}
		if len(cfg.Discovery.Timeframes) > 0 {
			discoveryCfg.SupportedTimeframes = cfg.Discovery.Timeframes
		}

		discovery = polymarket.NewDiscoveryService(discoveryCfg)

		// Subscribe to newly discovered markets in batch
		// (Polymarket WebSocket only sends initial snapshots for tokens in the same subscription message)
		discovery.OnBatchDiscovered(func(markets []*polymarket.DiscoveredMarket) {
			// Collect all token IDs
			tokenIDs := make([]string, 0, len(markets)*2)
			for _, market := range markets {
				tokenIDs = append(tokenIDs, market.TokenIDUp, market.TokenIDDown)
			}

			// Subscribe to all tokens in one batch
			if err := polyWS.SubscribeBooksMany(tokenIDs); err != nil {
				logger.Error("Failed to batch subscribe to books", zap.Error(err))
			}

			// Log individual markets
			for _, market := range markets {
				logger.Info("Subscribed to market",
					zap.String("slug", market.Slug),
					zap.String("tokenUp", market.TokenIDUp[:20]+"..."),
					zap.String("tokenDown", market.TokenIDDown[:20]+"..."),
				)
			}
		})

		// Remove expired markets from SHM
		discovery.OnMarketExpired(func(market *polymarket.DiscoveredMarket) {
			// Remove both up and down tokens from SHM
			if err := shmWriter.RemoveMarket(market.TokenIDUp); err != nil {
				logger.Debug("Failed to remove up token from SHM", zap.Error(err))
			}
			if err := shmWriter.RemoveMarket(market.TokenIDDown); err != nil {
				logger.Debug("Failed to remove down token from SHM", zap.Error(err))
			}
			logger.Info("Removed expired market from SHM",
				zap.String("slug", market.Slug),
			)
		})

		// Note: discovery.Start() is called later after aggregator is initialized
		defer discovery.Stop()
	}

	// Initialize Binance client
	binanceWS := binance.NewSpotClient(binance.SpotClientConfig{
		Logger: logger,
	})
	if len(cfg.Binance.Symbols) > 0 {
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
		Logger:         logger,
		SHMWriter:      shmWriter,
		PolyWS:         polyWS,
		BinanceWS:      binanceWS,
		ChainlinkReader: chainlinkReader,
	})
	if err := agg.Start(); err != nil {
		logger.Fatal("Failed to start aggregator", zap.Error(err))
	}
	defer agg.Stop()

	// Start market discovery after aggregator is initialized
	// (so callbacks are set before receiving initial snapshots)
	if discovery != nil {
		discovery.Start()
	}

	// Initialize risk manager
	riskMgr := executor.NewRiskManager(executor.RiskConfig{
		MaxPositionPerAsset: cfg.Risk.MaxPositionPerAsset,
		MaxTotalExposure:    cfg.Risk.MaxTotalExposure,
		MaxOrderSize:        cfg.Risk.MaxOrderSize,
		MinOrderSize:        cfg.Risk.MinOrderSize,
	})

	// Initialize executor
	exec := executor.New(executor.Config{
		Logger:      logger,
		SHMWriter:   shmWriter,
		Client:      polyClient,
		RiskManager: riskMgr,
	})
	if err := exec.Start(); err != nil {
		logger.Fatal("Failed to start executor", zap.Error(err))
	}
	defer exec.Stop()

	logger.Info("Executor started successfully")

	// Wait for shutdown signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	logger.Info("Shutting down...")
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
	if v := os.Getenv("POLYMARKET_API_KEY"); v != "" {
		cfg.Polymarket.APIKey = v
	}
	if v := os.Getenv("POLYMARKET_API_SECRET"); v != "" {
		cfg.Polymarket.APISecret = v
	}
	if v := os.Getenv("POLYMARKET_PASSPHRASE"); v != "" {
		cfg.Polymarket.Passphrase = v
	}
	if v := os.Getenv("POLYGON_RPC_URL"); v != "" {
		cfg.Chainlink.RPCURL = v
	}

	return &cfg, nil
}
