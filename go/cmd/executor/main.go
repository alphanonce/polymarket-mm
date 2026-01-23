// Package main is the entry point for the trading executor
package main

import (
	"flag"
	"os"
	"os/signal"
	"syscall"

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
