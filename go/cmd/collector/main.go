// Package main is the entry point for the data collector
package main

import (
	"flag"
	"os"
	"os/signal"
	"syscall"

	"github.com/alphanonce/polymarket-mm/internal/binance"
	"github.com/alphanonce/polymarket-mm/internal/collector"
	"github.com/alphanonce/polymarket-mm/internal/polymarket"
	"go.uber.org/zap"
	"gopkg.in/yaml.v3"
)

// Config represents the collector configuration
type Config struct {
	Polymarket struct {
		WSEndpoint string   `yaml:"ws_endpoint"`
		Markets    []string `yaml:"markets"`
	} `yaml:"polymarket"`

	Binance struct {
		Symbols []string `yaml:"symbols"`
	} `yaml:"binance"`

	Collector struct {
		OutputDir        string `yaml:"output_dir"`
		SnapshotInterval string `yaml:"snapshot_interval"`
		FlushInterval    string `yaml:"flush_interval"`
	} `yaml:"collector"`
}

func main() {
	configPath := flag.String("config", "data/config/collector.yaml", "Path to config file")
	flag.Parse()

	// Initialize logger
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// Load config
	cfg, err := loadConfig(*configPath)
	if err != nil {
		logger.Fatal("Failed to load config", zap.Error(err))
	}

	// Initialize Polymarket WebSocket client
	var polyWS *polymarket.WSClient
	if len(cfg.Polymarket.Markets) > 0 {
		polyWS = polymarket.NewWSClient(polymarket.WSClientConfig{
			Endpoint: cfg.Polymarket.WSEndpoint,
			Logger:   logger,
		})
		if err := polyWS.Connect(); err != nil {
			logger.Fatal("Failed to connect to Polymarket WebSocket", zap.Error(err))
		}
		defer polyWS.Close()

		for _, market := range cfg.Polymarket.Markets {
			if err := polyWS.SubscribeBook(market); err != nil {
				logger.Error("Failed to subscribe to book", zap.String("market", market), zap.Error(err))
			}
		}
	}

	// Initialize Binance WebSocket client
	var binanceWS *binance.SpotClient
	if len(cfg.Binance.Symbols) > 0 {
		binanceWS = binance.NewSpotClient(binance.SpotClientConfig{
			Logger: logger,
		})
		if err := binanceWS.Connect(cfg.Binance.Symbols); err != nil {
			logger.Fatal("Failed to connect to Binance WebSocket", zap.Error(err))
		}
		defer binanceWS.Close()
	}

	// Initialize collector
	coll := collector.NewCollector(collector.CollectorConfig{
		Logger:         logger,
		PolyWS:         polyWS,
		BinanceWS:      binanceWS,
		PolyMarkets:    cfg.Polymarket.Markets,
		BinanceSymbols: cfg.Binance.Symbols,
		OutputDir:      cfg.Collector.OutputDir,
	})

	if err := coll.Start(); err != nil {
		logger.Fatal("Failed to start collector", zap.Error(err))
	}
	defer coll.Stop()

	logger.Info("Collector started successfully")

	// Wait for shutdown signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	logger.Info("Shutting down...")
}

func loadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		// Return defaults if no config file
		return &Config{}, nil
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}

	return &cfg, nil
}
