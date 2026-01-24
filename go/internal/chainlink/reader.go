// Package chainlink provides access to Chainlink price feeds on Polygon
package chainlink

import (
	"context"
	"fmt"
	"math/big"
	"strings"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum"
	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/ethclient"
	"go.uber.org/zap"
)

// Chainlink Aggregator V3 ABI (partial)
const aggregatorV3ABI = `[
	{
		"inputs": [],
		"name": "latestRoundData",
		"outputs": [
			{"internalType": "uint80", "name": "roundId", "type": "uint80"},
			{"internalType": "int256", "name": "answer", "type": "int256"},
			{"internalType": "uint256", "name": "startedAt", "type": "uint256"},
			{"internalType": "uint256", "name": "updatedAt", "type": "uint256"},
			{"internalType": "uint80", "name": "answeredInRound", "type": "uint80"}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "decimals",
		"outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "description",
		"outputs": [{"internalType": "string", "name": "", "type": "string"}],
		"stateMutability": "view",
		"type": "function"
	}
]`

// Common Chainlink price feed addresses on Polygon
var PolygonPriceFeeds = map[string]string{
	"BTC/USD":  "0xc907E116054Ad103354f2D350FD2514433D57F6f",
	"ETH/USD":  "0xF9680D99D6C9589e2a93a78A04A279e509205945",
	"MATIC/USD": "0xAB594600376Ec9fD91F8e885dADF0CE036862dE0",
	"USDC/USD": "0xfE4A8cc5b5B2366C1B58Bea3858e81843581b2F7",
	"USDT/USD": "0x0A6513e40db6EB1b165753AD52E80663aeA50545",
	"DAI/USD":  "0x4746DeC9e833A82EC7C2C1356372CcF2cfcD2F3D",
	"LINK/USD": "0xd9FFdb71EbE7496cC440152d43986Aae0AB76665",
	"SOL/USD":  "0x10C8264C0935b3B9870013e057f330Ff3e9C56dC",
}

// PriceFeed represents a Chainlink price feed
type PriceFeed struct {
	Symbol      string
	Address     common.Address
	Decimals    uint8
	Description string
}

// PriceData contains the latest price data from a feed
type PriceData struct {
	Symbol      string
	Price       float64
	RoundID     uint64
	UpdatedAt   time.Time
	FetchedAt   time.Time
}

// Reader reads prices from Chainlink oracles
type Reader struct {
	client      *ethclient.Client
	logger      *zap.Logger
	abi         abi.ABI

	// Feeds
	feeds   map[string]*PriceFeed
	feedsMu sync.RWMutex

	// Price cache
	prices   map[string]*PriceData
	pricesMu sync.RWMutex

	// Polling
	pollInterval time.Duration
	ctx          context.Context
	cancel       context.CancelFunc
	wg           sync.WaitGroup

	// Callbacks
	onPriceUpdate func(symbol string, data *PriceData)
}

// ReaderConfig configures the Chainlink reader
type ReaderConfig struct {
	RPCURL       string
	PollInterval time.Duration
	Logger       *zap.Logger
}

// NewReader creates a new Chainlink reader
func NewReader(cfg ReaderConfig) (*Reader, error) {
	client, err := ethclient.Dial(cfg.RPCURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to RPC: %w", err)
	}

	if cfg.Logger == nil {
		cfg.Logger, _ = zap.NewProduction()
	}
	if cfg.PollInterval == 0 {
		cfg.PollInterval = 1 * time.Second
	}

	parsedABI, err := abi.JSON(strings.NewReader(aggregatorV3ABI))
	if err != nil {
		return nil, fmt.Errorf("failed to parse ABI: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &Reader{
		client:       client,
		logger:       cfg.Logger,
		abi:          parsedABI,
		feeds:        make(map[string]*PriceFeed),
		prices:       make(map[string]*PriceData),
		pollInterval: cfg.PollInterval,
		ctx:          ctx,
		cancel:       cancel,
	}, nil
}

// OnPriceUpdate sets the callback for price updates
func (r *Reader) OnPriceUpdate(fn func(symbol string, data *PriceData)) {
	r.onPriceUpdate = fn
}

// AddFeed adds a price feed to monitor
func (r *Reader) AddFeed(symbol, address string) error {
	addr := common.HexToAddress(address)

	// Get decimals
	decimals, err := r.getDecimals(addr)
	if err != nil {
		return fmt.Errorf("failed to get decimals: %w", err)
	}

	// Get description
	desc, err := r.getDescription(addr)
	if err != nil {
		r.logger.Warn("Failed to get description", zap.String("symbol", symbol), zap.Error(err))
		desc = symbol
	}

	r.feedsMu.Lock()
	r.feeds[symbol] = &PriceFeed{
		Symbol:      symbol,
		Address:     addr,
		Decimals:    decimals,
		Description: desc,
	}
	r.feedsMu.Unlock()

	r.logger.Info("Added price feed",
		zap.String("symbol", symbol),
		zap.String("address", address),
		zap.Uint8("decimals", decimals))

	return nil
}

// AddDefaultFeeds adds the default Polygon price feeds
func (r *Reader) AddDefaultFeeds() error {
	for symbol, address := range PolygonPriceFeeds {
		if err := r.AddFeed(symbol, address); err != nil {
			r.logger.Warn("Failed to add feed", zap.String("symbol", symbol), zap.Error(err))
		}
	}
	return nil
}

// GetPrice returns the cached price for a symbol
func (r *Reader) GetPrice(symbol string) (*PriceData, bool) {
	r.pricesMu.RLock()
	defer r.pricesMu.RUnlock()
	data, ok := r.prices[symbol]
	return data, ok
}

// GetAllPrices returns all cached prices
func (r *Reader) GetAllPrices() map[string]*PriceData {
	r.pricesMu.RLock()
	defer r.pricesMu.RUnlock()
	result := make(map[string]*PriceData, len(r.prices))
	for k, v := range r.prices {
		result[k] = v
	}
	return result
}

// FetchPrice fetches the latest price for a feed
func (r *Reader) FetchPrice(ctx context.Context, symbol string) (*PriceData, error) {
	r.feedsMu.RLock()
	feed, ok := r.feeds[symbol]
	r.feedsMu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("feed not found: %s", symbol)
	}

	data, err := r.abi.Pack("latestRoundData")
	if err != nil {
		return nil, fmt.Errorf("failed to pack call: %w", err)
	}

	msg := ethereum.CallMsg{
		To:   &feed.Address,
		Data: data,
	}

	result, err := r.client.CallContract(ctx, msg, nil)
	if err != nil {
		return nil, fmt.Errorf("call failed: %w", err)
	}

	outputs, err := r.abi.Unpack("latestRoundData", result)
	if err != nil {
		return nil, fmt.Errorf("failed to unpack result: %w", err)
	}

	roundID := outputs[0].(*big.Int).Uint64()
	answer := outputs[1].(*big.Int)
	updatedAt := outputs[3].(*big.Int).Int64()

	// Convert to float with decimals
	divisor := new(big.Float).SetInt(new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(feed.Decimals)), nil))
	price, _ := new(big.Float).Quo(new(big.Float).SetInt(answer), divisor).Float64()

	priceData := &PriceData{
		Symbol:    symbol,
		Price:     price,
		RoundID:   roundID,
		UpdatedAt: time.Unix(updatedAt, 0),
		FetchedAt: time.Now(),
	}

	r.pricesMu.Lock()
	r.prices[symbol] = priceData
	r.pricesMu.Unlock()

	return priceData, nil
}

// StartPolling starts the polling loop
func (r *Reader) StartPolling() {
	r.wg.Add(1)
	go r.pollLoop()
}

// Stop stops the reader
func (r *Reader) Stop() {
	r.cancel()
	r.wg.Wait()
	r.client.Close()
}

func (r *Reader) pollLoop() {
	defer r.wg.Done()

	ticker := time.NewTicker(r.pollInterval)
	defer ticker.Stop()

	// Initial fetch
	r.fetchAllPrices()

	for {
		select {
		case <-r.ctx.Done():
			return
		case <-ticker.C:
			r.fetchAllPrices()
		}
	}
}

func (r *Reader) fetchAllPrices() {
	r.feedsMu.RLock()
	symbols := make([]string, 0, len(r.feeds))
	for symbol := range r.feeds {
		symbols = append(symbols, symbol)
	}
	r.feedsMu.RUnlock()

	for _, symbol := range symbols {
		data, err := r.FetchPrice(r.ctx, symbol)
		if err != nil {
			r.logger.Error("Failed to fetch price",
				zap.String("symbol", symbol),
				zap.Error(err))
			continue
		}

		if r.onPriceUpdate != nil {
			r.onPriceUpdate(symbol, data)
		}
	}
}

func (r *Reader) getDecimals(addr common.Address) (uint8, error) {
	data, err := r.abi.Pack("decimals")
	if err != nil {
		return 0, err
	}

	msg := ethereum.CallMsg{
		To:   &addr,
		Data: data,
	}

	result, err := r.client.CallContract(r.ctx, msg, nil)
	if err != nil {
		return 0, err
	}

	outputs, err := r.abi.Unpack("decimals", result)
	if err != nil {
		return 0, err
	}

	return outputs[0].(uint8), nil
}

func (r *Reader) getDescription(addr common.Address) (string, error) {
	data, err := r.abi.Pack("description")
	if err != nil {
		return "", err
	}

	msg := ethereum.CallMsg{
		To:   &addr,
		Data: data,
	}

	result, err := r.client.CallContract(r.ctx, msg, nil)
	if err != nil {
		return "", err
	}

	outputs, err := r.abi.Unpack("description", result)
	if err != nil {
		return "", err
	}

	return outputs[0].(string), nil
}
