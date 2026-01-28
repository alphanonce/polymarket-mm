// Package binance provides WebSocket clients for Binance market data
package binance

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

// WebSocket endpoints
const (
	SpotWSEndpoint    = "wss://stream.binance.com:9443/ws"
	FuturesWSEndpoint = "wss://fstream.binance.com/ws"
)

// SpotClient is the Binance spot WebSocket client
type SpotClient struct {
	endpoint string
	conn     *websocket.Conn
	logger   *zap.Logger

	// Subscriptions
	subscribedTickers map[string]bool
	subMu             sync.RWMutex

	// Price cache
	prices   map[string]*TickerPrice
	pricesMu sync.RWMutex

	// Callbacks
	onPriceUpdate func(symbol string, price *TickerPrice)
	onError       func(err error)

	// Control
	ctx                  context.Context
	cancel               context.CancelFunc
	wg                   sync.WaitGroup
	reconnectDelay       time.Duration
	maxReconnectDelay    time.Duration
	maxReconnectAttempts int
}

// TickerPrice represents current ticker price
type TickerPrice struct {
	Symbol      string
	Price       float64
	BidPrice    float64
	AskPrice    float64
	BidQty      float64
	AskQty      float64
	Timestamp   time.Time
}

// BookTickerMessage is the WebSocket book ticker message
type BookTickerMessage struct {
	UpdateID int64  `json:"u"`
	Symbol   string `json:"s"`
	BidPrice string `json:"b"`
	BidQty   string `json:"B"`
	AskPrice string `json:"a"`
	AskQty   string `json:"A"`
}

// MiniTickerMessage is the WebSocket mini ticker message
type MiniTickerMessage struct {
	EventType string `json:"e"`
	EventTime int64  `json:"E"`
	Symbol    string `json:"s"`
	Close     string `json:"c"`
	Open      string `json:"o"`
	High      string `json:"h"`
	Low       string `json:"l"`
	Volume    string `json:"v"`
	QuoteVol  string `json:"q"`
}

// SpotClientConfig configures the spot WebSocket client
type SpotClientConfig struct {
	Endpoint             string
	Logger               *zap.Logger
	ReconnectDelay       time.Duration
	MaxReconnectDelay    time.Duration
	MaxReconnectAttempts int
}

// NewSpotClient creates a new spot WebSocket client
func NewSpotClient(cfg SpotClientConfig) *SpotClient {
	if cfg.Endpoint == "" {
		cfg.Endpoint = SpotWSEndpoint
	}
	if cfg.Logger == nil {
		cfg.Logger, _ = zap.NewProduction()
	}
	if cfg.ReconnectDelay == 0 {
		cfg.ReconnectDelay = 5 * time.Second
	}
	if cfg.MaxReconnectDelay == 0 {
		cfg.MaxReconnectDelay = 60 * time.Second
	}
	if cfg.MaxReconnectAttempts == 0 {
		cfg.MaxReconnectAttempts = 10
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &SpotClient{
		endpoint:             cfg.Endpoint,
		logger:               cfg.Logger,
		subscribedTickers:    make(map[string]bool),
		prices:               make(map[string]*TickerPrice),
		ctx:                  ctx,
		cancel:               cancel,
		reconnectDelay:       cfg.ReconnectDelay,
		maxReconnectDelay:    cfg.MaxReconnectDelay,
		maxReconnectAttempts: cfg.MaxReconnectAttempts,
	}
}

// OnPriceUpdate sets the callback for price updates
func (c *SpotClient) OnPriceUpdate(fn func(symbol string, price *TickerPrice)) {
	c.onPriceUpdate = fn
}

// OnError sets the callback for errors
func (c *SpotClient) OnError(fn func(err error)) {
	c.onError = fn
}

// Connect establishes the WebSocket connection
func (c *SpotClient) Connect(symbols []string) error {
	if len(symbols) == 0 {
		return fmt.Errorf("no symbols specified")
	}

	// Build combined stream URL
	streams := make([]string, 0, len(symbols)*2)
	for _, sym := range symbols {
		lower := strings.ToLower(sym)
		streams = append(streams, lower+"@bookTicker")
		streams = append(streams, lower+"@miniTicker")
	}
	url := strings.TrimSuffix(c.endpoint, "/ws") + "/stream?streams=" + strings.Join(streams, "/")

	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	conn, _, err := dialer.DialContext(c.ctx, url, nil)
	if err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}

	c.conn = conn
	c.subMu.Lock()
	for _, sym := range symbols {
		c.subscribedTickers[sym] = true
	}
	c.subMu.Unlock()

	c.logger.Info("Connected to Binance spot WebSocket",
		zap.Strings("symbols", symbols))

	// Start message handler
	c.wg.Add(1)
	go c.readLoop()

	return nil
}

// Close closes the WebSocket connection
func (c *SpotClient) Close() error {
	c.cancel()
	if c.conn != nil {
		c.conn.Close()
	}
	c.wg.Wait()
	return nil
}

// GetPrice returns the current price for a symbol
func (c *SpotClient) GetPrice(symbol string) (*TickerPrice, bool) {
	c.pricesMu.RLock()
	defer c.pricesMu.RUnlock()
	price, ok := c.prices[symbol]
	return price, ok
}

// GetAllPrices returns all current prices
func (c *SpotClient) GetAllPrices() map[string]*TickerPrice {
	c.pricesMu.RLock()
	defer c.pricesMu.RUnlock()
	result := make(map[string]*TickerPrice, len(c.prices))
	for k, v := range c.prices {
		result[k] = v
	}
	return result
}

func (c *SpotClient) readLoop() {
	defer c.wg.Done()

	for {
		select {
		case <-c.ctx.Done():
			return
		default:
		}

		_, data, err := c.conn.ReadMessage()
		if err != nil {
			if c.ctx.Err() != nil {
				return
			}

			c.logger.Error("WebSocket read error", zap.Error(err))
			if c.onError != nil {
				c.onError(err)
			}

			if c.conn != nil {
				_ = c.conn.Close()
			}
			c.reconnect()
			return
		}

		c.handleMessage(data)
	}
}

func (c *SpotClient) handleMessage(data []byte) {
	// Try book ticker first
	var bookTicker BookTickerMessage
	if err := json.Unmarshal(data, &bookTicker); err == nil && bookTicker.Symbol != "" {
		c.handleBookTicker(&bookTicker)
		return
	}

	// Try mini ticker
	var miniTicker MiniTickerMessage
	if err := json.Unmarshal(data, &miniTicker); err == nil && miniTicker.Symbol != "" {
		c.handleMiniTicker(&miniTicker)
		return
	}
}

func (c *SpotClient) handleBookTicker(msg *BookTickerMessage) {
	bidPrice, _ := strconv.ParseFloat(msg.BidPrice, 64)
	askPrice, _ := strconv.ParseFloat(msg.AskPrice, 64)
	bidQty, _ := strconv.ParseFloat(msg.BidQty, 64)
	askQty, _ := strconv.ParseFloat(msg.AskQty, 64)

	c.pricesMu.Lock()
	price, exists := c.prices[msg.Symbol]
	if !exists {
		price = &TickerPrice{Symbol: msg.Symbol}
		c.prices[msg.Symbol] = price
	}
	price.BidPrice = bidPrice
	price.AskPrice = askPrice
	price.BidQty = bidQty
	price.AskQty = askQty
	price.Price = (bidPrice + askPrice) / 2 // Mid price
	price.Timestamp = time.Now()
	c.pricesMu.Unlock()

	if c.onPriceUpdate != nil {
		c.onPriceUpdate(msg.Symbol, price)
	}
}

func (c *SpotClient) handleMiniTicker(msg *MiniTickerMessage) {
	closePrice, _ := strconv.ParseFloat(msg.Close, 64)

	c.pricesMu.Lock()
	price, exists := c.prices[msg.Symbol]
	if !exists {
		price = &TickerPrice{Symbol: msg.Symbol}
		c.prices[msg.Symbol] = price
	}
	// Only update price if we don't have bid/ask
	if price.Price == 0 {
		price.Price = closePrice
	}
	price.Timestamp = time.Now()
	c.pricesMu.Unlock()

	if c.onPriceUpdate != nil {
		c.onPriceUpdate(msg.Symbol, price)
	}
}

func (c *SpotClient) reconnect() {
	// Close existing connection if any
	if c.conn != nil {
		_ = c.conn.Close()
		c.conn = nil
	}

	c.subMu.RLock()
	symbols := make([]string, 0, len(c.subscribedTickers))
	for sym := range c.subscribedTickers {
		symbols = append(symbols, sym)
	}
	c.subMu.RUnlock()

	for attempt := 0; attempt < c.maxReconnectAttempts; attempt++ {
		select {
		case <-c.ctx.Done():
			return
		default:
		}

		// Exponential backoff: 5s → 10s → 20s → 40s → max 60s
		delay := c.reconnectDelay * time.Duration(1<<uint(attempt))
		if delay > c.maxReconnectDelay {
			delay = c.maxReconnectDelay
		}

		c.logger.Info("Attempting to reconnect...",
			zap.Int("attempt", attempt+1),
			zap.Int("maxAttempts", c.maxReconnectAttempts),
			zap.Duration("delay", delay))

		select {
		case <-c.ctx.Done():
			return
		case <-time.After(delay):
		}

		if err := c.Connect(symbols); err != nil {
			c.logger.Error("Reconnect failed", zap.Error(err), zap.Int("attempt", attempt+1))
			continue
		}

		return
	}

	c.logger.Error("Max reconnect attempts reached, giving up",
		zap.Int("maxAttempts", c.maxReconnectAttempts))
	if c.onError != nil {
		c.onError(fmt.Errorf("max reconnect attempts (%d) reached", c.maxReconnectAttempts))
	}
}
