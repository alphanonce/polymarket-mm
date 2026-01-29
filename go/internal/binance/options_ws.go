// Package binance provides WebSocket clients for Binance market data
package binance

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

// Options WebSocket endpoint
const (
	OptionsWSEndpoint = "wss://nbstream.binance.com/eoptions/ws"
)

// OptionsWSClient is the Binance options WebSocket client for streaming IV data
type OptionsWSClient struct {
	endpoint string
	conn     *websocket.Conn
	logger   *zap.Logger

	// Subscriptions
	underlyings    []string
	subscribedOpts map[string]bool
	subMu          sync.RWMutex

	// IV data cache - maps symbol to term structure
	ivData   map[string]*IVData
	ivDataMu sync.RWMutex

	// Callbacks
	onIVUpdate func(symbol string, data *IVData)
	onError    func(err error)

	// Control
	ctx                  context.Context
	cancel               context.CancelFunc
	wg                   sync.WaitGroup
	reconnectDelay       time.Duration
	maxReconnectDelay    time.Duration
	maxReconnectAttempts int
}

// IVTenorPoint represents IV at a specific tenor
type IVTenorPoint struct {
	DaysToExpiry float64
	ATMIV        float64
}

// IVData represents aggregated IV data for an underlying
type IVData struct {
	Symbol        string
	ATMIV         float64 // Shortest tenor IV (primary)
	TermStructure []IVTenorPoint
	Timestamp     time.Time
}

// MarkPriceMessage is the Binance options mark price message
type MarkPriceMessage struct {
	EventType       string `json:"e"`  // "markPrice"
	EventTime       int64  `json:"E"`  // Event time
	Symbol          string `json:"s"`  // Option symbol e.g., "BTC-250131-100000-C"
	MarkPrice       string `json:"mp"` // Mark price
	MarkIV          string `json:"v"`  // Mark implied volatility
	UnderlyingIndex string `json:"u"`  // Underlying index price
	Delta           string `json:"d"`  // Delta
	Gamma           string `json:"g"`  // Gamma
	Theta           string `json:"t"`  // Theta
	Vega            string `json:"V"`  // Vega
}

// CombinedStreamMessage wraps messages from combined streams
type CombinedStreamMessage struct {
	Stream string          `json:"stream"`
	Data   json.RawMessage `json:"data"`
}

// OptionsWSClientConfig configures the options WebSocket client
type OptionsWSClientConfig struct {
	Endpoint             string
	Logger               *zap.Logger
	Underlyings          []string // e.g., ["BTCUSDT", "ETHUSDT"]
	ReconnectDelay       time.Duration
	MaxReconnectDelay    time.Duration
	MaxReconnectAttempts int
}

// NewOptionsWSClient creates a new options WebSocket client for streaming IV data
func NewOptionsWSClient(cfg OptionsWSClientConfig) *OptionsWSClient {
	if cfg.Endpoint == "" {
		cfg.Endpoint = OptionsWSEndpoint
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

	return &OptionsWSClient{
		endpoint:             cfg.Endpoint,
		logger:               cfg.Logger,
		underlyings:          cfg.Underlyings,
		subscribedOpts:       make(map[string]bool),
		ivData:               make(map[string]*IVData),
		ctx:                  ctx,
		cancel:               cancel,
		reconnectDelay:       cfg.ReconnectDelay,
		maxReconnectDelay:    cfg.MaxReconnectDelay,
		maxReconnectAttempts: cfg.MaxReconnectAttempts,
	}
}

// OnIVUpdate sets the callback for IV updates
func (c *OptionsWSClient) OnIVUpdate(fn func(symbol string, data *IVData)) {
	c.onIVUpdate = fn
}

// OnError sets the callback for errors
func (c *OptionsWSClient) OnError(fn func(err error)) {
	c.onError = fn
}

// Connect establishes the WebSocket connection
func (c *OptionsWSClient) Connect() error {
	if len(c.underlyings) == 0 {
		return fmt.Errorf("no underlyings specified")
	}

	// Build stream URL for markPrice of each underlying
	// Format: {underlying}@markPrice (e.g., btcusdt@markPrice)
	streams := make([]string, 0, len(c.underlyings))
	for _, underlying := range c.underlyings {
		// Convert BTCUSDT -> BTC for options underlying
		baseAsset := strings.TrimSuffix(strings.ToUpper(underlying), "USDT")
		streams = append(streams, strings.ToLower(baseAsset)+"@markPrice")
	}
	url := c.endpoint + "/stream?streams=" + strings.Join(streams, "/")

	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	conn, _, err := dialer.DialContext(c.ctx, url, nil)
	if err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}

	c.conn = conn
	c.subMu.Lock()
	for _, u := range c.underlyings {
		c.subscribedOpts[u] = true
	}
	c.subMu.Unlock()

	c.logger.Info("Connected to Binance options WebSocket",
		zap.Strings("underlyings", c.underlyings))

	// Start message handler
	c.wg.Add(1)
	go c.readLoop()

	return nil
}

// Close closes the WebSocket connection
func (c *OptionsWSClient) Close() error {
	c.cancel()
	if c.conn != nil {
		c.conn.Close()
	}
	c.wg.Wait()
	return nil
}

// GetIVData returns the current IV data for a symbol (deep copy to prevent data races)
func (c *OptionsWSClient) GetIVData(symbol string) (*IVData, bool) {
	c.ivDataMu.RLock()
	defer c.ivDataMu.RUnlock()
	data, ok := c.ivData[symbol]
	if !ok {
		return nil, false
	}
	return copyIVData(data), true
}

// GetAllIVData returns all current IV data (deep copies to prevent data races)
func (c *OptionsWSClient) GetAllIVData() map[string]*IVData {
	c.ivDataMu.RLock()
	defer c.ivDataMu.RUnlock()
	result := make(map[string]*IVData, len(c.ivData))
	for k, v := range c.ivData {
		result[k] = copyIVData(v)
	}
	return result
}

// copyIVData creates a deep copy of IVData to prevent data races
func copyIVData(data *IVData) *IVData {
	if data == nil {
		return nil
	}
	termStructure := make([]IVTenorPoint, len(data.TermStructure))
	copy(termStructure, data.TermStructure)
	return &IVData{
		Symbol:        data.Symbol,
		ATMIV:         data.ATMIV,
		TermStructure: termStructure,
		Timestamp:     data.Timestamp,
	}
}

func (c *OptionsWSClient) readLoop() {
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

func (c *OptionsWSClient) handleMessage(data []byte) {
	// Try combined stream format first
	var combined CombinedStreamMessage
	if err := json.Unmarshal(data, &combined); err == nil && combined.Stream != "" {
		c.handleStreamData(combined.Stream, combined.Data)
		return
	}

	// Try direct mark price message
	var markPrice MarkPriceMessage
	if err := json.Unmarshal(data, &markPrice); err == nil && markPrice.Symbol != "" {
		c.handleMarkPrice(&markPrice)
		return
	}

	// Try array of mark price messages (batch update)
	var markPrices []MarkPriceMessage
	if err := json.Unmarshal(data, &markPrices); err == nil && len(markPrices) > 0 {
		for i := range markPrices {
			c.handleMarkPrice(&markPrices[i])
		}
		return
	}
}

func (c *OptionsWSClient) handleStreamData(stream string, data json.RawMessage) {
	// Stream format: btc@markPrice
	// Data is an array of mark price messages for all BTC options

	var markPrices []MarkPriceMessage
	if err := json.Unmarshal(data, &markPrices); err != nil {
		// Try single message
		var markPrice MarkPriceMessage
		if err := json.Unmarshal(data, &markPrice); err == nil && markPrice.Symbol != "" {
			c.handleMarkPrice(&markPrice)
		}
		return
	}

	// Process batch of mark prices
	for i := range markPrices {
		c.handleMarkPrice(&markPrices[i])
	}
}

func (c *OptionsWSClient) handleMarkPrice(msg *MarkPriceMessage) {
	// Parse option symbol to extract underlying and expiry
	// Format: BTC-250131-100000-C (underlying-expiry-strike-type)
	parts := strings.Split(msg.Symbol, "-")
	if len(parts) != 4 {
		return
	}

	underlying := parts[0] + "USDT" // BTC -> BTCUSDT
	expiryStr := parts[1]          // 250131 (YYMMDD)
	strikeStr := parts[2]          // 100000
	optType := parts[3]            // C or P

	// Only process ATM-ish calls for cleaner IV
	if optType != "C" {
		return
	}

	// Parse expiry date
	expiry, err := time.Parse("060102", expiryStr)
	if err != nil {
		return
	}

	// Calculate days to expiry
	daysToExpiry := time.Until(expiry).Hours() / 24
	if daysToExpiry < 0 {
		return // Expired
	}

	// Parse mark IV
	markIV, err := strconv.ParseFloat(msg.MarkIV, 64)
	if err != nil || markIV <= 0 {
		return
	}

	// Parse strike and underlying to check ATM-ness
	strike, err := strconv.ParseFloat(strikeStr, 64)
	if err != nil {
		return
	}
	underlyingPrice, err := strconv.ParseFloat(msg.UnderlyingIndex, 64)
	if err != nil || underlyingPrice <= 0 {
		return
	}

	// Check if roughly ATM (within 10% of underlying)
	moneyness := strike / underlyingPrice
	if moneyness < 0.9 || moneyness > 1.1 {
		return // Not ATM enough
	}

	// Update IV data for this underlying
	c.updateIVData(underlying, daysToExpiry, markIV)
}

func (c *OptionsWSClient) updateIVData(underlying string, daysToExpiry float64, iv float64) {
	// Update data while holding lock, prepare callback data
	var dataCopy *IVData

	c.ivDataMu.Lock()
	data, exists := c.ivData[underlying]
	if !exists {
		data = &IVData{
			Symbol:        underlying,
			TermStructure: make([]IVTenorPoint, 0, 10),
		}
		c.ivData[underlying] = data
	}

	// Update or add tenor point
	found := false
	for i := range data.TermStructure {
		// Consider same tenor if within 0.5 days
		if abs(data.TermStructure[i].DaysToExpiry-daysToExpiry) < 0.5 {
			// Exponential moving average for smoothing
			alpha := 0.3
			data.TermStructure[i].ATMIV = alpha*iv + (1-alpha)*data.TermStructure[i].ATMIV
			data.TermStructure[i].DaysToExpiry = daysToExpiry
			found = true
			break
		}
	}

	if !found {
		data.TermStructure = append(data.TermStructure, IVTenorPoint{
			DaysToExpiry: daysToExpiry,
			ATMIV:        iv,
		})
	}

	// Sort by days to expiry
	sort.Slice(data.TermStructure, func(i, j int) bool {
		return data.TermStructure[i].DaysToExpiry < data.TermStructure[j].DaysToExpiry
	})

	// Keep only the first 6 tenors (matching MAX_IV_TENORS)
	if len(data.TermStructure) > 6 {
		data.TermStructure = data.TermStructure[:6]
	}

	// Update shortest tenor as primary ATM IV
	if len(data.TermStructure) > 0 {
		data.ATMIV = data.TermStructure[0].ATMIV
	}

	data.Timestamp = time.Now()

	// Prepare callback data while still holding lock
	if c.onIVUpdate != nil {
		dataCopy = &IVData{
			Symbol:        data.Symbol,
			ATMIV:         data.ATMIV,
			TermStructure: make([]IVTenorPoint, len(data.TermStructure)),
			Timestamp:     data.Timestamp,
		}
		copy(dataCopy.TermStructure, data.TermStructure)
	}
	c.ivDataMu.Unlock()

	// Invoke callback outside lock to prevent deadlock if callback
	// calls GetIVData/GetAllIVData
	if c.onIVUpdate != nil && dataCopy != nil {
		c.onIVUpdate(underlying, dataCopy)
	}
}

func (c *OptionsWSClient) reconnect() {
	// Close existing connection if any
	if c.conn != nil {
		_ = c.conn.Close()
		c.conn = nil
	}

	for attempt := 0; attempt < c.maxReconnectAttempts; attempt++ {
		select {
		case <-c.ctx.Done():
			return
		default:
		}

		// Exponential backoff
		delay := c.reconnectDelay * time.Duration(1<<uint(attempt))
		if delay > c.maxReconnectDelay {
			delay = c.maxReconnectDelay
		}

		c.logger.Info("Attempting to reconnect options WebSocket...",
			zap.Int("attempt", attempt+1),
			zap.Int("maxAttempts", c.maxReconnectAttempts),
			zap.Duration("delay", delay))

		select {
		case <-c.ctx.Done():
			return
		case <-time.After(delay):
		}

		if err := c.Connect(); err != nil {
			c.logger.Error("Reconnect failed", zap.Error(err), zap.Int("attempt", attempt+1))
			continue
		}

		return
	}

	c.logger.Error("Max reconnect attempts reached for options WebSocket, giving up",
		zap.Int("maxAttempts", c.maxReconnectAttempts))
	if c.onError != nil {
		c.onError(fmt.Errorf("max reconnect attempts (%d) reached", c.maxReconnectAttempts))
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
