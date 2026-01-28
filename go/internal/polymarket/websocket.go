package polymarket

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

// BookSnapshotMessage is the actual format from Polymarket WebSocket
type BookSnapshotMessage struct {
	Market    string      `json:"market"`
	AssetID   string      `json:"asset_id"`
	Hash      string      `json:"hash"`
	Timestamp string      `json:"timestamp"`
	Bids      []BookLevel `json:"bids"`
	Asks      []BookLevel `json:"asks"`
}

// TradesMessage contains trade updates
type TradesMessage struct {
	Market  string  `json:"market"`
	Trades  []Trade `json:"trades"`
}

// WSClient is the Polymarket WebSocket client
type WSClient struct {
	endpoint string
	conn     *websocket.Conn
	logger   *zap.Logger

	// Subscriptions
	subscribedBooks  map[string]bool
	subscribedTrades map[string]bool
	subMu            sync.RWMutex

	// Orderbook state
	orderbooks   map[string]*OrderbookState
	orderbooksMu sync.RWMutex

	// Callbacks
	onBookUpdate  func(assetID string, book *OrderbookState)
	onTrade       func(trade Trade)
	onError       func(err error)
	onReconnect   func()

	// Control
	ctx              context.Context
	cancel           context.CancelFunc
	wg               sync.WaitGroup
	reconnectDelay   time.Duration
	maxReconnectDelay time.Duration
	maxReconnectAttempts int
}

// WSClientConfig configures the WebSocket client
type WSClientConfig struct {
	Endpoint             string
	Logger               *zap.Logger
	ReconnectDelay       time.Duration
	MaxReconnectDelay    time.Duration
	MaxReconnectAttempts int
}

// NewWSClient creates a new WebSocket client
func NewWSClient(cfg WSClientConfig) *WSClient {
	if cfg.Endpoint == "" {
		cfg.Endpoint = WSEndpointMainnet
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

	return &WSClient{
		endpoint:             cfg.Endpoint,
		logger:               cfg.Logger,
		subscribedBooks:      make(map[string]bool),
		subscribedTrades:     make(map[string]bool),
		orderbooks:           make(map[string]*OrderbookState),
		ctx:                  ctx,
		cancel:               cancel,
		reconnectDelay:       cfg.ReconnectDelay,
		maxReconnectDelay:    cfg.MaxReconnectDelay,
		maxReconnectAttempts: cfg.MaxReconnectAttempts,
	}
}

// OnBookUpdate sets the callback for orderbook updates
func (c *WSClient) OnBookUpdate(fn func(assetID string, book *OrderbookState)) {
	c.onBookUpdate = fn
}

// OnTrade sets the callback for trades
func (c *WSClient) OnTrade(fn func(trade Trade)) {
	c.onTrade = fn
}

// OnError sets the callback for errors
func (c *WSClient) OnError(fn func(err error)) {
	c.onError = fn
}

// OnReconnect sets the callback for reconnections
func (c *WSClient) OnReconnect(fn func()) {
	c.onReconnect = fn
}

// Connect establishes the WebSocket connection
func (c *WSClient) Connect() error {
	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	conn, _, err := dialer.DialContext(c.ctx, c.endpoint, nil)
	if err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}

	c.conn = conn
	c.logger.Info("Connected to Polymarket WebSocket", zap.String("endpoint", c.endpoint))

	// Start message handler
	c.wg.Add(1)
	go c.readLoop()

	return nil
}

// Close closes the WebSocket connection
func (c *WSClient) Close() error {
	c.cancel()
	if c.conn != nil {
		c.conn.Close()
	}
	c.wg.Wait()
	return nil
}

// SubscribeBook subscribes to orderbook updates for an asset
func (c *WSClient) SubscribeBook(assetID string) error {
	return c.SubscribeBooksMany([]string{assetID})
}

// SubscribeBooksMany subscribes to orderbook updates for multiple assets in a single message.
// This is important because Polymarket's WebSocket only sends initial snapshots for tokens
// included in the same subscription message.
func (c *WSClient) SubscribeBooksMany(assetIDs []string) error {
	c.subMu.Lock()
	defer c.subMu.Unlock()

	// Filter out already subscribed assets
	newAssets := make([]string, 0, len(assetIDs))
	for _, assetID := range assetIDs {
		if !c.subscribedBooks[assetID] {
			newAssets = append(newAssets, assetID)
		}
	}

	if len(newAssets) == 0 {
		return nil
	}

	msg := SubscribeMessage{
		Type:     MsgTypeSubscribe,
		Channel:  ChannelBook,
		AssetIDs: newAssets,
	}

	if err := c.send(msg); err != nil {
		return err
	}

	// Mark all as subscribed and create orderbook states
	c.orderbooksMu.Lock()
	for _, assetID := range newAssets {
		c.subscribedBooks[assetID] = true
		c.orderbooks[assetID] = NewOrderbookState(assetID)
	}
	c.orderbooksMu.Unlock()

	c.logger.Info("Subscribed to books", zap.Int("count", len(newAssets)))
	return nil
}

// SubscribeTrades subscribes to trade updates for an asset
func (c *WSClient) SubscribeTrades(assetID string) error {
	c.subMu.Lock()
	defer c.subMu.Unlock()

	if c.subscribedTrades[assetID] {
		return nil
	}

	msg := SubscribeMessage{
		Type:     MsgTypeSubscribe,
		Channel:  ChannelTrades,
		AssetIDs: []string{assetID},
	}

	if err := c.send(msg); err != nil {
		return err
	}

	c.subscribedTrades[assetID] = true
	c.logger.Info("Subscribed to trades", zap.String("assetID", assetID))
	return nil
}

// GetOrderbook returns the current orderbook state for an asset
func (c *WSClient) GetOrderbook(assetID string) (*OrderbookState, bool) {
	c.orderbooksMu.RLock()
	defer c.orderbooksMu.RUnlock()
	ob, ok := c.orderbooks[assetID]
	return ob, ok
}

func (c *WSClient) send(msg any) error {
	if c.conn == nil {
		return fmt.Errorf("not connected")
	}
	return c.conn.WriteJSON(msg)
}

func (c *WSClient) readLoop() {
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

			// Attempt reconnect - return to prevent double read loops
			c.reconnect()
			return
		}

		c.handleMessage(data)
	}
}

func (c *WSClient) handleMessage(data []byte) {
	// Check if message is an array (initial snapshot) or object (update)
	data = bytes.TrimSpace(data)
	if len(data) == 0 {
		return
	}

	if data[0] == '[' {
		// Array of book snapshots
		var snapshots []BookSnapshotMessage
		if err := json.Unmarshal(data, &snapshots); err != nil {
			c.logger.Debug("Failed to parse array message", zap.Error(err))
			return
		}
		for _, snap := range snapshots {
			c.handleBookSnapshot(&snap)
		}
	} else if data[0] == '{' {
		// Single message - could be book update, price_change, or trade
		var raw map[string]json.RawMessage
		if err := json.Unmarshal(data, &raw); err != nil {
			c.logger.Debug("Failed to parse message", zap.Error(err))
			return
		}

		// Check event_type for price_change messages (legacy format)
		if eventType, ok := raw["event_type"]; ok {
			var et string
			json.Unmarshal(eventType, &et)
			if et == "price_change" {
				// Skip old format price_change events
				return
			}
		}

		// Check if it's a price_changes message (new format)
		if priceChanges, ok := raw["price_changes"]; ok {
			c.handlePriceChanges(priceChanges)
			return
		}

		// Check if it's a book update (has bids/asks at top level)
		if _, hasBids := raw["bids"]; hasBids {
			var bookMsg BookSnapshotMessage
			if err := json.Unmarshal(data, &bookMsg); err != nil {
				c.logger.Debug("Failed to parse book message", zap.Error(err))
				return
			}
			c.handleBookSnapshot(&bookMsg)
			return
		}

		// Check if it's a trade message
		if _, hasTrades := raw["trades"]; hasTrades {
			var tradeMsg TradesMessage
			if err := json.Unmarshal(data, &tradeMsg); err != nil {
				c.logger.Debug("Failed to parse trades message", zap.Error(err))
				return
			}
			c.handleTradesMsg(&tradeMsg)
			return
		}

		// Try legacy format
		var msg WSMessage
		if err := json.Unmarshal(data, &msg); err != nil {
			c.logger.Debug("Failed to parse legacy message", zap.Error(err))
			return
		}

		switch msg.Channel {
		case ChannelBook:
			c.handleBookMessage(&msg)
		case ChannelTrades:
			c.handleTradesMessage(&msg)
		}
	}
}

func (c *WSClient) handleBookSnapshot(msg *BookSnapshotMessage) {
	assetID := msg.AssetID
	if assetID == "" {
		c.logger.Debug("handleBookSnapshot: empty asset ID")
		return
	}
	c.logger.Info("Book snapshot received", zap.String("assetID", assetID[:20]+"..."), zap.Int("bids", len(msg.Bids)), zap.Int("asks", len(msg.Asks)))

	c.orderbooksMu.Lock()
	ob, exists := c.orderbooks[assetID]
	if !exists {
		ob = NewOrderbookState(assetID)
		c.orderbooks[assetID] = ob
	}

	// Apply as snapshot
	ob.Bids = make(map[string]string)
	ob.Asks = make(map[string]string)
	for _, level := range msg.Bids {
		ob.Bids[level.Price] = level.Size
	}
	for _, level := range msg.Asks {
		ob.Asks[level.Price] = level.Size
	}
	ob.LastHash = msg.Hash
	ob.LastUpdateTime = time.Now()
	c.orderbooksMu.Unlock()

	if c.onBookUpdate != nil {
		c.onBookUpdate(assetID, ob)
	}
}

// PriceChange represents a single price change in the price_changes array
type PriceChange struct {
	AssetID string `json:"asset_id"`
	Price   string `json:"price"`
	Size    string `json:"size"`
	Side    string `json:"side"`
	Hash    string `json:"hash"`
}

func (c *WSClient) handlePriceChanges(raw json.RawMessage) {
	var changes []PriceChange
	if err := json.Unmarshal(raw, &changes); err != nil {
		c.logger.Debug("Failed to parse price_changes", zap.Error(err))
		return
	}
	c.logger.Debug("Price changes received", zap.Int("count", len(changes)))

	// Group changes by asset ID
	assetChanges := make(map[string][]PriceChange)
	for _, change := range changes {
		assetChanges[change.AssetID] = append(assetChanges[change.AssetID], change)
	}

	// Apply changes to each orderbook
	for assetID, changes := range assetChanges {
		c.orderbooksMu.Lock()
		ob, exists := c.orderbooks[assetID]
		if !exists {
			ob = NewOrderbookState(assetID)
			c.orderbooks[assetID] = ob
		}

		for _, change := range changes {
			if change.Side == "BUY" {
				if change.Size == "0" || change.Size == "" {
					delete(ob.Bids, change.Price)
				} else {
					ob.Bids[change.Price] = change.Size
				}
			} else if change.Side == "SELL" {
				if change.Size == "0" || change.Size == "" {
					delete(ob.Asks, change.Price)
				} else {
					ob.Asks[change.Price] = change.Size
				}
			}
			ob.LastHash = change.Hash
		}
		ob.LastUpdateTime = time.Now()
		c.orderbooksMu.Unlock()

		if c.onBookUpdate != nil {
			c.onBookUpdate(assetID, ob)
		}
	}
}

func (c *WSClient) handleTradesMsg(msg *TradesMessage) {
	if c.onTrade == nil {
		return
	}
	for _, trade := range msg.Trades {
		c.onTrade(trade)
	}
}

func (c *WSClient) handleBookMessage(msg *WSMessage) {
	assetID := msg.AssetID
	if assetID == "" {
		if msg.Book != nil {
			assetID = msg.Book.AssetID
		} else if msg.Delta != nil {
			assetID = msg.Delta.AssetID
		}
	}

	c.orderbooksMu.Lock()
	ob, exists := c.orderbooks[assetID]
	if !exists {
		ob = NewOrderbookState(assetID)
		c.orderbooks[assetID] = ob
	}

	if msg.Book != nil {
		ob.ApplySnapshot(msg.Book)
	} else if msg.Delta != nil {
		ob.ApplyDelta(msg.Delta)
	}
	c.orderbooksMu.Unlock()

	if c.onBookUpdate != nil {
		c.onBookUpdate(assetID, ob)
	}
}

func (c *WSClient) handleTradesMessage(msg *WSMessage) {
	if c.onTrade == nil {
		return
	}

	for _, trade := range msg.Trades {
		c.onTrade(trade)
	}
}

func (c *WSClient) reconnect() {
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

		if err := c.Connect(); err != nil {
			c.logger.Error("Reconnect failed", zap.Error(err), zap.Int("attempt", attempt+1))
			continue
		}

		// Resubscribe
		c.resubscribe()

		if c.onReconnect != nil {
			c.onReconnect()
		}

		return
	}

	c.logger.Error("Max reconnect attempts reached, giving up",
		zap.Int("maxAttempts", c.maxReconnectAttempts))
	if c.onError != nil {
		c.onError(fmt.Errorf("max reconnect attempts (%d) reached", c.maxReconnectAttempts))
	}
}

func (c *WSClient) resubscribe() {
	c.subMu.RLock()
	defer c.subMu.RUnlock()

	for assetID := range c.subscribedBooks {
		msg := SubscribeMessage{
			Type:     MsgTypeSubscribe,
			Channel:  ChannelBook,
			AssetIDs: []string{assetID},
		}
		if err := c.send(msg); err != nil {
			c.logger.Error("Failed to resubscribe to book", zap.String("assetID", assetID), zap.Error(err))
		}
	}

	for assetID := range c.subscribedTrades {
		msg := SubscribeMessage{
			Type:     MsgTypeSubscribe,
			Channel:  ChannelTrades,
			AssetIDs: []string{assetID},
		}
		if err := c.send(msg); err != nil {
			c.logger.Error("Failed to resubscribe to trades", zap.String("assetID", assetID), zap.Error(err))
		}
	}
}

// RemoveOrderbook removes an orderbook and its subscription
func (c *WSClient) RemoveOrderbook(assetID string) {
	c.orderbooksMu.Lock()
	delete(c.orderbooks, assetID)
	c.orderbooksMu.Unlock()

	c.subMu.Lock()
	delete(c.subscribedBooks, assetID)
	delete(c.subscribedTrades, assetID)
	c.subMu.Unlock()
}

// CleanupStaleOrderbooks removes orderbooks that haven't been updated for the specified duration
func (c *WSClient) CleanupStaleOrderbooks(maxAge time.Duration) int {
	c.orderbooksMu.Lock()
	defer c.orderbooksMu.Unlock()

	now := time.Now()
	removed := 0
	var staleIDs []string

	for assetID, ob := range c.orderbooks {
		if now.Sub(ob.LastUpdateTime) > maxAge {
			staleIDs = append(staleIDs, assetID)
		}
	}

	for _, assetID := range staleIDs {
		delete(c.orderbooks, assetID)
		removed++
	}

	// Also clean up subscriptions
	if removed > 0 {
		c.subMu.Lock()
		for _, assetID := range staleIDs {
			delete(c.subscribedBooks, assetID)
			delete(c.subscribedTrades, assetID)
		}
		c.subMu.Unlock()
		c.logger.Info("Cleaned up stale orderbooks", zap.Int("removed", removed))
	}

	return removed
}

// GetOrderbookCount returns the current number of tracked orderbooks
func (c *WSClient) GetOrderbookCount() int {
	c.orderbooksMu.RLock()
	defer c.orderbooksMu.RUnlock()
	return len(c.orderbooks)
}
