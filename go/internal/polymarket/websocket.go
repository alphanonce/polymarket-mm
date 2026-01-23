package polymarket

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

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
	ctx        context.Context
	cancel     context.CancelFunc
	wg         sync.WaitGroup
	reconnectDelay time.Duration
}

// WSClientConfig configures the WebSocket client
type WSClientConfig struct {
	Endpoint       string
	Logger         *zap.Logger
	ReconnectDelay time.Duration
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

	ctx, cancel := context.WithCancel(context.Background())

	return &WSClient{
		endpoint:         cfg.Endpoint,
		logger:           cfg.Logger,
		subscribedBooks:  make(map[string]bool),
		subscribedTrades: make(map[string]bool),
		orderbooks:       make(map[string]*OrderbookState),
		ctx:              ctx,
		cancel:           cancel,
		reconnectDelay:   cfg.ReconnectDelay,
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
	c.subMu.Lock()
	defer c.subMu.Unlock()

	if c.subscribedBooks[assetID] {
		return nil
	}

	msg := SubscribeMessage{
		Type:     MsgTypeSubscribe,
		Channel:  ChannelBook,
		AssetIDs: []string{assetID},
	}

	if err := c.send(msg); err != nil {
		return err
	}

	c.subscribedBooks[assetID] = true
	c.orderbooksMu.Lock()
	c.orderbooks[assetID] = NewOrderbookState(assetID)
	c.orderbooksMu.Unlock()

	c.logger.Info("Subscribed to book", zap.String("assetID", assetID))
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
	var msg WSMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		c.logger.Error("Failed to parse message", zap.Error(err))
		return
	}

	switch msg.Channel {
	case ChannelBook:
		c.handleBookMessage(&msg)
	case ChannelTrades:
		c.handleTradesMessage(&msg)
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
	for {
		select {
		case <-c.ctx.Done():
			return
		case <-time.After(c.reconnectDelay):
		}

		c.logger.Info("Attempting to reconnect...")

		if err := c.Connect(); err != nil {
			c.logger.Error("Reconnect failed", zap.Error(err))
			continue
		}

		// Resubscribe
		c.resubscribe()

		if c.onReconnect != nil {
			c.onReconnect()
		}

		return
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
