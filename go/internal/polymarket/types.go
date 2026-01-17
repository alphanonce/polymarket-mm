// Package polymarket provides a client for the Polymarket CLOB API
package polymarket

import (
	"time"
)

// WebSocket endpoints
const (
	WSEndpointMainnet = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
	WSEndpointTestnet = "wss://ws-subscriptions-clob-staging.polymarket.com/ws/market"

	// REST endpoints
	APIEndpointMainnet = "https://clob.polymarket.com"
	APIEndpointTestnet = "https://clob-staging.polymarket.com"
)

// Message types
const (
	MsgTypeSubscribe   = "subscribe"
	MsgTypeUnsubscribe = "unsubscribe"
)

// Channel types for subscription
const (
	ChannelBook   = "book"
	ChannelTrades = "trades"
	ChannelUser   = "user"
)

// SubscribeMessage is sent to subscribe to channels
type SubscribeMessage struct {
	Type     string   `json:"type"`
	Channel  string   `json:"channel"`
	Markets  []string `json:"markets,omitempty"`
	AssetIDs []string `json:"assets_ids,omitempty"`
}

// BookLevel represents a price level in the orderbook
type BookLevel struct {
	Price string `json:"price"`
	Size  string `json:"size"`
}

// BookSnapshot is the initial orderbook snapshot
type BookSnapshot struct {
	AssetID   string      `json:"asset_id"`
	Hash      string      `json:"hash"`
	Timestamp int64       `json:"timestamp"`
	Bids      []BookLevel `json:"bids"`
	Asks      []BookLevel `json:"asks"`
}

// BookDelta represents an orderbook update
type BookDelta struct {
	AssetID   string      `json:"asset_id"`
	Hash      string      `json:"hash"`
	Timestamp int64       `json:"timestamp"`
	Bids      []BookLevel `json:"bids"`
	Asks      []BookLevel `json:"asks"`
}

// Trade represents a trade event
type Trade struct {
	AssetID     string `json:"asset_id"`
	ID          string `json:"id"`
	Price       string `json:"price"`
	Size        string `json:"size"`
	Side        string `json:"side"` // "BUY" or "SELL"
	Timestamp   int64  `json:"timestamp"`
	TradeOwner  string `json:"trade_owner"`
	Maker       string `json:"maker"`
	Taker       string `json:"taker"`
	MakerOrderID string `json:"maker_order_id"`
	TakerOrderID string `json:"taker_order_id"`
}

// WSMessage is the generic WebSocket message wrapper
type WSMessage struct {
	Type     string `json:"type"`
	Channel  string `json:"channel"`
	AssetID  string `json:"asset_id,omitempty"`

	// For book channel
	Book     *BookSnapshot `json:"book,omitempty"`
	Delta    *BookDelta    `json:"delta,omitempty"`

	// For trades channel
	Trades   []Trade       `json:"trades,omitempty"`
}

// MarketInfo contains market metadata
type MarketInfo struct {
	ConditionID     string    `json:"condition_id"`
	QuestionID      string    `json:"question_id"`
	TokenID         string    `json:"token_id"`
	Outcome         string    `json:"outcome"`        // "Yes" or "No"
	Question        string    `json:"question"`
	EndDate         time.Time `json:"end_date_iso"`
	Active          bool      `json:"active"`
	Closed          bool      `json:"closed"`
	MinTickSize     string    `json:"minimum_tick_size"`
	MinOrderSize    string    `json:"minimum_order_size"`
}

// OrderbookState maintains the local orderbook state
type OrderbookState struct {
	AssetID        string
	Bids           map[string]string // price -> size
	Asks           map[string]string // price -> size
	LastHash       string
	LastTimestamp  int64
	LastUpdateTime time.Time
}

// NewOrderbookState creates a new orderbook state
func NewOrderbookState(assetID string) *OrderbookState {
	return &OrderbookState{
		AssetID: assetID,
		Bids:    make(map[string]string),
		Asks:    make(map[string]string),
	}
}

// ApplySnapshot applies a full snapshot to the orderbook
func (ob *OrderbookState) ApplySnapshot(snap *BookSnapshot) {
	ob.Bids = make(map[string]string)
	ob.Asks = make(map[string]string)

	for _, level := range snap.Bids {
		ob.Bids[level.Price] = level.Size
	}
	for _, level := range snap.Asks {
		ob.Asks[level.Price] = level.Size
	}

	ob.LastHash = snap.Hash
	ob.LastTimestamp = snap.Timestamp
	ob.LastUpdateTime = time.Now()
}

// ApplyDelta applies an incremental update to the orderbook
func (ob *OrderbookState) ApplyDelta(delta *BookDelta) {
	for _, level := range delta.Bids {
		if level.Size == "0" {
			delete(ob.Bids, level.Price)
		} else {
			ob.Bids[level.Price] = level.Size
		}
	}
	for _, level := range delta.Asks {
		if level.Size == "0" {
			delete(ob.Asks, level.Price)
		} else {
			ob.Asks[level.Price] = level.Size
		}
	}

	ob.LastHash = delta.Hash
	ob.LastTimestamp = delta.Timestamp
	ob.LastUpdateTime = time.Now()
}
