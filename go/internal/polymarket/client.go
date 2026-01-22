package polymarket

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"go.uber.org/zap"
)

// Client is the Polymarket REST API client
type Client struct {
	endpoint   string
	apiKey     string
	apiSecret  string
	passphrase string
	httpClient *http.Client
	logger     *zap.Logger
}

// ClientConfig configures the REST client
type ClientConfig struct {
	Endpoint   string
	APIKey     string
	APISecret  string
	Passphrase string
	Logger     *zap.Logger
}

// NewClient creates a new REST API client
func NewClient(cfg ClientConfig) *Client {
	if cfg.Endpoint == "" {
		cfg.Endpoint = APIEndpointMainnet
	}
	if cfg.Logger == nil {
		cfg.Logger, _ = zap.NewProduction()
	}

	return &Client{
		endpoint:   cfg.Endpoint,
		apiKey:     cfg.APIKey,
		apiSecret:  cfg.APISecret,
		passphrase: cfg.Passphrase,
		httpClient: &http.Client{Timeout: 10 * time.Second},
		logger:     cfg.Logger,
	}
}

// Order represents an order to place
type Order struct {
	TokenID   string  `json:"tokenID"`
	Price     float64 `json:"price"`
	Size      float64 `json:"size"`
	Side      string  `json:"side"` // "BUY" or "SELL"
	OrderType string  `json:"type"` // "GTC", "GTD", "FOK"
	ExpireTime int64  `json:"expiration,omitempty"`
}

// OrderResponse is returned when placing an order
type OrderResponse struct {
	OrderID     string  `json:"orderID"`
	Status      string  `json:"status"`
	Price       string  `json:"price"`
	OriginalSize string `json:"originalSize"`
	RemainingSize string `json:"remainingSize"`
	Side        string  `json:"side"`
	Outcome     string  `json:"outcome"`
	TokenID     string  `json:"asset_id"`
	CreatedAt   int64   `json:"createdAt"`
}

// CancelResponse is returned when canceling an order
type CancelResponse struct {
	Canceled []string `json:"canceled"`
	NotCanceled []struct {
		OrderID string `json:"order_id"`
		Reason  string `json:"reason"`
	} `json:"notCanceled"`
}

// BalanceResponse contains account balances
type BalanceResponse struct {
	Collateral string `json:"collateral"`
}

// PositionResponse contains position information
type PositionResponse struct {
	AssetID  string `json:"asset_id"`
	Size     string `json:"size"`
	AvgPrice string `json:"avg_price"`
}

// sign creates the HMAC signature for authenticated requests
func (c *Client) sign(timestamp, method, path string, body []byte) string {
	message := timestamp + method + path
	if body != nil {
		message += string(body)
	}

	h := hmac.New(sha256.New, []byte(c.apiSecret))
	h.Write([]byte(message))
	return base64.StdEncoding.EncodeToString(h.Sum(nil))
}

// doRequest executes an HTTP request with authentication
func (c *Client) doRequest(ctx context.Context, method, path string, body any) ([]byte, error) {
	var bodyBytes []byte
	var err error
	if body != nil {
		bodyBytes, err = json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal body: %w", err)
		}
	}

	url := c.endpoint + path
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Add authentication headers
	timestamp := strconv.FormatInt(time.Now().Unix(), 10)
	signature := c.sign(timestamp, method, path, bodyBytes)

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("POLY_API_KEY", c.apiKey)
	req.Header.Set("POLY_SIGNATURE", signature)
	req.Header.Set("POLY_TIMESTAMP", timestamp)
	req.Header.Set("POLY_PASSPHRASE", c.passphrase)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

// GetMarket retrieves market information
func (c *Client) GetMarket(ctx context.Context, tokenID string) (*MarketInfo, error) {
	path := fmt.Sprintf("/markets/%s", tokenID)
	data, err := c.doRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, err
	}

	var market MarketInfo
	if err := json.Unmarshal(data, &market); err != nil {
		return nil, fmt.Errorf("failed to parse market: %w", err)
	}

	return &market, nil
}

// GetOrderbook retrieves the current orderbook
func (c *Client) GetOrderbook(ctx context.Context, tokenID string) (*BookSnapshot, error) {
	path := fmt.Sprintf("/book?token_id=%s", tokenID)
	data, err := c.doRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, err
	}

	var book BookSnapshot
	if err := json.Unmarshal(data, &book); err != nil {
		return nil, fmt.Errorf("failed to parse orderbook: %w", err)
	}

	return &book, nil
}

// PlaceOrder places a new order
func (c *Client) PlaceOrder(ctx context.Context, order Order) (*OrderResponse, error) {
	data, err := c.doRequest(ctx, "POST", "/order", order)
	if err != nil {
		return nil, err
	}

	var resp OrderResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse order response: %w", err)
	}

	c.logger.Info("Order placed",
		zap.String("orderID", resp.OrderID),
		zap.String("tokenID", order.TokenID),
		zap.String("side", order.Side),
		zap.Float64("price", order.Price),
		zap.Float64("size", order.Size),
	)

	return &resp, nil
}

// CancelOrder cancels an order by ID
func (c *Client) CancelOrder(ctx context.Context, orderID string) (*CancelResponse, error) {
	body := map[string]string{"orderID": orderID}
	data, err := c.doRequest(ctx, "DELETE", "/order", body)
	if err != nil {
		return nil, err
	}

	var resp CancelResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse cancel response: %w", err)
	}

	c.logger.Info("Order canceled", zap.String("orderID", orderID))
	return &resp, nil
}

// CancelAll cancels all orders for a market
func (c *Client) CancelAll(ctx context.Context, tokenID string) (*CancelResponse, error) {
	body := map[string]string{"asset_id": tokenID}
	data, err := c.doRequest(ctx, "DELETE", "/order-all", body)
	if err != nil {
		return nil, err
	}

	var resp CancelResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse cancel response: %w", err)
	}

	c.logger.Info("All orders canceled", zap.String("tokenID", tokenID))
	return &resp, nil
}

// GetOpenOrders retrieves all open orders
func (c *Client) GetOpenOrders(ctx context.Context) ([]OrderResponse, error) {
	data, err := c.doRequest(ctx, "GET", "/orders", nil)
	if err != nil {
		return nil, err
	}

	var orders []OrderResponse
	if err := json.Unmarshal(data, &orders); err != nil {
		return nil, fmt.Errorf("failed to parse orders: %w", err)
	}

	return orders, nil
}

// GetBalance retrieves account balance
func (c *Client) GetBalance(ctx context.Context) (*BalanceResponse, error) {
	data, err := c.doRequest(ctx, "GET", "/balance", nil)
	if err != nil {
		return nil, err
	}

	var balance BalanceResponse
	if err := json.Unmarshal(data, &balance); err != nil {
		return nil, fmt.Errorf("failed to parse balance: %w", err)
	}

	return &balance, nil
}

// GetPositions retrieves all positions
func (c *Client) GetPositions(ctx context.Context) ([]PositionResponse, error) {
	data, err := c.doRequest(ctx, "GET", "/positions", nil)
	if err != nil {
		return nil, err
	}

	var positions []PositionResponse
	if err := json.Unmarshal(data, &positions); err != nil {
		return nil, fmt.Errorf("failed to parse positions: %w", err)
	}

	return positions, nil
}
