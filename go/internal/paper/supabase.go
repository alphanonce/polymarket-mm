package paper

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"go.uber.org/zap"
)

// SupabaseClient is an HTTP client for Supabase REST API
type SupabaseClient struct {
	baseURL    string
	apiKey     string
	client     *http.Client
	logger     *zap.Logger
	maxRetries int
}

// NewSupabaseClient creates a new Supabase client
func NewSupabaseClient(baseURL, apiKey string, logger *zap.Logger) *SupabaseClient {
	return &SupabaseClient{
		baseURL: baseURL,
		apiKey:  apiKey,
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
		logger:     logger,
		maxRetries: 3,
	}
}

// request makes an HTTP request to Supabase
func (c *SupabaseClient) request(method, path string, body any, headers map[string]string) ([]byte, error) {
	var bodyReader io.Reader
	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("marshal body: %w", err)
		}
		bodyReader = bytes.NewReader(jsonBody)
	}

	url := c.baseURL + path
	req, err := http.NewRequest(method, url, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	// Set headers
	req.Header.Set("apikey", c.apiKey)
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Prefer", "return=minimal")

	for k, v := range headers {
		req.Header.Set(k, v)
	}

	// Retry logic
	var lastErr error
	for i := 0; i < c.maxRetries; i++ {
		resp, err := c.client.Do(req)
		if err != nil {
			lastErr = err
			time.Sleep(time.Duration(i+1) * 100 * time.Millisecond)
			continue
		}
		defer resp.Body.Close()

		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			lastErr = fmt.Errorf("read response: %w", err)
			continue
		}

		if resp.StatusCode >= 400 {
			lastErr = fmt.Errorf("supabase error %d: %s", resp.StatusCode, string(respBody))
			if resp.StatusCode >= 500 {
				time.Sleep(time.Duration(i+1) * 100 * time.Millisecond)
				continue
			}
			return nil, lastErr
		}

		return respBody, nil
	}

	return nil, fmt.Errorf("max retries exceeded: %w", lastErr)
}

// post makes a POST request
func (c *SupabaseClient) post(path string, body any) error {
	_, err := c.request("POST", path, body, nil)
	return err
}

// postUpsert makes a POST request with upsert semantics
func (c *SupabaseClient) postUpsert(path string, body any, conflictColumn string) error {
	headers := map[string]string{
		"Prefer": "resolution=merge-duplicates",
	}
	fullPath := path + "?on_conflict=" + conflictColumn
	_, err := c.request("POST", fullPath, body, headers)
	return err
}


// InsertTrade inserts a new trade record
func (c *SupabaseClient) InsertTrade(trade Trade) error {
	if trade.Timestamp.IsZero() {
		trade.Timestamp = time.Now()
	}
	err := c.post("/rest/v1/trades", trade)
	if err != nil {
		c.logger.Error("Failed to insert trade", zap.Error(err), zap.String("asset_id", trade.AssetID))
		return err
	}
	c.logger.Debug("Inserted trade", zap.String("asset_id", trade.AssetID), zap.Float64("price", trade.Price))
	return nil
}

// InsertTrades inserts multiple trade records in a batch
func (c *SupabaseClient) InsertTrades(trades []Trade) error {
	if len(trades) == 0 {
		return nil
	}
	err := c.post("/rest/v1/trades", trades)
	if err != nil {
		c.logger.Error("Failed to insert trades batch", zap.Error(err), zap.Int("count", len(trades)))
		return err
	}
	c.logger.Debug("Inserted trades batch", zap.Int("count", len(trades)))
	return nil
}

// UpsertPosition inserts or updates a position
func (c *SupabaseClient) UpsertPosition(pos Position) error {
	if pos.UpdatedAt.IsZero() {
		pos.UpdatedAt = time.Now()
	}
	err := c.postUpsert("/rest/v1/positions", pos, "asset_id")
	if err != nil {
		c.logger.Error("Failed to upsert position", zap.Error(err), zap.String("asset_id", pos.AssetID))
		return err
	}
	c.logger.Debug("Upserted position", zap.String("asset_id", pos.AssetID), zap.Float64("size", pos.Size))
	return nil
}

// UpsertPositions inserts or updates multiple positions
func (c *SupabaseClient) UpsertPositions(positions []Position) error {
	if len(positions) == 0 {
		return nil
	}
	for i := range positions {
		if positions[i].UpdatedAt.IsZero() {
			positions[i].UpdatedAt = time.Now()
		}
	}
	err := c.postUpsert("/rest/v1/positions", positions, "asset_id")
	if err != nil {
		c.logger.Error("Failed to upsert positions batch", zap.Error(err), zap.Int("count", len(positions)))
		return err
	}
	c.logger.Debug("Upserted positions batch", zap.Int("count", len(positions)))
	return nil
}

// InsertEquitySnapshot inserts an equity snapshot
func (c *SupabaseClient) InsertEquitySnapshot(snap EquitySnapshot) error {
	if snap.Timestamp.IsZero() {
		snap.Timestamp = time.Now()
	}
	err := c.post("/rest/v1/equity_snapshots", snap)
	if err != nil {
		c.logger.Error("Failed to insert equity snapshot", zap.Error(err))
		return err
	}
	c.logger.Debug("Inserted equity snapshot", zap.Float64("equity", snap.Equity))
	return nil
}

// UpsertMetrics updates the metrics singleton
func (c *SupabaseClient) UpsertMetrics(metrics Metrics) error {
	metrics.ID = 1 // Singleton
	if metrics.UpdatedAt.IsZero() {
		metrics.UpdatedAt = time.Now()
	}
	err := c.postUpsert("/rest/v1/metrics", metrics, "id")
	if err != nil {
		c.logger.Error("Failed to upsert metrics", zap.Error(err))
		return err
	}
	c.logger.Debug("Upserted metrics",
		zap.Float64("total_pnl", metrics.TotalPnL),
		zap.Int("total_trades", metrics.TotalTrades))
	return nil
}

// UpsertMarket inserts or updates a market record
func (c *SupabaseClient) UpsertMarket(market Market) error {
	if market.CreatedAt.IsZero() {
		market.CreatedAt = time.Now()
	}
	err := c.postUpsert("/rest/v1/markets", market, "slug")
	if err != nil {
		c.logger.Error("Failed to upsert market", zap.Error(err), zap.String("slug", market.Slug))
		return err
	}
	c.logger.Debug("Upserted market", zap.String("slug", market.Slug), zap.String("status", market.Status))
	return nil
}

// InsertMarketSnapshot inserts a market snapshot
func (c *SupabaseClient) InsertMarketSnapshot(snap MarketSnapshot) error {
	if snap.Timestamp.IsZero() {
		snap.Timestamp = time.Now()
	}
	err := c.post("/rest/v1/market_snapshots", snap)
	if err != nil {
		c.logger.Error("Failed to insert market snapshot", zap.Error(err), zap.String("slug", snap.Slug))
		return err
	}
	c.logger.Debug("Inserted market snapshot", zap.String("slug", snap.Slug))
	return nil
}

// InsertMarketSnapshots inserts multiple market snapshots in a batch
func (c *SupabaseClient) InsertMarketSnapshots(snapshots []MarketSnapshot) error {
	if len(snapshots) == 0 {
		return nil
	}
	now := time.Now()
	for i := range snapshots {
		if snapshots[i].Timestamp.IsZero() {
			snapshots[i].Timestamp = now
		}
	}
	err := c.post("/rest/v1/market_snapshots", snapshots)
	if err != nil {
		c.logger.Error("Failed to insert market snapshots batch", zap.Error(err), zap.Int("count", len(snapshots)))
		return err
	}
	c.logger.Debug("Inserted market snapshots batch", zap.Int("count", len(snapshots)))
	return nil
}

// HealthCheck tests the connection to Supabase
func (c *SupabaseClient) HealthCheck() error {
	_, err := c.request("GET", "/rest/v1/metrics?select=id&limit=1", nil, nil)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	return nil
}
