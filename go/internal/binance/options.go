package binance

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"

	"go.uber.org/zap"
)

// Options API endpoints
const (
	OptionsAPIEndpoint = "https://eapi.binance.com"
)

// OptionsClient provides access to Binance options data
type OptionsClient struct {
	endpoint   string
	httpClient *http.Client
	logger     *zap.Logger

	// Cache
	optionData map[string]*OptionData
	dataMu     sync.RWMutex

	// Polling
	pollInterval time.Duration
	ctx          context.Context
	cancel       context.CancelFunc
	wg           sync.WaitGroup

	// Callbacks
	onUpdate func(symbol string, data *OptionData)
}

// OptionData contains option market data
type OptionData struct {
	Symbol          string
	Underlying      string
	ExpiryDate      time.Time
	StrikePrice     float64
	OptionType      string // "CALL" or "PUT"
	MarkPrice       float64
	BidPrice        float64
	AskPrice        float64
	Delta           float64
	Gamma           float64
	Theta           float64
	Vega            float64
	ImpliedVol      float64
	OpenInterest    float64
	LastUpdateTime  time.Time
}

// OptionTickerResponse is the API response for option ticker
type OptionTickerResponse struct {
	Symbol          string `json:"symbol"`
	PriceChange     string `json:"priceChange"`
	PriceChangePercent string `json:"priceChangePercent"`
	MarkPrice       string `json:"markPrice"`
	BidPrice        string `json:"bidPrice"`
	AskPrice        string `json:"askPrice"`
	High            string `json:"high"`
	Low             string `json:"low"`
	Volume          string `json:"volume"`
	Amount          string `json:"amount"`
	BidIV           string `json:"bidIV"`
	AskIV           string `json:"askIV"`
	MarkIV          string `json:"markIV"`
	Delta           string `json:"delta"`
	Gamma           string `json:"gamma"`
	Theta           string `json:"theta"`
	Vega            string `json:"vega"`
	OpenInterest    string `json:"openInterest"`
	Exercised       string `json:"exercised"`
	Time            int64  `json:"time"`
}

// OptionsClientConfig configures the options client
type OptionsClientConfig struct {
	Endpoint     string
	PollInterval time.Duration
	Logger       *zap.Logger
}

// NewOptionsClient creates a new options client
func NewOptionsClient(cfg OptionsClientConfig) *OptionsClient {
	if cfg.Endpoint == "" {
		cfg.Endpoint = OptionsAPIEndpoint
	}
	if cfg.Logger == nil {
		cfg.Logger, _ = zap.NewProduction()
	}
	if cfg.PollInterval == 0 {
		cfg.PollInterval = 1 * time.Second
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &OptionsClient{
		endpoint:     cfg.Endpoint,
		httpClient:   &http.Client{Timeout: 10 * time.Second},
		logger:       cfg.Logger,
		optionData:   make(map[string]*OptionData),
		pollInterval: cfg.PollInterval,
		ctx:          ctx,
		cancel:       cancel,
	}
}

// OnUpdate sets the callback for option data updates
func (c *OptionsClient) OnUpdate(fn func(symbol string, data *OptionData)) {
	c.onUpdate = fn
}

// GetOptionData returns cached option data for a symbol
func (c *OptionsClient) GetOptionData(symbol string) (*OptionData, bool) {
	c.dataMu.RLock()
	defer c.dataMu.RUnlock()
	data, ok := c.optionData[symbol]
	return data, ok
}

// GetAllOptionData returns all cached option data
func (c *OptionsClient) GetAllOptionData() map[string]*OptionData {
	c.dataMu.RLock()
	defer c.dataMu.RUnlock()
	result := make(map[string]*OptionData, len(c.optionData))
	for k, v := range c.optionData {
		result[k] = v
	}
	return result
}

// FetchTicker fetches current ticker data for an option
func (c *OptionsClient) FetchTicker(ctx context.Context, symbol string) (*OptionData, error) {
	url := fmt.Sprintf("%s/eapi/v1/ticker?symbol=%s", c.endpoint, symbol)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error: status %d", resp.StatusCode)
	}

	var ticker OptionTickerResponse
	if err := json.NewDecoder(resp.Body).Decode(&ticker); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return c.parseTickerResponse(&ticker), nil
}

// FetchAllTickers fetches all option tickers for an underlying
func (c *OptionsClient) FetchAllTickers(ctx context.Context, underlying string) ([]*OptionData, error) {
	url := fmt.Sprintf("%s/eapi/v1/ticker?underlying=%s", c.endpoint, underlying)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error: status %d", resp.StatusCode)
	}

	var tickers []OptionTickerResponse
	if err := json.NewDecoder(resp.Body).Decode(&tickers); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	result := make([]*OptionData, len(tickers))
	for i, ticker := range tickers {
		result[i] = c.parseTickerResponse(&ticker)
	}

	return result, nil
}

// StartPolling starts polling for option data
func (c *OptionsClient) StartPolling(underlyings []string) {
	c.wg.Add(1)
	go c.pollLoop(underlyings)
}

// Stop stops the polling loop
func (c *OptionsClient) Stop() {
	c.cancel()
	c.wg.Wait()
}

func (c *OptionsClient) pollLoop(underlyings []string) {
	defer c.wg.Done()

	ticker := time.NewTicker(c.pollInterval)
	defer ticker.Stop()

	// Initial fetch
	c.fetchAll(underlyings)

	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			c.fetchAll(underlyings)
		}
	}
}

func (c *OptionsClient) fetchAll(underlyings []string) {
	for _, underlying := range underlyings {
		options, err := c.FetchAllTickers(c.ctx, underlying)
		if err != nil {
			c.logger.Error("Failed to fetch options",
				zap.String("underlying", underlying),
				zap.Error(err))
			continue
		}

		c.dataMu.Lock()
		for _, opt := range options {
			c.optionData[opt.Symbol] = opt
		}
		c.dataMu.Unlock()

		if c.onUpdate != nil {
			for _, opt := range options {
				c.onUpdate(opt.Symbol, opt)
			}
		}
	}
}

func (c *OptionsClient) parseTickerResponse(ticker *OptionTickerResponse) *OptionData {
	markPrice, _ := strconv.ParseFloat(ticker.MarkPrice, 64)
	bidPrice, _ := strconv.ParseFloat(ticker.BidPrice, 64)
	askPrice, _ := strconv.ParseFloat(ticker.AskPrice, 64)
	delta, _ := strconv.ParseFloat(ticker.Delta, 64)
	gamma, _ := strconv.ParseFloat(ticker.Gamma, 64)
	theta, _ := strconv.ParseFloat(ticker.Theta, 64)
	vega, _ := strconv.ParseFloat(ticker.Vega, 64)
	markIV, _ := strconv.ParseFloat(ticker.MarkIV, 64)
	openInterest, _ := strconv.ParseFloat(ticker.OpenInterest, 64)

	return &OptionData{
		Symbol:         ticker.Symbol,
		MarkPrice:      markPrice,
		BidPrice:       bidPrice,
		AskPrice:       askPrice,
		Delta:          delta,
		Gamma:          gamma,
		Theta:          theta,
		Vega:           vega,
		ImpliedVol:     markIV,
		OpenInterest:   openInterest,
		LastUpdateTime: time.UnixMilli(ticker.Time),
	}
}
