package polymarket

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"sync"
	"time"

	"go.uber.org/zap"
)

// DiscoveredMarket contains information about a discovered market
type DiscoveredMarket struct {
	Slug        string
	ConditionID string
	TokenIDUp   string
	TokenIDDown string
	Asset       string
	Timeframe   string
	StartTS     int64 // Unix timestamp in seconds
	EndTS       int64 // Unix timestamp in seconds
}

// TimeToExpiry returns the time until expiry in seconds
func (m *DiscoveredMarket) TimeToExpiry() float64 {
	now := time.Now().Unix()
	return float64(m.EndTS - now)
}

// IsExpired checks if the market has expired
func (m *DiscoveredMarket) IsExpired() bool {
	return m.TimeToExpiry() <= 0
}

// DiscoveryConfig configures the market discovery service
type DiscoveryConfig struct {
	Logger              *zap.Logger
	PollInterval        time.Duration
	SupportedTimeframes []string
	SupportedAssets     []string
	GammaAPIURL         string
}

// DiscoveryService discovers updown markets from Polymarket
type DiscoveryService struct {
	cfg    DiscoveryConfig
	logger *zap.Logger

	// HTTP client
	client *http.Client

	// Known markets
	markets   map[string]*DiscoveredMarket
	marketsMu sync.RWMutex

	// Callbacks
	onNewMarket    func(*DiscoveredMarket)
	onBatchDiscovered func([]*DiscoveredMarket) // Called after each discovery round with newly found markets

	// Control
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	// Slug pattern: btc-updown-15m-1234567890
	slugPattern *regexp.Regexp
}

// TimeframeSeconds maps timeframe strings to seconds
var TimeframeSeconds = map[string]int64{
	"15m": 900,
	"1h":  3600,
	"4h":  14400,
}

// NewDiscoveryService creates a new market discovery service
func NewDiscoveryService(cfg DiscoveryConfig) *DiscoveryService {
	if cfg.Logger == nil {
		cfg.Logger, _ = zap.NewProduction()
	}
	if cfg.PollInterval == 0 {
		cfg.PollInterval = 30 * time.Second
	}
	if len(cfg.SupportedTimeframes) == 0 {
		cfg.SupportedTimeframes = []string{"15m", "1h", "4h"}
	}
	if len(cfg.SupportedAssets) == 0 {
		cfg.SupportedAssets = []string{"btc", "eth", "sol", "xrp"}
	}
	if cfg.GammaAPIURL == "" {
		cfg.GammaAPIURL = "https://gamma-api.polymarket.com"
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &DiscoveryService{
		cfg:         cfg,
		logger:      cfg.Logger,
		client:      &http.Client{Timeout: 30 * time.Second},
		markets:     make(map[string]*DiscoveredMarket),
		ctx:         ctx,
		cancel:      cancel,
		slugPattern: regexp.MustCompile(`^([a-z]+)-updown-(\d+[mh])-(\d+)$`),
	}
}

// OnNewMarket sets the callback for newly discovered markets
func (d *DiscoveryService) OnNewMarket(fn func(*DiscoveredMarket)) {
	d.onNewMarket = fn
}

// OnBatchDiscovered sets the callback that's invoked after each discovery round
// with all newly discovered markets. This is useful for batch subscription.
func (d *DiscoveryService) OnBatchDiscovered(fn func([]*DiscoveredMarket)) {
	d.onBatchDiscovered = fn
}

// Start begins the discovery loop
func (d *DiscoveryService) Start() {
	d.wg.Add(1)
	go d.discoveryLoop()
	d.logger.Info("Market discovery started",
		zap.Strings("assets", d.cfg.SupportedAssets),
		zap.Strings("timeframes", d.cfg.SupportedTimeframes),
	)
}

// Stop stops the discovery service
func (d *DiscoveryService) Stop() {
	d.cancel()
	d.wg.Wait()
	d.logger.Info("Market discovery stopped")
}

// GetActiveMarkets returns all currently active markets
func (d *DiscoveryService) GetActiveMarkets() []*DiscoveredMarket {
	d.marketsMu.RLock()
	defer d.marketsMu.RUnlock()

	var active []*DiscoveredMarket
	for _, m := range d.markets {
		if !m.IsExpired() {
			active = append(active, m)
		}
	}
	return active
}

// GetMarket returns a specific market by slug
func (d *DiscoveryService) GetMarket(slug string) (*DiscoveredMarket, bool) {
	d.marketsMu.RLock()
	defer d.marketsMu.RUnlock()
	m, ok := d.markets[slug]
	return m, ok
}

func (d *DiscoveryService) discoveryLoop() {
	defer d.wg.Done()

	// Initial discovery
	d.discoverMarkets()

	ticker := time.NewTicker(d.cfg.PollInterval)
	defer ticker.Stop()

	// Cleanup ticker for expired markets (every 5 minutes)
	cleanupTicker := time.NewTicker(5 * time.Minute)
	defer cleanupTicker.Stop()

	for {
		select {
		case <-d.ctx.Done():
			return
		case <-ticker.C:
			d.discoverMarkets()
		case <-cleanupTicker.C:
			d.cleanupExpiredMarkets()
		}
	}
}

// cleanupExpiredMarkets removes markets that have been expired for more than 5 minutes
func (d *DiscoveryService) cleanupExpiredMarkets() {
	d.marketsMu.Lock()
	defer d.marketsMu.Unlock()

	now := time.Now().Unix()
	expiredGracePeriod := int64(5 * 60) // 5 minutes after expiry

	var toDelete []string
	for slug, market := range d.markets {
		// Remove if expired for more than grace period
		if now > market.EndTS+expiredGracePeriod {
			toDelete = append(toDelete, slug)
		}
	}

	for _, slug := range toDelete {
		delete(d.markets, slug)
	}

	if len(toDelete) > 0 {
		d.logger.Info("Cleaned up expired markets", zap.Int("removed", len(toDelete)))
	}
}

func (d *DiscoveryService) discoverMarkets() {
	now := time.Now().Unix()
	var newMarkets []*DiscoveredMarket

	for _, asset := range d.cfg.SupportedAssets {
		for _, timeframe := range d.cfg.SupportedTimeframes {
			intervalSec, ok := TimeframeSeconds[timeframe]
			if !ok {
				continue
			}

			// Calculate current and next market timestamps
			currentTS := (now / intervalSec) * intervalSec
			nextTS := currentTS + intervalSec

			// Try both current and next slots
			for _, marketTS := range []int64{currentTS, nextTS} {
				slug := fmt.Sprintf("%s-updown-%s-%d", asset, timeframe, marketTS)

				// Skip if already known
				d.marketsMu.RLock()
				_, known := d.markets[slug]
				d.marketsMu.RUnlock()
				if known {
					continue
				}

				// Fetch market from API
				market, err := d.fetchMarketBySlug(slug)
				if err != nil {
					d.logger.Debug("Failed to fetch market",
						zap.String("slug", slug),
						zap.Error(err),
					)
					continue
				}

				if market != nil {
					d.marketsMu.Lock()
					d.markets[slug] = market
					d.marketsMu.Unlock()

					d.logger.Info("New market discovered",
						zap.String("slug", slug),
						zap.String("asset", market.Asset),
						zap.String("timeframe", market.Timeframe),
						zap.Float64("expires_in_s", market.TimeToExpiry()),
					)

					newMarkets = append(newMarkets, market)

					if d.onNewMarket != nil {
						d.onNewMarket(market)
					}
				}
			}
		}
	}

	// Call batch callback if there are new markets
	if len(newMarkets) > 0 && d.onBatchDiscovered != nil {
		d.onBatchDiscovered(newMarkets)
	}
}

// GammaMarketResponse represents the gamma API response for a market
type GammaMarketResponse struct {
	Slug         string `json:"slug"`
	ConditionID  string `json:"conditionId"`
	Outcomes     string `json:"outcomes"`
	ClobTokenIDs string `json:"clobTokenIds"`
	EndDate      string `json:"endDate"`
	Active       bool   `json:"active"`
	Closed       bool   `json:"closed"`
}

func (d *DiscoveryService) fetchMarketBySlug(slug string) (*DiscoveredMarket, error) {
	url := fmt.Sprintf("%s/markets?slug=%s", d.cfg.GammaAPIURL, slug)

	req, err := http.NewRequestWithContext(d.ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := d.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API returned status %d", resp.StatusCode)
	}

	var markets []GammaMarketResponse
	if err := json.NewDecoder(resp.Body).Decode(&markets); err != nil {
		return nil, err
	}

	if len(markets) == 0 {
		return nil, nil
	}

	return d.processMarket(&markets[0])
}

func (d *DiscoveryService) processMarket(data *GammaMarketResponse) (*DiscoveredMarket, error) {
	// Parse slug to extract asset, timeframe, and timestamp
	matches := d.slugPattern.FindStringSubmatch(data.Slug)
	if matches == nil {
		return nil, fmt.Errorf("invalid slug pattern: %s", data.Slug)
	}

	asset := matches[1]
	timeframe := matches[2]
	var marketTS int64
	fmt.Sscanf(matches[3], "%d", &marketTS)

	// Check if timeframe is supported
	intervalSec, ok := TimeframeSeconds[timeframe]
	if !ok {
		return nil, fmt.Errorf("unsupported timeframe: %s", timeframe)
	}

	// Parse token IDs from JSON strings
	var outcomes []string
	var tokenIDs []string

	if err := json.Unmarshal([]byte(data.Outcomes), &outcomes); err != nil {
		return nil, fmt.Errorf("failed to parse outcomes: %w", err)
	}
	if err := json.Unmarshal([]byte(data.ClobTokenIDs), &tokenIDs); err != nil {
		return nil, fmt.Errorf("failed to parse token IDs: %w", err)
	}

	var tokenIDUp, tokenIDDown string
	for i, outcome := range outcomes {
		if i >= len(tokenIDs) {
			break
		}
		switch outcome {
		case "Up":
			tokenIDUp = tokenIDs[i]
		case "Down":
			tokenIDDown = tokenIDs[i]
		}
	}

	if tokenIDUp == "" || tokenIDDown == "" {
		return nil, fmt.Errorf("missing token IDs")
	}

	return &DiscoveredMarket{
		Slug:        data.Slug,
		ConditionID: data.ConditionID,
		TokenIDUp:   tokenIDUp,
		TokenIDDown: tokenIDDown,
		Asset:       asset,
		Timeframe:   timeframe,
		StartTS:     marketTS,
		EndTS:       marketTS + intervalSec,
	}, nil
}
