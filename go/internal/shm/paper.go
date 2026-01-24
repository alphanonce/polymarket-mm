package shm

import (
	"fmt"
	"os"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	"github.com/alphanonce/polymarket-mm/internal/paper"
)

// PaperReader provides read access to paper trading shared memory
type PaperReader struct {
	file   *os.File
	data   []byte
	layout *paper.PaperTradingState
}

// NewPaperReader creates a new paper trading SHM reader
func NewPaperReader() (*PaperReader, error) {
	// Try /dev/shm first, fall back to /tmp
	shmPath := "/dev/shm" + paper.PaperSHMName
	file, err := os.OpenFile(shmPath, os.O_RDONLY, 0666)
	if err != nil {
		shmPath = "/tmp" + paper.PaperSHMName
		file, err = os.OpenFile(shmPath, os.O_RDONLY, 0666)
		if err != nil {
			return nil, fmt.Errorf("failed to open paper SHM: %w", err)
		}
	}

	// Get file size
	info, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to stat paper SHM: %w", err)
	}

	size := int(info.Size())
	if size < int(unsafe.Sizeof(paper.PaperTradingState{})) {
		file.Close()
		return nil, fmt.Errorf("paper SHM file too small: %d", size)
	}

	// Memory map the file (read-only)
	data, err := syscall.Mmap(
		int(file.Fd()),
		0,
		size,
		syscall.PROT_READ,
		syscall.MAP_SHARED,
	)
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to mmap paper SHM: %w", err)
	}

	layout := (*paper.PaperTradingState)(unsafe.Pointer(&data[0]))

	// Validate magic number
	if layout.Magic != paper.PaperSHMMagic {
		syscall.Munmap(data)
		file.Close()
		return nil, fmt.Errorf("invalid paper SHM magic: 0x%X", layout.Magic)
	}

	return &PaperReader{
		file:   file,
		data:   data,
		layout: layout,
	}, nil
}

// Close unmaps and closes the shared memory
func (r *PaperReader) Close() error {
	if err := syscall.Munmap(r.data); err != nil {
		return fmt.Errorf("failed to unmap paper SHM: %w", err)
	}
	return r.file.Close()
}

// Layout returns the raw layout pointer
func (r *PaperReader) Layout() *paper.PaperTradingState {
	return r.layout
}

// GetStateSequence returns the current state sequence number
func (r *PaperReader) GetStateSequence() uint32 {
	return atomic.LoadUint32(&r.layout.StateSequence)
}

// GetPositions returns all positions
func (r *PaperReader) GetPositions() []paper.Position {
	numPositions := atomic.LoadUint32(&r.layout.NumPositions)
	positions := make([]paper.Position, 0, numPositions)

	for i := uint32(0); i < numPositions; i++ {
		p := r.layout.Positions[i]
		positions = append(positions, paper.Position{
			AssetID:       bytesToString(p.AssetID[:]),
			Slug:          bytesToString(p.Slug[:]),
			Side:          bytesToString(p.Side[:]),
			Size:          p.Size,
			AvgEntryPrice: p.AvgEntryPrice,
			UnrealizedPnL: p.UnrealizedPnL,
			RealizedPnL:   p.RealizedPnL,
			UpdatedAt:     time.Unix(0, int64(p.UpdatedAtNs)),
		})
	}

	return positions
}

// GetQuotes returns all current quotes
func (r *PaperReader) GetQuotes() []paper.MarketSnapshot {
	numQuotes := atomic.LoadUint32(&r.layout.NumQuotes)
	quotes := make([]paper.MarketSnapshot, 0, numQuotes)

	for i := uint32(0); i < numQuotes; i++ {
		q := r.layout.Quotes[i]
		quotes = append(quotes, paper.MarketSnapshot{
			Slug:      bytesToString(q.Slug[:]),
			BestBid:   q.BestBid,
			BestAsk:   q.BestAsk,
			MidPrice:  q.MidPrice,
			Spread:    q.Spread,
			OurBid:    q.OurBid,
			OurAsk:    q.OurAsk,
			Inventory: q.Inventory,
			Timestamp: time.Unix(0, int64(q.UpdatedNs)),
		})
	}

	return quotes
}

// GetEquity returns the current equity state
func (r *PaperReader) GetEquity() paper.EquitySnapshot {
	return paper.EquitySnapshot{
		Equity:        r.layout.TotalEquity,
		Cash:          r.layout.Cash,
		PositionValue: r.layout.PositionValue,
		Timestamp:     time.Now(),
	}
}

// GetMetrics returns the current metrics
func (r *PaperReader) GetMetrics() paper.Metrics {
	winRate := float64(0)
	if r.layout.TotalTrades > 0 {
		winRate = float64(r.layout.WinCount) / float64(r.layout.TotalTrades)
	}

	return paper.Metrics{
		ID:            1,
		TotalPnL:      r.layout.TotalPnL,
		RealizedPnL:   r.layout.RealizedPnL,
		UnrealizedPnL: r.layout.UnrealizedPnL,
		TotalTrades:   int(r.layout.TotalTrades),
		WinRate:       winRate,
		SharpeRatio:   r.layout.SharpeRatio,
		MaxDrawdown:   r.layout.MaxDrawdown,
	}
}

// GetPendingTrades returns pending trades that need to be persisted
func (r *PaperReader) GetPendingTrades() []paper.Trade {
	numTrades := atomic.LoadUint32(&r.layout.NumTrades)
	head := atomic.LoadUint32(&r.layout.TradesHead)
	tail := atomic.LoadUint32(&r.layout.TradesTail)

	if head >= tail || numTrades == 0 {
		return nil
	}

	trades := make([]paper.Trade, 0, tail-head)
	for i := head; i < tail && i < paper.MaxPaperTrades; i++ {
		t := r.layout.Trades[i%paper.MaxPaperTrades]
		if t.Persisted == 0 {
			trades = append(trades, paper.Trade{
				AssetID:   bytesToString(t.AssetID[:]),
				Slug:      bytesToString(t.Slug[:]),
				Side:      int(t.Side),
				Price:     t.Price,
				Size:      t.Size,
				PnL:       t.PnL,
				Timestamp: time.Unix(0, int64(t.TimestampNs)),
			})
		}
	}

	return trades
}

// HasStateChanged checks if state has changed since last sequence
func (r *PaperReader) HasStateChanged(lastSeq uint32) bool {
	return atomic.LoadUint32(&r.layout.StateSequence) != lastSeq
}

// bytesToString converts a null-terminated byte array to string
func bytesToString(b []byte) string {
	for i, v := range b {
		if v == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}

// PaperWriter provides write access to paper trading shared memory (for marking trades persisted)
type PaperWriter struct {
	file   *os.File
	data   []byte
	layout *paper.PaperTradingState
}

// NewPaperWriter creates a new paper trading SHM writer
func NewPaperWriter() (*PaperWriter, error) {
	// Create or open the shared memory file
	shmPath := "/dev/shm" + paper.PaperSHMName
	file, err := os.OpenFile(shmPath, os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		shmPath = "/tmp" + paper.PaperSHMName
		file, err = os.OpenFile(shmPath, os.O_RDWR|os.O_CREATE, 0666)
		if err != nil {
			return nil, fmt.Errorf("failed to create paper SHM: %w", err)
		}
	}

	// Set the file size
	size := int(unsafe.Sizeof(paper.PaperTradingState{}))
	if err := file.Truncate(int64(size)); err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to set paper SHM size: %w", err)
	}

	// Memory map the file
	data, err := syscall.Mmap(
		int(file.Fd()),
		0,
		size,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_SHARED,
	)
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to mmap paper SHM: %w", err)
	}

	layout := (*paper.PaperTradingState)(unsafe.Pointer(&data[0]))

	// Initialize if new
	if layout.Magic != paper.PaperSHMMagic {
		layout.Magic = paper.PaperSHMMagic
		layout.Version = paper.PaperSHMVersion
		layout.StateSequence = 0
	}

	return &PaperWriter{
		file:   file,
		data:   data,
		layout: layout,
	}, nil
}

// Close unmaps and closes the shared memory
func (w *PaperWriter) Close() error {
	if err := syscall.Munmap(w.data); err != nil {
		return fmt.Errorf("failed to unmap paper SHM: %w", err)
	}
	return w.file.Close()
}

// Layout returns the raw layout pointer
func (w *PaperWriter) Layout() *paper.PaperTradingState {
	return w.layout
}

// MarkTradesPersisted marks trades from head to newHead as persisted
func (w *PaperWriter) MarkTradesPersisted(newHead uint32) {
	atomic.StoreUint32(&w.layout.TradesHead, newHead)
}

// IncrementStateSequence increments the state sequence number
func (w *PaperWriter) IncrementStateSequence() {
	atomic.AddUint32(&w.layout.StateSequence, 1)
}
