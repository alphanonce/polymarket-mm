package shm

import (
	"fmt"
	"os"
	"sync/atomic"
	"syscall"
	"unsafe"
)

// Reader provides read access to shared memory
// Used primarily for testing and monitoring
type Reader struct {
	file   *os.File
	data   []byte
	layout *SharedMemoryLayout
}

// NewReader creates a new shared memory reader
func NewReader() (*Reader, error) {
	// Try /dev/shm first, fall back to /tmp
	shmPath := "/dev/shm" + SHMName
	file, err := os.OpenFile(shmPath, os.O_RDONLY, 0666)
	if err != nil {
		shmPath = "/tmp" + SHMName
		file, err = os.OpenFile(shmPath, os.O_RDONLY, 0666)
		if err != nil {
			return nil, fmt.Errorf("failed to open shared memory file: %w", err)
		}
	}

	// Get file size
	info, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to stat shared memory file: %w", err)
	}

	size := int(info.Size())
	if size != SHMSize() {
		file.Close()
		return nil, fmt.Errorf("shared memory size mismatch: got %d, expected %d", size, SHMSize())
	}

	// Memory map the file
	data, err := syscall.Mmap(
		int(file.Fd()),
		0,
		size,
		syscall.PROT_READ,
		syscall.MAP_SHARED,
	)
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to mmap shared memory: %w", err)
	}

	layout := (*SharedMemoryLayout)(unsafe.Pointer(&data[0]))

	// Verify magic number
	if layout.Magic != SHMMagic {
		syscall.Munmap(data)
		file.Close()
		return nil, fmt.Errorf("invalid shared memory magic: got 0x%X, expected 0x%X", layout.Magic, SHMMagic)
	}

	return &Reader{
		file:   file,
		data:   data,
		layout: layout,
	}, nil
}

// Close unmaps and closes the shared memory
func (r *Reader) Close() error {
	if err := syscall.Munmap(r.data); err != nil {
		return fmt.Errorf("failed to unmap shared memory: %w", err)
	}
	return r.file.Close()
}

// Layout returns the raw layout pointer for direct access
func (r *Reader) Layout() *SharedMemoryLayout {
	return r.layout
}

// StateSequence returns the current state sequence number
func (r *Reader) StateSequence() uint32 {
	return atomic.LoadUint32(&r.layout.StateSequence)
}

// SignalSequence returns the current signal sequence number
func (r *Reader) SignalSequence() uint32 {
	return atomic.LoadUint32(&r.layout.SignalSequence)
}

// GetMarkets returns all current markets
func (r *Reader) GetMarkets() []MarketBook {
	numMarkets := atomic.LoadUint32(&r.layout.NumMarkets)
	markets := make([]MarketBook, numMarkets)
	for i := uint32(0); i < numMarkets; i++ {
		markets[i] = r.layout.Markets[i]
	}
	return markets
}

// GetMarket returns a specific market by asset ID
func (r *Reader) GetMarket(assetID string) (MarketBook, bool) {
	numMarkets := atomic.LoadUint32(&r.layout.NumMarkets)
	for i := uint32(0); i < numMarkets; i++ {
		if AssetIDToString(r.layout.Markets[i].AssetID) == assetID {
			return r.layout.Markets[i], true
		}
	}
	return MarketBook{}, false
}

// GetExternalPrices returns all external prices
func (r *Reader) GetExternalPrices() []ExternalPrice {
	numPrices := atomic.LoadUint32(&r.layout.NumExternalPrices)
	prices := make([]ExternalPrice, numPrices)
	for i := uint32(0); i < numPrices; i++ {
		prices[i] = r.layout.ExternalPrices[i]
	}
	return prices
}

// GetExternalPrice returns a specific external price by symbol
func (r *Reader) GetExternalPrice(symbol string) (ExternalPrice, bool) {
	numPrices := atomic.LoadUint32(&r.layout.NumExternalPrices)
	for i := uint32(0); i < numPrices; i++ {
		if SymbolToString(r.layout.ExternalPrices[i].Symbol) == symbol {
			return r.layout.ExternalPrices[i], true
		}
	}
	return ExternalPrice{}, false
}

// GetPositions returns all positions
func (r *Reader) GetPositions() []Position {
	numPositions := atomic.LoadUint32(&r.layout.NumPositions)
	positions := make([]Position, numPositions)
	for i := uint32(0); i < numPositions; i++ {
		positions[i] = r.layout.Positions[i]
	}
	return positions
}

// GetPosition returns a specific position by asset ID
func (r *Reader) GetPosition(assetID string) (Position, bool) {
	numPositions := atomic.LoadUint32(&r.layout.NumPositions)
	for i := uint32(0); i < numPositions; i++ {
		if AssetIDToString(r.layout.Positions[i].AssetID) == assetID {
			return r.layout.Positions[i], true
		}
	}
	return Position{}, false
}

// GetOpenOrders returns all open orders
func (r *Reader) GetOpenOrders() []OpenOrder {
	numOrders := atomic.LoadUint32(&r.layout.NumOpenOrders)
	orders := make([]OpenOrder, numOrders)
	for i := uint32(0); i < numOrders; i++ {
		orders[i] = r.layout.OpenOrders[i]
	}
	return orders
}

// GetEquity returns total equity and available margin
func (r *Reader) GetEquity() (total, available float64) {
	return r.layout.TotalEquity, r.layout.AvailableMargin
}

// IsTradingEnabled returns whether trading is enabled
func (r *Reader) IsTradingEnabled() bool {
	return r.layout.TradingEnabled == 1
}
