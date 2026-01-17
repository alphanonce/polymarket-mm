package shm

import (
	"fmt"
	"os"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

// Writer provides write access to shared memory for the Go executor
type Writer struct {
	file   *os.File
	data   []byte
	layout *SharedMemoryLayout
}

// NewWriter creates a new shared memory writer
func NewWriter() (*Writer, error) {
	// Create or open the shared memory file
	// On Unix systems, we use a file in /dev/shm or a regular temp file
	shmPath := "/dev/shm" + SHMName

	// Try /dev/shm first, fall back to /tmp
	file, err := os.OpenFile(shmPath, os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		shmPath = "/tmp" + SHMName
		file, err = os.OpenFile(shmPath, os.O_RDWR|os.O_CREATE, 0666)
		if err != nil {
			return nil, fmt.Errorf("failed to create shared memory file: %w", err)
		}
	}

	// Set the file size
	size := SHMSize()
	if err := file.Truncate(int64(size)); err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to set shared memory size: %w", err)
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
		return nil, fmt.Errorf("failed to mmap shared memory: %w", err)
	}

	// Cast to layout struct
	layout := (*SharedMemoryLayout)(unsafe.Pointer(&data[0]))

	// Initialize header
	layout.Magic = SHMMagic
	layout.Version = SHMVersion
	layout.StateSequence = 0
	layout.SignalSequence = 0
	layout.TradingEnabled = 1

	return &Writer{
		file:   file,
		data:   data,
		layout: layout,
	}, nil
}

// Close unmaps and closes the shared memory
func (w *Writer) Close() error {
	if err := syscall.Munmap(w.data); err != nil {
		return fmt.Errorf("failed to unmap shared memory: %w", err)
	}
	return w.file.Close()
}

// Layout returns the raw layout pointer for direct access
func (w *Writer) Layout() *SharedMemoryLayout {
	return w.layout
}

// UpdateMarket updates or adds a market book
func (w *Writer) UpdateMarket(book MarketBook) error {
	assetID := AssetIDToString(book.AssetID)

	// Find existing or add new
	idx := -1
	numMarkets := int(atomic.LoadUint32(&w.layout.NumMarkets))
	for i := 0; i < numMarkets; i++ {
		if AssetIDToString(w.layout.Markets[i].AssetID) == assetID {
			idx = i
			break
		}
	}

	if idx == -1 {
		if numMarkets >= MaxMarkets {
			return fmt.Errorf("max markets reached")
		}
		idx = numMarkets
		atomic.StoreUint32(&w.layout.NumMarkets, uint32(numMarkets+1))
	}

	// Update the market
	book.TimestampNs = uint64(time.Now().UnixNano())
	w.layout.Markets[idx] = book

	// Update state sequence and timestamp
	atomic.AddUint32(&w.layout.StateSequence, 1)
	atomic.StoreUint64(&w.layout.StateTimestampNs, uint64(time.Now().UnixNano()))

	return nil
}

// UpdateExternalPrice updates or adds an external price
func (w *Writer) UpdateExternalPrice(price ExternalPrice) error {
	symbol := SymbolToString(price.Symbol)

	// Find existing or add new
	idx := -1
	numPrices := int(atomic.LoadUint32(&w.layout.NumExternalPrices))
	for i := 0; i < numPrices; i++ {
		if SymbolToString(w.layout.ExternalPrices[i].Symbol) == symbol {
			idx = i
			break
		}
	}

	if idx == -1 {
		if numPrices >= MaxExternalPrices {
			return fmt.Errorf("max external prices reached")
		}
		idx = numPrices
		atomic.StoreUint32(&w.layout.NumExternalPrices, uint32(numPrices+1))
	}

	// Update the price
	price.TimestampNs = uint64(time.Now().UnixNano())
	w.layout.ExternalPrices[idx] = price

	// Update state sequence
	atomic.AddUint32(&w.layout.StateSequence, 1)
	atomic.StoreUint64(&w.layout.StateTimestampNs, uint64(time.Now().UnixNano()))

	return nil
}

// UpdatePosition updates or adds a position
func (w *Writer) UpdatePosition(pos Position) error {
	assetID := AssetIDToString(pos.AssetID)

	// Find existing or add new
	idx := -1
	numPositions := int(atomic.LoadUint32(&w.layout.NumPositions))
	for i := 0; i < numPositions; i++ {
		if AssetIDToString(w.layout.Positions[i].AssetID) == assetID {
			idx = i
			break
		}
	}

	if idx == -1 {
		if numPositions >= MaxPositions {
			return fmt.Errorf("max positions reached")
		}
		idx = numPositions
		atomic.StoreUint32(&w.layout.NumPositions, uint32(numPositions+1))
	}

	w.layout.Positions[idx] = pos

	// Update state sequence
	atomic.AddUint32(&w.layout.StateSequence, 1)
	atomic.StoreUint64(&w.layout.StateTimestampNs, uint64(time.Now().UnixNano()))

	return nil
}

// AddOpenOrder adds a new open order
func (w *Writer) AddOpenOrder(order OpenOrder) error {
	numOrders := int(atomic.LoadUint32(&w.layout.NumOpenOrders))
	if numOrders >= MaxOpenOrders {
		return fmt.Errorf("max open orders reached")
	}

	order.CreatedAtNs = uint64(time.Now().UnixNano())
	order.UpdatedAtNs = order.CreatedAtNs
	w.layout.OpenOrders[numOrders] = order
	atomic.StoreUint32(&w.layout.NumOpenOrders, uint32(numOrders+1))

	return nil
}

// UpdateOpenOrder updates an existing open order
func (w *Writer) UpdateOpenOrder(orderID string, status uint8, filledSize float64) error {
	numOrders := int(atomic.LoadUint32(&w.layout.NumOpenOrders))
	for i := 0; i < numOrders; i++ {
		if OrderIDToString(w.layout.OpenOrders[i].OrderID) == orderID {
			w.layout.OpenOrders[i].Status = status
			w.layout.OpenOrders[i].FilledSize = filledSize
			w.layout.OpenOrders[i].UpdatedAtNs = uint64(time.Now().UnixNano())
			return nil
		}
	}
	return fmt.Errorf("order not found: %s", orderID)
}

// RemoveOpenOrder removes an open order by ID
func (w *Writer) RemoveOpenOrder(orderID string) error {
	numOrders := int(atomic.LoadUint32(&w.layout.NumOpenOrders))
	for i := 0; i < numOrders; i++ {
		if OrderIDToString(w.layout.OpenOrders[i].OrderID) == orderID {
			// Shift remaining orders
			for j := i; j < numOrders-1; j++ {
				w.layout.OpenOrders[j] = w.layout.OpenOrders[j+1]
			}
			atomic.StoreUint32(&w.layout.NumOpenOrders, uint32(numOrders-1))
			return nil
		}
	}
	return fmt.Errorf("order not found: %s", orderID)
}

// SetEquity updates the equity values
func (w *Writer) SetEquity(total, available float64) {
	w.layout.TotalEquity = total
	w.layout.AvailableMargin = available
}

// SetTradingEnabled enables or disables trading
func (w *Writer) SetTradingEnabled(enabled bool) {
	if enabled {
		w.layout.TradingEnabled = 1
	} else {
		w.layout.TradingEnabled = 0
	}
}

// ReadSignals reads pending signals from the strategy
func (w *Writer) ReadSignals() []OrderSignal {
	numSignals := atomic.LoadUint32(&w.layout.NumSignals)
	processed := atomic.LoadUint32(&w.layout.SignalsProcessed)

	if numSignals == 0 || processed >= numSignals {
		return nil
	}

	signals := make([]OrderSignal, numSignals-processed)
	for i := processed; i < numSignals; i++ {
		signals[i-processed] = w.layout.Signals[i]
	}

	return signals
}

// MarkSignalsProcessed marks all current signals as processed
func (w *Writer) MarkSignalsProcessed() {
	numSignals := atomic.LoadUint32(&w.layout.NumSignals)
	atomic.StoreUint32(&w.layout.SignalsProcessed, numSignals)
}

// ClearSignals clears all signals after they've been processed
func (w *Writer) ClearSignals() {
	atomic.StoreUint32(&w.layout.NumSignals, 0)
	atomic.StoreUint32(&w.layout.SignalsProcessed, 0)
}
