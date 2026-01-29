package shm

import (
    "fmt"
    "unsafe"
    "testing"
)

func TestSizes(t *testing.T) {
    fmt.Println("Go sizes:")
    fmt.Printf("  PriceLevel: %d bytes\n", unsafe.Sizeof(PriceLevel{}))
    fmt.Printf("  MarketBook: %d bytes\n", unsafe.Sizeof(MarketBook{}))
    fmt.Printf("  SharedMemoryLayout: %d bytes\n", unsafe.Sizeof(SharedMemoryLayout{}))
}
