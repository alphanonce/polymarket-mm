package shm

import (
    "testing"
    "unsafe"
)

func TestSizes(t *testing.T) {
    // These expected sizes must match the C header (shm_layout.h) and Python ctypes.
    // If this test fails, update all three implementations to match.
    cases := []struct {
        name string
        got  uintptr
        want uintptr
    }{
        {"PriceLevel", unsafe.Sizeof(PriceLevel{}), 16},
        {"MarketBook", unsafe.Sizeof(MarketBook{}), 776},
        {"SharedMemoryLayout", unsafe.Sizeof(SharedMemoryLayout{}), 86880},
    }
    for _, c := range cases {
        if c.got != c.want {
            t.Errorf("%s size mismatch: got %d, want %d", c.name, c.got, c.want)
        }
    }
}
