/**
 * Shared Memory Layout for Polymarket Market Making System
 *
 * This is the SINGLE SOURCE OF TRUTH for the shared memory structure.
 * Both Go and Python implementations must match this layout exactly.
 *
 * Memory Layout Rules:
 * - All structs are packed (no padding)
 * - Little-endian byte order
 * - Strings are null-terminated, fixed-size buffers
 * - Timestamps are nanoseconds since Unix epoch
 */

#ifndef SHM_LAYOUT_H
#define SHM_LAYOUT_H

#include <stdint.h>

#define SHM_MAGIC 0x504D4D4D  // "PMMM" - Polymarket Market Maker
#define SHM_VERSION 1
#define SHM_NAME "/polymarket_mm_shm"

#define MAX_MARKETS 64
#define MAX_ORDERBOOK_LEVELS 20
#define MAX_EXTERNAL_PRICES 32
#define MAX_POSITIONS 64
#define MAX_SIGNALS 16
#define MAX_OPEN_ORDERS 128

#define ASSET_ID_LEN 66      // 0x + 64 hex chars
#define SYMBOL_LEN 16
#define ORDER_ID_LEN 64

// Order sides
#define SIDE_BUY 1
#define SIDE_SELL -1

// Order types
#define ORDER_TYPE_LIMIT 0
#define ORDER_TYPE_MARKET 1

// Signal actions
#define ACTION_PLACE 0
#define ACTION_CANCEL 1
#define ACTION_MODIFY 2

// Order status
#define ORDER_STATUS_PENDING 0
#define ORDER_STATUS_OPEN 1
#define ORDER_STATUS_FILLED 2
#define ORDER_STATUS_CANCELLED 3
#define ORDER_STATUS_REJECTED 4

#pragma pack(push, 1)

/**
 * Single price level in orderbook
 */
typedef struct {
    double price;
    double size;
} PriceLevel;

/**
 * Market orderbook state for a single market
 */
typedef struct {
    char asset_id[ASSET_ID_LEN];       // Token ID (condition ID)
    uint64_t timestamp_ns;              // Last update timestamp
    double mid_price;                   // Calculated mid price
    double spread;                      // Current spread
    PriceLevel bids[MAX_ORDERBOOK_LEVELS];
    PriceLevel asks[MAX_ORDERBOOK_LEVELS];
    uint32_t bid_levels;                // Number of valid bid levels
    uint32_t ask_levels;                // Number of valid ask levels
    double last_trade_price;
    double last_trade_size;
    int8_t last_trade_side;             // SIDE_BUY or SIDE_SELL
    uint8_t _padding[7];                // Align to 8 bytes
} MarketBook;

/**
 * External price feed (e.g., Binance spot, futures, options)
 */
typedef struct {
    char symbol[SYMBOL_LEN];            // e.g., "BTCUSDT"
    double price;
    double bid;
    double ask;
    uint64_t timestamp_ns;
} ExternalPrice;

/**
 * Position in a market
 */
typedef struct {
    char asset_id[ASSET_ID_LEN];
    double position;                    // Positive = long, negative = short
    double avg_entry_price;
    double unrealized_pnl;
    double realized_pnl;
} Position;

/**
 * Open order tracking
 */
typedef struct {
    char order_id[ORDER_ID_LEN];
    char asset_id[ASSET_ID_LEN];
    int8_t side;
    double price;
    double size;
    double filled_size;
    uint8_t status;
    uint64_t created_at_ns;
    uint64_t updated_at_ns;
    uint8_t _padding[6];
} OpenOrder;

/**
 * Order signal from strategy to executor
 */
typedef struct {
    uint64_t signal_id;                 // Unique signal ID
    char asset_id[ASSET_ID_LEN];
    int8_t side;                        // SIDE_BUY or SIDE_SELL
    double price;
    double size;
    uint8_t order_type;                 // ORDER_TYPE_LIMIT or ORDER_TYPE_MARKET
    uint8_t action;                     // ACTION_PLACE, ACTION_CANCEL, ACTION_MODIFY
    char cancel_order_id[ORDER_ID_LEN]; // For cancel/modify actions
    uint8_t _padding[5];
} OrderSignal;

/**
 * Main shared memory layout
 *
 * Go executor writes: markets, external_prices, positions, open_orders
 * Python strategy writes: signals
 */
typedef struct {
    // Header
    uint32_t magic;                     // SHM_MAGIC
    uint32_t version;                   // SHM_VERSION

    // Synchronization
    uint32_t state_sequence;            // Incremented on state update
    uint32_t signal_sequence;           // Incremented on signal update

    // Timestamps
    uint64_t state_timestamp_ns;        // Last state update
    uint64_t signal_timestamp_ns;       // Last signal update

    // Market state (Go writes, Python reads)
    uint32_t num_markets;
    uint32_t _padding1;
    MarketBook markets[MAX_MARKETS];

    // External prices (Go writes, Python reads)
    uint32_t num_external_prices;
    uint32_t _padding2;
    ExternalPrice external_prices[MAX_EXTERNAL_PRICES];

    // Positions (Go writes, Python reads)
    uint32_t num_positions;
    uint32_t _padding3;
    Position positions[MAX_POSITIONS];

    // Open orders (Go writes, Python reads)
    uint32_t num_open_orders;
    uint32_t _padding4;
    OpenOrder open_orders[MAX_OPEN_ORDERS];

    // Strategy state
    double total_equity;
    double available_margin;
    uint8_t trading_enabled;
    uint8_t _padding5[7];

    // Order signals (Python writes, Go reads)
    uint32_t num_signals;
    uint32_t signals_processed;         // Go sets this after processing
    OrderSignal signals[MAX_SIGNALS];

} SharedMemoryLayout;

#pragma pack(pop)

// Compile-time size check (for documentation)
// Expected size: approximately 150KB depending on packing

#endif // SHM_LAYOUT_H
