/**
 * Paper Trading Dashboard - Main Application
 */

// State
let currentStrategyId = null;
let strategies = [];
let ws = null;
let pollInterval = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 3;  // Reduced for faster fallback to polling
const RECONNECT_DELAY = 1000;
const POLL_INTERVAL_MS = 2000;  // Poll every 2 seconds (reduced from 1s for performance)
let usePolling = true;  // Use HTTP polling by default (WebSocket often blocked by proxies)

// Filter state
let lastQuoteHistory = [];
let lastQuotes = [];
let currentMarketSlug = null;  // Track current market for quote chart filtering
let allStrategiesData = []; // Store full strategy data for filtering
let currentAssetTab = 'all';  // Track current asset tab
let lastFullState = null;  // Store last full state for asset filtering

// DOM Elements
const connectionDot = document.getElementById('connectionDot');
const connectionText = document.getElementById('connectionText');
const strategyCards = document.getElementById('strategyCards');
const strategySelect = document.getElementById('strategySelect');
const mainContent = document.getElementById('mainContent');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadStrategies();
    initCharts();
});

// API Functions
async function loadStrategies() {
    try {
        const response = await fetch('/api/strategies');
        strategies = await response.json();

        // Also fetch full strategy data for filtering
        allStrategiesData = await Promise.all(
            strategies.map(async s => {
                try {
                    const resp = await fetch(`/api/strategies/${s.strategy_id}`);
                    return await resp.json();
                } catch {
                    return s;
                }
            })
        );

        updateStrategyFilterOptions();
        renderStrategyCards();
        updateStrategySelect();
        applyStrategyFilters();
    } catch (error) {
        console.error('Failed to load strategies:', error);
    }
}

async function createStrategy(data) {
    try {
        const response = await fetch('/api/strategies', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (response.ok) {
            await loadStrategies();
            return true;
        }
        return false;
    } catch (error) {
        console.error('Failed to create strategy:', error);
        return false;
    }
}

async function deleteStrategy(strategyId) {
    if (!confirm('Are you sure you want to delete this strategy?')) return;
    try {
        await fetch(`/api/strategies/${strategyId}`, { method: 'DELETE' });
        await loadStrategies();
        if (currentStrategyId === strategyId) {
            currentStrategyId = null;
            mainContent.style.display = 'none';
            if (ws) ws.close();
        }
    } catch (error) {
        console.error('Failed to delete strategy:', error);
    }
}

// Connection Functions (WebSocket with HTTP Polling fallback)
function connectToStrategy(strategyId) {
    // Stop any existing connections
    stopConnection();

    // Try WebSocket first, fallback to polling if it fails
    if (!usePolling) {
        connectWebSocket(strategyId);
    } else {
        startPolling(strategyId);
    }
}

function stopConnection() {
    if (ws) {
        ws.close();
        ws = null;
    }
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

function connectWebSocket(strategyId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${strategyId}`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
        setConnectionStatus(true, 'ws');
        reconnectAttempts = 0;
    };

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
    };

    ws.onclose = (event) => {
        console.log('WebSocket disconnected', event.code);
        setConnectionStatus(false, 'ws');

        // If abnormal closure (1006) or too many retries, switch to polling.
        // Note: HTTP 403 from server results in code 1006 (abnormal closure) since
        // the WebSocket handshake never completes.
        if (event.code === 1006 || reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
            console.log('Switching to HTTP polling mode');
            usePolling = true;
            if (currentStrategyId) {
                startPolling(currentStrategyId);
            }
            return;
        }

        // Attempt to reconnect
        if (currentStrategyId && reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            reconnectAttempts++;
            setTimeout(() => {
                if (currentStrategyId && !usePolling) {
                    connectWebSocket(currentStrategyId);
                }
            }, RECONNECT_DELAY);
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// HTTP Polling Functions
function startPolling(strategyId) {
    if (pollInterval) {
        clearInterval(pollInterval);
    }

    console.log('Starting HTTP polling for strategy:', strategyId);
    setConnectionStatus(true, 'poll');

    // Initial fetch
    fetchStrategyState(strategyId);

    // Set up polling interval
    pollInterval = setInterval(() => {
        if (currentStrategyId === strategyId) {
            fetchStrategyState(strategyId);
        } else {
            clearInterval(pollInterval);
            pollInterval = null;
        }
    }, POLL_INTERVAL_MS);
}

async function fetchStrategyState(strategyId) {
    try {
        const response = await fetch(`/api/strategies/${strategyId}/state`);
        if (!response.ok) {
            console.error(`HTTP error ${response.status} fetching strategy state`);
            setConnectionStatus(false, 'poll');
            return;
        }
        const state = await response.json();
        updateDashboardState(state);
    } catch (error) {
        console.error('Failed to fetch strategy state:', error);
        setConnectionStatus(false, 'poll');
    }
}

function handleWebSocketMessage(message) {
    switch (message.type) {
        case 'state':
            updateDashboardState(message.data);
            break;
        case 'fill':
            handleNewFill(message.data);
            break;
        case 'connected':
            console.log('Connected to strategy:', message.data);
            break;
        case 'error':
            console.error('Server error:', message.error);
            break;
    }
}

function setConnectionStatus(connected, mode = 'ws') {
    connectionDot.classList.toggle('connected', connected);
    if (connected) {
        connectionText.textContent = mode === 'poll' ? 'Polling' : 'Connected';
    } else {
        connectionText.textContent = 'Disconnected';
    }
}

// Render Functions
function renderStrategyCards() {
    strategyCards.innerHTML = strategies.map(strategy => {
        // Get full strategy data for param tags
        const fullData = allStrategiesData.find(s => s.strategy_id === strategy.strategy_id) || strategy;
        const paramTags = getStrategyParamTags(fullData);
        const pnlByAsset = strategy.pnl_by_asset || {};

        return `
        <div class="strategy-card ${strategy.strategy_id === currentStrategyId ? 'active' : ''}"
             data-strategy-id="${strategy.strategy_id}"
             onclick="selectStrategy('${strategy.strategy_id}')">
            <div class="strategy-card-header">
                <span class="strategy-card-name">${escapeHtml(strategy.name)}</span>
                <span class="strategy-card-status ${strategy.status}">${strategy.status}</span>
            </div>
            <div class="strategy-card-pnl ${strategy.total_pnl >= 0 ? 'positive' : 'negative'}">
                ${strategy.pnl_percent >= 0 ? '+' : ''}${strategy.pnl_percent.toFixed(2)}%
            </div>
            <div class="strategy-card-asset-pnl">
                ${['btc', 'eth', 'sol', 'xrp'].map(asset => {
                    const pnl = pnlByAsset[asset] || 0;
                    const pct = (pnl / 10000) * 100;  // Each asset has $10k base
                    const cls = pnl >= 0 ? 'positive' : 'negative';
                    return `<span class="asset-pnl ${asset} ${cls}">${asset.toUpperCase()}: ${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%</span>`;
                }).join('')}
            </div>
            <div class="strategy-card-meta">
                ${strategy.assets.join(', ').toUpperCase()} | ${strategy.timeframe} |
                ${strategy.active_markets} markets
            </div>
            ${paramTags.length > 0 ? `
            <div class="strategy-card-params">
                ${paramTags.map(tag => `<span class="param-tag">${tag}</span>`).join('')}
            </div>
            ` : ''}
        </div>
    `}).join('');
}

function updateStrategySelect() {
    strategySelect.innerHTML = `
        <option value="">Select a strategy...</option>
        ${strategies.map(s => `
            <option value="${s.strategy_id}" ${s.strategy_id === currentStrategyId ? 'selected' : ''}>
                ${escapeHtml(s.name)} (${s.pnl_percent >= 0 ? '+' : ''}${s.pnl_percent.toFixed(1)}%)
            </option>
        `).join('')}
    `;
}

function selectStrategy(strategyId) {
    if (!strategyId) {
        currentStrategyId = null;
        mainContent.style.display = 'none';
        stopConnection();
        return;
    }

    currentStrategyId = strategyId;
    mainContent.style.display = 'grid';
    renderStrategyCards();
    updateStrategySelect();
    connectToStrategy(strategyId);
}

function updateDashboardState(state) {
    // Store full state for asset tab filtering
    // Only store if this is a full state update (not already filtered)
    if (state.pnl_by_asset !== undefined) {
        lastFullState = state;
    }

    // Update asset performance grid if All tab is selected
    if (currentAssetTab === 'all' && lastFullState) {
        updateAssetPerformanceGrid(lastFullState);
    }

    // Apply asset tab filter if needed
    const displayState = (currentAssetTab !== 'all' && lastFullState)
        ? filterStateByAsset(lastFullState, currentAssetTab)
        : state;

    // Update metrics
    // Show PnL % instead of equity amount (each asset has independent $10k capital)
    const pnlPct = (displayState.total_pnl / displayState.starting_capital) * 100;
    const pnlPctEl = document.getElementById('currentPnlPct');
    pnlPctEl.textContent = `${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%`;
    pnlPctEl.className = pnlPct >= 0 ? 'positive' : 'negative';
    const metricPnlPct = (displayState.total_pnl / displayState.starting_capital) * 100;
    document.getElementById('metricPnl').textContent = `${metricPnlPct >= 0 ? '+' : ''}${metricPnlPct.toFixed(2)}%`;
    document.getElementById('metricPnl').className = `metric-value ${displayState.total_pnl >= 0 ? 'positive' : 'negative'}`;
    document.getElementById('metricWinRate').textContent = `${(displayState.win_rate * 100).toFixed(1)}%`;
    document.getElementById('metricSharpe').textContent = displayState.sharpe_ratio.toFixed(2);
    document.getElementById('metricDrawdown').textContent = `${(displayState.max_drawdown * 100).toFixed(2)}%`;
    document.getElementById('metricTrades').textContent = displayState.total_trades;
    document.getElementById('metricPositions').textContent = displayState.positions.length;
    document.getElementById('currentInventory').textContent = displayState.total_inventory.toFixed(2);

    // Update orderbook
    if (displayState.quotes && displayState.quotes.length > 0) {
        const quote = displayState.quotes[0];
        currentMarketSlug = quote.slug;
        document.getElementById('marketSlug').textContent = quote.slug || '-';
        renderOrderbook(quote);
    }

    // Update fills table
    renderFillsTable(displayState.recent_fills || []);

    // Update charts (filter to last 1 week for better visualization)
    const oneWeekAgoMs = Date.now() - (7 * 24 * 3600000);  // 1 week in ms

    // Filter equity history (timestamps in nanoseconds)
    const filteredEquity = (displayState.equity_history || []).filter(
        ([ts, _]) => ts / 1000000 > oneWeekAgoMs
    );

    // Get inventory history - use per-asset if filtered, otherwise aggregate
    let inventoryHistory = [];
    if (displayState.inventory_history) {
        // Already filtered by asset
        inventoryHistory = displayState.inventory_history;
    } else if (displayState.inventory_history_by_asset) {
        // Aggregate all assets for "all" view
        const allPoints = [];
        Object.entries(displayState.inventory_history_by_asset).forEach(([asset, history]) => {
            history.forEach(([ts, inv]) => allPoints.push({ ts, inv, asset }));
        });
        // Sum inventory at each unique timestamp
        const byTime = {};
        allPoints.forEach(({ ts, inv }) => {
            byTime[ts] = (byTime[ts] || 0) + inv;
        });
        inventoryHistory = Object.entries(byTime)
            .map(([ts, inv]) => [parseInt(ts), inv])
            .sort((a, b) => a[0] - b[0]);
    }
    // Filter inventory history (timestamps in milliseconds)
    const filteredInventory = inventoryHistory.filter(([ts, _]) => ts > oneWeekAgoMs);

    updateEquityChartWrapper(filteredEquity);
    updateInventoryChartWrapper(filteredInventory);

    // Store quote data
    lastQuoteHistory = displayState.quote_history || [];
    lastQuotes = displayState.quotes || [];

    // Update quote chart (filter by current market slug and last 1 week)
    let filteredHistory = currentMarketSlug
        ? lastQuoteHistory.filter(p => p.slug === currentMarketSlug)
        : lastQuoteHistory;
    // Filter to last 1 week
    filteredHistory = filteredHistory.filter(p => p.timestamp_ms > oneWeekAgoMs);
    updateQuoteChartWrapper(filteredHistory);

    // Update current quote display
    if (lastQuotes.length > 0) {
        updateCurrentQuoteDisplay(lastQuotes[0]);
    }

    // Update strategy cards with new data (use full state, not filtered)
    const fullState = lastFullState || state;
    const idx = strategies.findIndex(s => s.strategy_id === fullState.strategy_id);
    if (idx >= 0) {
        strategies[idx].total_pnl = fullState.total_pnl;
        strategies[idx].pnl_percent = (fullState.total_pnl / fullState.starting_capital) * 100;
        strategies[idx].active_markets = fullState.quotes ? fullState.quotes.length : 0;
        strategies[idx].position_count = fullState.positions ? fullState.positions.length : 0;
        strategies[idx].pnl_by_asset = fullState.pnl_by_asset || {};
        renderStrategyCards();
        applyStrategyFilters();  // Re-apply sort/filter after re-render
    }
}

function renderOrderbook(quote) {
    const orderbook = document.getElementById('orderbook');

    // Take top 5 levels
    const asks = (quote.asks || []).slice(0, 5).reverse();
    const bids = (quote.bids || []).slice(0, 5);

    let html = '';

    // Asks (in reverse order, highest first)
    asks.forEach(level => {
        const isOurQuote = quote.our_ask && Math.abs(level.price - quote.our_ask) < 0.001;
        html += `
            <div class="orderbook-row ask ${isOurQuote ? 'our-quote' : ''}">
                <span>${level.price.toFixed(4)}</span>
                <span>${level.size.toFixed(2)}</span>
            </div>
        `;
    });

    // Spread
    const spread = quote.spread || (quote.best_ask - quote.best_bid);
    html += `<div class="orderbook-spread">Spread: ${(spread * 100).toFixed(2)}%</div>`;

    // Bids
    bids.forEach(level => {
        const isOurQuote = quote.our_bid && Math.abs(level.price - quote.our_bid) < 0.001;
        html += `
            <div class="orderbook-row bid ${isOurQuote ? 'our-quote' : ''}">
                <span>${level.price.toFixed(4)}</span>
                <span>${level.size.toFixed(2)}</span>
            </div>
        `;
    });

    orderbook.innerHTML = html;
}

function renderFillsTable(fills) {
    const tbody = document.getElementById('fillsTableBody');

    if (fills.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-secondary);">No fills yet</td></tr>';
        return;
    }

    tbody.innerHTML = fills.slice(0, 50).map(fill => {
        // Determine display direction based on side and token_side
        // BUY UP = going long (positive inventory)
        // BUY DOWN = going short (negative inventory)
        // SELL UP = closing long
        // SELL DOWN = closing short
        const tokenSide = (fill.token_side || 'up').toUpperCase();
        const isBuy = fill.side === 'BUY';
        const isUp = tokenSide === 'UP';

        // Color: green for long direction, red for short direction
        const directionClass = (isBuy && isUp) || (!isBuy && !isUp) ? 'buy' : 'sell';
        const directionText = `${fill.side} ${tokenSide}`;

        return `
            <tr>
                <td>${formatTime(fill.timestamp_ms)}</td>
                <td>${escapeHtml(fill.slug)}</td>
                <td class="${directionClass}">${directionText}</td>
                <td>${fill.price.toFixed(4)}</td>
                <td>${fill.size.toFixed(2)}</td>
                <td class="${fill.pnl >= 0 ? 'positive' : 'negative'}">${formatCurrency(fill.pnl)}</td>
            </tr>
        `;
    }).join('');
}

function handleNewFill(fill) {
    // Could add animation or notification here
    console.log('New fill:', fill);
}

// Chart Functions (using Charts module from charts.js)
function initCharts() {
    if (typeof Charts !== 'undefined') {
        Charts.createEquityChart('equityChart');
        Charts.createInventoryChart('inventoryChart');
        Charts.createQuoteChart('quoteChart');
    } else {
        console.warn('Charts module not loaded, charts will not be available');
    }
}

function updateEquityChartWrapper(history) {
    if (typeof Charts !== 'undefined') {
        Charts.updateEquityChart('equityChart', history);
    }
}

function updateInventoryChartWrapper(history) {
    if (typeof Charts !== 'undefined') {
        Charts.updateInventoryChart('inventoryChart', history);
    }
}

function updateQuoteChartWrapper(history) {
    if (typeof Charts !== 'undefined') {
        Charts.updateQuoteChart('quoteChart', history);
    }
}

function updateCurrentQuoteDisplay(quote) {
    const ourBidEl = document.getElementById('ourBid');
    const ourAskEl = document.getElementById('ourAsk');
    const timeToExpiryEl = document.getElementById('timeToExpiry');

    if (ourBidEl) {
        if (quote && quote.our_bid !== null) {
            ourBidEl.textContent = quote.our_bid.toFixed(4);
            ourBidEl.className = 'quote-value bid';
        } else {
            ourBidEl.textContent = '-';
            ourBidEl.className = 'quote-value';
        }
    }

    if (ourAskEl) {
        if (quote && quote.our_ask !== null) {
            ourAskEl.textContent = quote.our_ask.toFixed(4);
            ourAskEl.className = 'quote-value ask';
        } else {
            ourAskEl.textContent = '-';
            ourAskEl.className = 'quote-value';
        }
    }

    if (timeToExpiryEl) {
        if (quote && quote.time_to_expiry_s > 0) {
            const minutes = Math.floor(quote.time_to_expiry_s / 60);
            const seconds = Math.floor(quote.time_to_expiry_s % 60);
            timeToExpiryEl.textContent = `${minutes}m ${seconds}s`;
        } else {
            timeToExpiryEl.textContent = '-';
        }
    }
}

// Strategy Filter Functions
function updateStrategyFilterOptions() {
    // Extract unique values from all strategies
    const dfValues = new Set();
    const volModes = new Set();
    const zValues = new Set();
    const assets = new Set();

    allStrategiesData.forEach(s => {
        if (s.config && s.config.quote_model) {
            const qm = s.config.quote_model;
            if (qm.t_df !== undefined) dfValues.add(qm.t_df);
            if (qm.vol_mode) volModes.add(qm.vol_mode);
            if (qm.z !== undefined) zValues.add(qm.z);
        }
        if (s.assets) {
            s.assets.forEach(a => assets.add(a.toUpperCase()));
        }
    });

    // Update filter dropdowns
    updateFilterDropdown('filterDf', Array.from(dfValues).sort((a, b) => a - b), v => `df=${v}`);
    updateFilterDropdown('filterVolMode', Array.from(volModes).sort(), v => v);
    updateFilterDropdown('filterZ', Array.from(zValues).sort((a, b) => a - b), v => `z=${v}`);
    updateFilterDropdown('filterAsset', Array.from(assets).sort(), v => v);
}

function updateFilterDropdown(elementId, values, labelFn) {
    const el = document.getElementById(elementId);
    if (!el) return;

    const currentValue = el.value;
    let html = '<option value="all">All</option>';
    values.forEach(v => {
        const strVal = String(v);
        const selected = currentValue === strVal ? 'selected' : '';
        html += `<option value="${strVal}" ${selected}>${labelFn(v)}</option>`;
    });
    el.innerHTML = html;
    if (currentValue && currentValue !== 'all') {
        el.value = currentValue;
    }
}

function applyStrategyFilters() {
    const filterDf = document.getElementById('filterDf')?.value || 'all';
    const filterVolMode = document.getElementById('filterVolMode')?.value || 'all';
    const filterZ = document.getElementById('filterZ')?.value || 'all';
    const filterAsset = document.getElementById('filterAsset')?.value || 'all';
    const sortPnl = document.getElementById('sortPnl')?.value || 'default';

    let visibleCount = 0;
    const cardsContainer = document.getElementById('strategyCards');
    const cards = Array.from(cardsContainer.querySelectorAll('.strategy-card'));

    // Apply filters
    cards.forEach(card => {
        const strategyId = card.dataset.strategyId;
        const strategy = allStrategiesData.find(s => s.strategy_id === strategyId);

        if (!strategy) {
            card.classList.add('filtered-out');
            return;
        }

        let visible = true;
        const qm = strategy.config?.quote_model || {};

        if (filterDf !== 'all' && String(qm.t_df) !== filterDf) visible = false;
        if (filterVolMode !== 'all' && qm.vol_mode !== filterVolMode) visible = false;
        if (filterZ !== 'all' && String(qm.z) !== filterZ) visible = false;
        if (filterAsset !== 'all') {
            const strategyAssets = (strategy.assets || []).map(a => a.toUpperCase());
            if (!strategyAssets.includes(filterAsset)) visible = false;
        }

        if (visible) {
            card.classList.remove('filtered-out');
            visibleCount++;
        } else {
            card.classList.add('filtered-out');
        }
    });

    // Apply sorting
    if (sortPnl !== 'default') {
        cards.sort((a, b) => {
            const strategyA = strategies.find(s => s.strategy_id === a.dataset.strategyId);
            const strategyB = strategies.find(s => s.strategy_id === b.dataset.strategyId);

            const pnlA = getStrategyPnlForSort(strategyA, sortPnl);
            const pnlB = getStrategyPnlForSort(strategyB, sortPnl);

            // Determine sort direction (all options end with -desc or -asc)
            const direction = sortPnl.endsWith('-asc') ? 1 : -1;
            return (pnlB - pnlA) * direction;
        });

        // Re-append cards in sorted order
        cards.forEach(card => cardsContainer.appendChild(card));
    }

    // Update filter count
    const countEl = document.getElementById('filterCount');
    if (countEl) {
        countEl.textContent = `${visibleCount} / ${cards.length} strategies`;
    }
}

function clearStrategyFilters() {
    document.getElementById('filterDf').value = 'all';
    document.getElementById('filterVolMode').value = 'all';
    document.getElementById('filterZ').value = 'all';
    document.getElementById('filterAsset').value = 'all';
    document.getElementById('sortPnl').value = 'default';
    applyStrategyFilters();
}

function getStrategyPnlForSort(strategy, sortOption) {
    if (!strategy) return 0;

    // Parse sort option: pnl-{asset}-{direction}
    const parts = sortOption.split('-');
    if (parts.length < 2) return strategy.total_pnl || 0;

    const asset = parts[1];  // total, btc, eth, sol, xrp

    if (asset === 'total') {
        return strategy.total_pnl || 0;
    }

    // Asset-specific PnL
    const pnlByAsset = strategy.pnl_by_asset || {};
    return pnlByAsset[asset] || 0;
}

// Asset Tab Functions
function selectAssetTab(asset) {
    currentAssetTab = asset;

    // Update tab UI
    document.querySelectorAll('.asset-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.asset === asset);
    });

    // Toggle between asset performance grid (All) and detailed dashboard (individual)
    const perfGrid = document.getElementById('assetPerformanceGrid');
    const mainContentEl = document.getElementById('mainContent');

    if (asset === 'all') {
        // Show asset performance grid, hide main dashboard
        if (perfGrid) perfGrid.style.display = 'grid';
        if (mainContentEl) mainContentEl.style.display = 'none';
        updateAssetPerformanceGrid(lastFullState);
    } else {
        // Hide asset performance grid, show main dashboard
        if (perfGrid) perfGrid.style.display = 'none';
        if (mainContentEl && currentStrategyId) mainContentEl.style.display = 'grid';
        // Filter and update dashboard for selected asset
        if (lastFullState) {
            updateDashboardState(filterStateByAsset(lastFullState, asset));
        }
    }
}

// Update asset performance grid when All tab is selected
function updateAssetPerformanceGrid(state) {
    if (!state) return;

    const pnlByAsset = state.pnl_by_asset || {};
    const CAPITAL_PER_ASSET = 10000;  // $10,000 per asset

    ['btc', 'eth', 'sol', 'xrp'].forEach(asset => {
        const pnl = pnlByAsset[asset] || 0;
        const percent = (pnl / CAPITAL_PER_ASSET) * 100;

        const pnlEl = document.getElementById(`${asset}Pnl`);
        const percentEl = document.getElementById(`${asset}Percent`);

        if (pnlEl) {
            pnlEl.textContent = formatCurrency(pnl);
            pnlEl.className = `asset-perf-pnl ${pnl >= 0 ? 'positive' : 'negative'}`;
        }
        if (percentEl) {
            const sign = percent >= 0 ? '+' : '';
            percentEl.textContent = `(${sign}${percent.toFixed(2)}%)`;
            percentEl.className = `asset-perf-percent ${percent >= 0 ? 'positive' : 'negative'}`;
        }
    });
}

function filterStateByAsset(state, asset) {
    if (asset === 'all') return state;

    // Create a filtered copy
    const filtered = { ...state };

    // Helper to check if item matches asset (by slug prefix or asset field)
    const matchesAsset = (item) => {
        if (!item) return false;
        // Check asset field first (more reliable)
        if (item.asset?.toLowerCase() === asset) return true;
        // Fallback to slug prefix check
        return item.slug?.toLowerCase().startsWith(asset);
    };

    filtered.positions = (state.positions || []).filter(matchesAsset);
    filtered.quotes = (state.quotes || []).filter(matchesAsset);
    filtered.quote_history = (state.quote_history || []).filter(matchesAsset);

    // Use per-asset fill history if available, otherwise filter from recent_fills
    if (state.fills_by_asset && state.fills_by_asset[asset]) {
        filtered.recent_fills = state.fills_by_asset[asset];
    } else {
        filtered.recent_fills = (state.recent_fills || []).filter(matchesAsset);
    }

    // Use per-asset inventory history if available
    if (state.inventory_history_by_asset && state.inventory_history_by_asset[asset]) {
        filtered.inventory_history = state.inventory_history_by_asset[asset];
    } else {
        filtered.inventory_history = [];
    }

    // Use per-asset metrics if available
    const assetMetrics = state.metrics_by_asset && state.metrics_by_asset[asset];
    if (assetMetrics) {
        filtered.total_pnl = assetMetrics.total_pnl;
        filtered.realized_pnl = assetMetrics.realized_pnl;
        filtered.unrealized_pnl = assetMetrics.unrealized_pnl;
        filtered.total_inventory = assetMetrics.inventory;
        filtered.total_trades = assetMetrics.total_trades;
        filtered.win_count = assetMetrics.win_count;
        filtered.win_rate = assetMetrics.win_rate;
    } else {
        // Fallback: recalculate from filtered positions
        filtered.total_pnl = filtered.positions.reduce(
            (sum, p) => sum + (p.realized_pnl || 0) + (p.unrealized_pnl || 0), 0
        );
        filtered.realized_pnl = filtered.positions.reduce(
            (sum, p) => sum + (p.realized_pnl || 0), 0
        );
        filtered.unrealized_pnl = filtered.positions.reduce(
            (sum, p) => sum + (p.unrealized_pnl || 0), 0
        );
        filtered.total_inventory = filtered.positions.reduce(
            (sum, p) => sum + (p.side === 'up' ? p.size : -p.size), 0
        );
    }

    // Use asset-specific PnL if available (overrides metrics)
    if (state.pnl_by_asset && state.pnl_by_asset[asset] !== undefined) {
        filtered.total_pnl = state.pnl_by_asset[asset];
    }

    // Use asset-specific equity history if available
    if (state.equity_history_by_asset && state.equity_history_by_asset[asset]) {
        filtered.equity_history = state.equity_history_by_asset[asset];
    }

    return filtered;
}

function getStrategyParamTags(strategy) {
    const tags = [];
    const qm = strategy.config?.quote_model || {};

    if (qm.t_df !== undefined) tags.push(`df=${qm.t_df}`);
    if (qm.vol_mode) tags.push(qm.vol_mode);
    if (qm.z !== undefined) tags.push(`z=${qm.z}`);
    if (qm.distribution) tags.push(qm.distribution);

    return tags;
}

// Modal Functions
function openAddStrategyModal() {
    document.getElementById('addStrategyModal').classList.add('active');
}

function closeAddStrategyModal() {
    document.getElementById('addStrategyModal').classList.remove('active');
    document.getElementById('addStrategyForm').reset();
}

async function submitAddStrategy(event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);

    const data = {
        name: formData.get('name'),
        assets: Array.from(form.assets.selectedOptions).map(o => o.value),
        timeframe: formData.get('timeframe'),
        starting_capital: parseFloat(formData.get('starting_capital')),
        quote_model: {
            base_spread: parseFloat(formData.get('base_spread'))
        },
        size_model: {
            max_position: parseFloat(formData.get('max_position'))
        }
    };

    const success = await createStrategy(data);
    if (success) {
        closeAddStrategyModal();
    } else {
        alert('Failed to create strategy');
    }
}

// Utility Functions
function formatCurrency(value) {
    if (value === undefined || value === null) return '$0.00';
    const sign = value >= 0 ? '' : '-';
    return `${sign}$${Math.abs(value).toFixed(2)}`;
}

function formatTime(timestampMs) {
    const date = new Date(timestampMs);
    return date.toLocaleTimeString();
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Ping to keep WebSocket alive (only when using WebSocket mode)
setInterval(() => {
    if (!usePolling && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
    }
}, 30000);

// Periodically refresh strategy cards (for polling mode to update cards)
setInterval(() => {
    if (usePolling) {
        loadStrategies();
    }
}, 2000);  // Reduced from 5s to 2s for faster PnL updates on strategy cards
