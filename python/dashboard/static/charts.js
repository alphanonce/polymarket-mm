/**
 * Paper Trading Dashboard - Chart Configurations
 *
 * Plotly.js chart setup and update functions.
 */

// Chart color scheme (matches CSS variables)
const CHART_COLORS = {
    bgPrimary: '#0d1117',
    bgSecondary: '#161b22',
    bgTertiary: '#21262d',
    textPrimary: '#e6edf3',
    textSecondary: '#8b949e',
    borderColor: '#30363d',
    green: '#3fb950',
    red: '#f85149',
    blue: '#58a6ff',
    yellow: '#d29922',
};

// Common layout settings
const COMMON_LAYOUT = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {
        color: CHART_COLORS.textPrimary,
        size: 11,
        family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif'
    },
    margin: { l: 50, r: 20, t: 10, b: 30 },
};

// Common chart config (disables mode bar, enables responsiveness)
const COMMON_CONFIG = {
    responsive: true,
    displayModeBar: false,
};

/**
 * Create the equity curve chart
 * @param {string} elementId - DOM element ID
 */
function createEquityChart(elementId) {
    const layout = {
        ...COMMON_LAYOUT,
        xaxis: {
            showgrid: false,
            color: CHART_COLORS.textSecondary,
            tickformat: '%H:%M',
        },
        yaxis: {
            showgrid: true,
            gridcolor: CHART_COLORS.bgTertiary,
            color: CHART_COLORS.textSecondary,
            tickprefix: '$',
            tickformat: ',.0f',
        },
    };

    const trace = {
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines',
        line: {
            color: CHART_COLORS.blue,
            width: 2,
        },
        fill: 'tozeroy',
        fillcolor: 'rgba(88, 166, 255, 0.1)',
        name: 'Equity',
    };

    Plotly.newPlot(elementId, [trace], layout, COMMON_CONFIG);
}

/**
 * Update the equity chart with new data
 * @param {string} elementId - DOM element ID
 * @param {Array} history - Array of [timestamp_ns, equity] tuples
 */
function updateEquityChart(elementId, history) {
    if (!history || history.length === 0) return;

    // Convert nanosecond timestamps to Date objects
    const x = history.map(([ts, _]) => new Date(ts / 1000000));
    const y = history.map(([_, equity]) => equity);

    // Determine line color based on overall performance
    const startEquity = y[0];
    const currentEquity = y[y.length - 1];
    const lineColor = currentEquity >= startEquity ? CHART_COLORS.green : CHART_COLORS.red;
    const fillColor = currentEquity >= startEquity
        ? 'rgba(63, 185, 80, 0.1)'
        : 'rgba(248, 81, 73, 0.1)';

    Plotly.update(elementId, {
        x: [x],
        y: [y],
        'line.color': [lineColor],
        fillcolor: [fillColor],
    });
}

/**
 * Create the inventory history chart
 * @param {string} elementId - DOM element ID
 */
function createInventoryChart(elementId) {
    const layout = {
        ...COMMON_LAYOUT,
        xaxis: {
            showgrid: false,
            color: CHART_COLORS.textSecondary,
            tickformat: '%H:%M',
        },
        yaxis: {
            showgrid: true,
            gridcolor: CHART_COLORS.bgTertiary,
            color: CHART_COLORS.textSecondary,
            zeroline: true,
            zerolinecolor: CHART_COLORS.textSecondary,
            zerolinewidth: 1,
        },
    };

    const trace = {
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines',
        line: {
            color: CHART_COLORS.yellow,
            width: 2,
        },
        name: 'Inventory',
    };

    Plotly.newPlot(elementId, [trace], layout, COMMON_CONFIG);
}

/**
 * Update the inventory chart with new data
 * @param {string} elementId - DOM element ID
 * @param {Array} history - Array of [timestamp_ms, inventory] tuples
 */
function updateInventoryChart(elementId, history) {
    if (!history || history.length === 0) return;

    const x = history.map(([ts, _]) => new Date(ts));
    const y = history.map(([_, inv]) => inv);

    Plotly.update(elementId, { x: [x], y: [y] });
}

/**
 * Create a PnL distribution histogram
 * @param {string} elementId - DOM element ID
 */
function createPnLDistributionChart(elementId) {
    const layout = {
        ...COMMON_LAYOUT,
        xaxis: {
            showgrid: false,
            color: CHART_COLORS.textSecondary,
            title: 'PnL ($)',
        },
        yaxis: {
            showgrid: true,
            gridcolor: CHART_COLORS.bgTertiary,
            color: CHART_COLORS.textSecondary,
            title: 'Frequency',
        },
        bargap: 0.1,
    };

    const trace = {
        x: [],
        type: 'histogram',
        marker: {
            color: CHART_COLORS.blue,
            line: {
                color: CHART_COLORS.bgSecondary,
                width: 1,
            },
        },
        name: 'PnL Distribution',
    };

    Plotly.newPlot(elementId, [trace], layout, COMMON_CONFIG);
}

/**
 * Update PnL distribution chart
 * @param {string} elementId - DOM element ID
 * @param {Array} pnls - Array of PnL values
 */
function updatePnLDistributionChart(elementId, pnls) {
    if (!pnls || pnls.length === 0) return;

    // Color bins based on positive/negative
    const colors = pnls.map(pnl => pnl >= 0 ? CHART_COLORS.green : CHART_COLORS.red);

    Plotly.update(elementId, {
        x: [pnls],
        'marker.color': [colors],
    });
}

/**
 * Create a multi-strategy comparison chart
 * @param {string} elementId - DOM element ID
 */
function createStrategyComparisonChart(elementId) {
    const layout = {
        ...COMMON_LAYOUT,
        xaxis: {
            showgrid: false,
            color: CHART_COLORS.textSecondary,
            tickformat: '%H:%M',
        },
        yaxis: {
            showgrid: true,
            gridcolor: CHART_COLORS.bgTertiary,
            color: CHART_COLORS.textSecondary,
            ticksuffix: '%',
            title: 'Return',
        },
        showlegend: true,
        legend: {
            orientation: 'h',
            yanchor: 'bottom',
            y: 1.02,
            xanchor: 'right',
            x: 1,
            font: { size: 10 },
        },
    };

    Plotly.newPlot(elementId, [], layout, COMMON_CONFIG);
}

/**
 * Update strategy comparison chart
 * @param {string} elementId - DOM element ID
 * @param {Array} strategies - Array of {name, history: [[ts, equity]...], startingCapital}
 */
function updateStrategyComparisonChart(elementId, strategies) {
    if (!strategies || strategies.length === 0) return;

    const colors = [
        CHART_COLORS.blue,
        CHART_COLORS.green,
        CHART_COLORS.yellow,
        CHART_COLORS.red,
        '#a371f7',  // purple
        '#f778ba',  // pink
    ];

    const traces = strategies.map((strategy, idx) => {
        const history = strategy.history || [];
        const startingCapital = strategy.startingCapital || 10000;

        return {
            x: history.map(([ts, _]) => new Date(ts / 1000000)),
            y: history.map(([_, equity]) => ((equity - startingCapital) / startingCapital) * 100),
            type: 'scatter',
            mode: 'lines',
            line: {
                color: colors[idx % colors.length],
                width: 2,
            },
            name: strategy.name,
        };
    });

    Plotly.react(elementId, traces, undefined, COMMON_CONFIG);
}

/**
 * Create a drawdown chart
 * @param {string} elementId - DOM element ID
 */
function createDrawdownChart(elementId) {
    const layout = {
        ...COMMON_LAYOUT,
        xaxis: {
            showgrid: false,
            color: CHART_COLORS.textSecondary,
            tickformat: '%H:%M',
        },
        yaxis: {
            showgrid: true,
            gridcolor: CHART_COLORS.bgTertiary,
            color: CHART_COLORS.textSecondary,
            ticksuffix: '%',
            autorange: 'reversed',  // Drawdown goes down
        },
    };

    const trace = {
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines',
        fill: 'tozeroy',
        line: {
            color: CHART_COLORS.red,
            width: 1,
        },
        fillcolor: 'rgba(248, 81, 73, 0.2)',
        name: 'Drawdown',
    };

    Plotly.newPlot(elementId, [trace], layout, COMMON_CONFIG);
}

/**
 * Update drawdown chart from equity history
 * @param {string} elementId - DOM element ID
 * @param {Array} history - Array of [timestamp_ns, equity] tuples
 */
function updateDrawdownChart(elementId, history) {
    if (!history || history.length < 2) return;

    const x = [];
    const y = [];
    let maxEquity = 0;

    for (const [ts, equity] of history) {
        maxEquity = Math.max(maxEquity, equity);
        const drawdown = maxEquity > 0 ? ((maxEquity - equity) / maxEquity) * 100 : 0;
        x.push(new Date(ts / 1000000));
        y.push(drawdown);
    }

    Plotly.update(elementId, { x: [x], y: [y] });
}

/**
 * Create the quote history chart
 * @param {string} elementId - DOM element ID
 */
function createQuoteChart(elementId) {
    const layout = {
        ...COMMON_LAYOUT,
        xaxis: {
            showgrid: false,
            color: CHART_COLORS.textSecondary,
            tickformat: '%H:%M:%S',
        },
        yaxis: {
            showgrid: true,
            gridcolor: CHART_COLORS.bgTertiary,
            color: CHART_COLORS.textSecondary,
            tickformat: '.4f',
        },
        showlegend: true,
        legend: {
            orientation: 'h',
            yanchor: 'bottom',
            y: 1.02,
            xanchor: 'right',
            x: 1,
            font: { size: 10 },
        },
    };

    const bidTrace = {
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines',
        line: {
            color: CHART_COLORS.green,
            width: 2,
        },
        name: 'Our Bid',
    };

    const askTrace = {
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines',
        line: {
            color: CHART_COLORS.red,
            width: 2,
        },
        name: 'Our Ask',
    };

    const marketBidTrace = {
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines',
        line: {
            color: '#2dd4bf',  // teal
            width: 1,
            dash: 'dot',
        },
        name: 'Market Bid',
    };

    const marketAskTrace = {
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines',
        line: {
            color: '#fb923c',  // orange
            width: 1,
            dash: 'dot',
        },
        name: 'Market Ask',
    };

    Plotly.newPlot(elementId, [bidTrace, askTrace, marketBidTrace, marketAskTrace], layout, COMMON_CONFIG);
}

/**
 * Update the quote chart with new data
 * @param {string} elementId - DOM element ID
 * @param {Array} history - Array of QuoteHistoryPoint objects
 */
function updateQuoteChart(elementId, history) {
    if (!history || history.length === 0) return;

    const x = history.map(p => new Date(p.timestamp_ms));
    const bidY = history.map(p => p.our_bid);
    const askY = history.map(p => p.our_ask);
    const marketBidY = history.map(p => p.best_bid);
    const marketAskY = history.map(p => p.best_ask);

    Plotly.update(elementId, {
        x: [x, x, x, x],
        y: [bidY, askY, marketBidY, marketAskY],
    });
}

// Export functions for use in app.js
window.Charts = {
    createEquityChart,
    updateEquityChart,
    createInventoryChart,
    updateInventoryChart,
    createPnLDistributionChart,
    updatePnLDistributionChart,
    createStrategyComparisonChart,
    updateStrategyComparisonChart,
    createDrawdownChart,
    updateDrawdownChart,
    createQuoteChart,
    updateQuoteChart,
    COLORS: CHART_COLORS,
};
