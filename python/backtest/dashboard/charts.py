"""
Plotly Charts for Backtest Dashboard

Creates interactive charts for visualizing backtest results.
"""

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtest.timeseries import HourlyPeriodResult, TimeSeriesPoint, timeseries_to_dataframe


def create_price_chart(
    timeseries: List[TimeSeriesPoint],
    title: str = "Price & Quotes",
    height: int = 400,
) -> go.Figure:
    """
    Create price chart with trade prices, bid/ask quotes, and fill markers.

    - Blue line: Market trade price
    - Green dashed: Our bid quote
    - Red dashed: Our ask quote
    - Green triangle up: Bid fill (we bought)
    - Red triangle down: Ask fill (we sold)
    """
    df = timeseries_to_dataframe(timeseries)
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    fig = go.Figure()

    # Trade price line
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["trade_price"],
            mode="lines",
            name="Trade Price",
            line=dict(color="blue", width=1.5),
        )
    )

    # Bid quote line
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["bid_quote"],
            mode="lines",
            name="Bid Quote",
            line=dict(color="green", width=1, dash="dash"),
        )
    )

    # Ask quote line
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["ask_quote"],
            mode="lines",
            name="Ask Quote",
            line=dict(color="red", width=1, dash="dash"),
        )
    )

    # Bid fills (triangles up)
    bid_fills = df[df["fill_side"] == 1]
    if not bid_fills.empty:
        fig.add_trace(
            go.Scatter(
                x=bid_fills["timestamp"],
                y=bid_fills["fill_price"],
                mode="markers",
                name="Bid Fill (Buy)",
                marker=dict(
                    symbol="triangle-up",
                    size=10,
                    color="green",
                    line=dict(color="darkgreen", width=1),
                ),
                text=[f"Size: {s:.2f}" for s in bid_fills["fill_size"]],
                hoverinfo="text+x+y",
            )
        )

    # Ask fills (triangles down)
    ask_fills = df[df["fill_side"] == -1]
    if not ask_fills.empty:
        fig.add_trace(
            go.Scatter(
                x=ask_fills["timestamp"],
                y=ask_fills["fill_price"],
                mode="markers",
                name="Ask Fill (Sell)",
                marker=dict(
                    symbol="triangle-down",
                    size=10,
                    color="red",
                    line=dict(color="darkred", width=1),
                ),
                text=[f"Size: {s:.2f}" for s in ask_fills["fill_size"]],
                hoverinfo="text+x+y",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        height=height,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def create_position_chart(
    timeseries: List[TimeSeriesPoint],
    title: str = "Position",
    height: int = 250,
) -> go.Figure:
    """Create position chart over time."""
    df = timeseries_to_dataframe(timeseries)
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    fig = go.Figure()

    # Fill area based on position sign
    colors = ["green" if p >= 0 else "red" for p in df["position"]]

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["position"],
            mode="lines",
            name="Position",
            fill="tozeroy",
            line=dict(color="purple", width=1.5),
            fillcolor="rgba(128, 0, 128, 0.3)",
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Position",
        height=height,
        showlegend=False,
    )

    return fig


def create_pnl_chart(
    timeseries: List[TimeSeriesPoint],
    title: str = "PnL",
    height: int = 250,
) -> go.Figure:
    """Create PnL chart showing realized and total PnL."""
    df = timeseries_to_dataframe(timeseries)
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    fig = go.Figure()

    # Total PnL
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["total_pnl"],
            mode="lines",
            name="Total PnL",
            line=dict(color="blue", width=1.5),
        )
    )

    # Realized PnL
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["realized_pnl"],
            mode="lines",
            name="Realized PnL",
            line=dict(color="green", width=1, dash="dot"),
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="PnL ($)",
        height=height,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def create_fills_chart(
    timeseries: List[TimeSeriesPoint],
    title: str = "Fill Distribution",
    height: int = 300,
) -> go.Figure:
    """Create bar chart of fill sizes over time."""
    df = timeseries_to_dataframe(timeseries)
    fills = df[df["fill_side"] != 0].copy()

    if fills.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No fills",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Color by side
    fills["color"] = fills["fill_side"].map({1: "green", -1: "red"})
    fills["signed_size"] = fills["fill_size"] * fills["fill_side"]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=fills["timestamp"],
            y=fills["signed_size"],
            marker_color=fills["color"],
            name="Fill Size",
            text=[f"${p:.4f}" for p in fills["fill_price"]],
            hoverinfo="text+x+y",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Fill Size (+ = Buy, - = Sell)",
        height=height,
        showlegend=False,
    )

    return fig


def create_period_summary_chart(
    periods: List[HourlyPeriodResult],
    metric: str = "total_pnl",
    title: Optional[str] = None,
    height: int = 300,
) -> go.Figure:
    """
    Create bar chart summarizing periods by a metric.

    Args:
        periods: List of period results
        metric: Which metric to plot (total_pnl, volume, n_fills, etc.)
        title: Chart title
        height: Chart height
    """
    if not periods:
        fig = go.Figure()
        fig.add_annotation(
            text="No periods available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    df = pd.DataFrame([p.to_dict() for p in periods])
    df["period_start"] = pd.to_datetime(df["period_start"])

    if metric not in df.columns:
        metric = "total_pnl"

    values = df[metric]
    colors = ["green" if v >= 0 else "red" for v in values]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["period_start"],
            y=values,
            marker_color=colors,
            name=metric,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
        )
    )

    title = title or f"{metric.replace('_', ' ').title()} by Period"

    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title=metric.replace("_", " ").title(),
        height=height,
        showlegend=False,
    )

    return fig


def create_combined_period_chart(
    period: HourlyPeriodResult,
    height: int = 800,
) -> go.Figure:
    """
    Create combined chart for a single period with price, position, and PnL.
    Uses subplots with shared x-axis.
    """
    df = timeseries_to_dataframe(period.timeseries)

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for this period",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Price & Quotes", "Position", "PnL"),
    )

    # Row 1: Price chart
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["trade_price"],
            mode="lines",
            name="Trade Price",
            line=dict(color="blue", width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["bid_quote"],
            mode="lines",
            name="Bid",
            line=dict(color="green", width=1, dash="dash"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["ask_quote"],
            mode="lines",
            name="Ask",
            line=dict(color="red", width=1, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Fill markers
    bid_fills = df[df["fill_side"] == 1]
    if not bid_fills.empty:
        fig.add_trace(
            go.Scatter(
                x=bid_fills["timestamp"],
                y=bid_fills["fill_price"],
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", size=10, color="green"),
            ),
            row=1,
            col=1,
        )

    ask_fills = df[df["fill_side"] == -1]
    if not ask_fills.empty:
        fig.add_trace(
            go.Scatter(
                x=ask_fills["timestamp"],
                y=ask_fills["fill_price"],
                mode="markers",
                name="Sell",
                marker=dict(symbol="triangle-down", size=10, color="red"),
            ),
            row=1,
            col=1,
        )

    # Row 2: Position
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["position"],
            mode="lines",
            name="Position",
            fill="tozeroy",
            line=dict(color="purple", width=1.5),
            fillcolor="rgba(128, 0, 128, 0.3)",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Row 3: PnL
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["total_pnl"],
            mode="lines",
            name="Total PnL",
            line=dict(color="blue", width=1.5),
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["realized_pnl"],
            mode="lines",
            name="Realized",
            line=dict(color="green", width=1, dash="dot"),
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    # Zero lines for position and PnL
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=3, col=1)

    title = f"{period.asset.upper()} - {period.period_start.strftime('%Y-%m-%d %H:%M')}"
    fig.update_layout(
        title=title,
        height=height,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )

    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Position", row=2, col=1)
    fig.update_yaxes(title_text="PnL ($)", row=3, col=1)

    return fig


def create_asset_summary_chart(
    asset_results: dict,
    metric: str = "total_pnl",
    height: int = 400,
) -> go.Figure:
    """Create bar chart comparing assets by a metric."""
    if not asset_results:
        fig = go.Figure()
        fig.add_annotation(
            text="No asset data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    data = []
    for asset, result in asset_results.items():
        value = getattr(result, metric, 0)
        data.append({"asset": asset.upper(), "value": value})

    df = pd.DataFrame(data).sort_values("value", ascending=False)
    colors = ["green" if v >= 0 else "red" for v in df["value"]]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["asset"],
            y=df["value"],
            marker_color=colors,
            text=[f"{v:.2f}" for v in df["value"]],
            textposition="outside",
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} by Asset",
        xaxis_title="Asset",
        yaxis_title=metric.replace("_", " ").title(),
        height=height,
        showlegend=False,
    )

    return fig
