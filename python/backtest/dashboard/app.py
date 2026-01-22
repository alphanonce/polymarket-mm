"""
Backtest Dashboard - Streamlit App

Interactive dashboard for visualizing backtest results.
Supports hourly period breakdown with price/quote/fill charts.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import streamlit as st

from backtest.dashboard.charts import (
    create_asset_summary_chart,
    create_combined_period_chart,
    create_fills_chart,
    create_period_summary_chart,
    create_pnl_chart,
    create_position_chart,
    create_price_chart,
)
from backtest.storage import BacktestStorage

# Page config
st.set_page_config(
    page_title="Backtest Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def format_currency(value: float) -> str:
    """Format as currency with color."""
    if value >= 0:
        return f"${value:,.2f}"
    else:
        return f"-${abs(value):,.2f}"


def format_number(value: float) -> str:
    """Format large numbers."""
    return f"{value:,.0f}"


def main():
    st.title("📊 Backtest Dashboard")

    # Initialize storage
    storage = BacktestStorage()

    # Sidebar: Run selection
    st.sidebar.header("📁 Select Run")

    runs = storage.list_runs()
    if not runs:
        st.warning("No backtest runs found. Run a backtest first:")
        st.code("make run-backtest-enhanced")
        return

    run_options = {
        f"{r['run_id']} ({r['start_time'][:10] if r['start_time'] else 'N/A'})": r["run_id"]
        for r in runs
    }

    selected_label = st.sidebar.selectbox("Run", list(run_options.keys()))
    selected_run_id = run_options[selected_label]

    # Load report
    report = storage.load(selected_run_id)
    if not report:
        st.error(f"Failed to load run: {selected_run_id}")
        return

    # Sidebar: Asset filter
    st.sidebar.header("🎯 Filters")
    assets = list(report.asset_results.keys())
    selected_asset = st.sidebar.selectbox("Asset", ["All"] + [a.upper() for a in assets])

    # === SUMMARY VIEW ===
    st.header("Summary")

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total PnL",
            format_currency(report.total_pnl),
            delta=f"{report.avg_sharpe:.2f} Sharpe" if report.avg_sharpe else None,
        )

    with col2:
        st.metric("Total Fills", format_number(report.total_fills))

    with col3:
        st.metric("Total Volume", format_number(report.total_volume))

    with col4:
        st.metric("Win Rate", f"{report.avg_win_rate:.1%}")

    # Config summary
    with st.expander("⚙️ Configuration", expanded=False):
        config = report.config
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Spread:** {config.get('base_spread', 0):.2%}")
            st.write(f"**Size:** {config.get('base_size', 0)}")
        with col2:
            st.write(f"**Max Position:** {config.get('max_position', 0)}")
            st.write(f"**Refresh:** {config.get('quote_refresh_sec', 0)}s")
        with col3:
            st.write(f"**Start Date:** {config.get('start_date', 'All')}")
            st.write(f"**End Date:** {config.get('end_date', 'All')}")

    # Asset summary chart
    if selected_asset == "All" and len(assets) > 1:
        st.subheader("📈 PnL by Asset")
        fig = create_asset_summary_chart(report.asset_results, "total_pnl")
        st.plotly_chart(fig, use_container_width=True)

    # === HOURLY PERIOD TABLE ===
    st.header("⏰ Hourly Periods")

    # Get periods for selected asset(s)
    if selected_asset == "All":
        all_periods_df = report.get_all_periods()
    else:
        asset_key = selected_asset.lower()
        if asset_key in report.asset_results:
            result = report.asset_results[asset_key]
            all_periods_df = result.get_period_dataframe()
        else:
            all_periods_df = pd.DataFrame()

    if all_periods_df.empty:
        st.info("No period data available.")
    else:
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Periods", len(all_periods_df))
        with col2:
            avg_pnl = all_periods_df["total_pnl"].mean()
            st.metric("Avg PnL/Period", format_currency(avg_pnl))
        with col3:
            profitable = (all_periods_df["total_pnl"] > 0).sum()
            st.metric("Profitable Periods", f"{profitable}/{len(all_periods_df)}")
        with col4:
            total_fills = all_periods_df["n_fills"].sum()
            st.metric("Total Fills", format_number(total_fills))

        # Period summary chart
        st.subheader("📊 PnL by Period")
        if selected_asset != "All":
            asset_key = selected_asset.lower()
            if asset_key in report.asset_results:
                periods = report.asset_results[asset_key].hourly_periods
                fig = create_period_summary_chart(periods, "total_pnl")
                st.plotly_chart(fig, use_container_width=True)

        # Period table
        st.subheader("📋 Period Details")

        # Format for display
        display_df = all_periods_df.copy()
        display_df["period_start"] = display_df["period_start"].dt.strftime("%Y-%m-%d %H:%M")
        display_df = display_df[
            [
                "asset",
                "period_start",
                "final_position",
                "total_pnl",
                "realized_pnl",
                "volume",
                "n_fills",
                "n_trades",
            ]
        ]
        display_df.columns = [
            "Asset",
            "Period",
            "Position",
            "Total PnL",
            "Realized PnL",
            "Volume",
            "Fills",
            "Trades",
        ]

        # Style the dataframe
        st.dataframe(
            display_df.style.format(
                {
                    "Position": "{:.2f}",
                    "Total PnL": "${:.2f}",
                    "Realized PnL": "${:.2f}",
                    "Volume": "{:,.0f}",
                }
            ),
            use_container_width=True,
            height=400,
        )

    # === DETAIL VIEW ===
    st.header("🔍 Period Detail")

    if selected_asset == "All":
        st.info("Select a specific asset to view period details.")
    else:
        asset_key = selected_asset.lower()
        if asset_key not in report.asset_results:
            st.warning(f"Asset {selected_asset} not found.")
        else:
            result = report.asset_results[asset_key]
            periods = result.hourly_periods

            if not periods:
                st.info("No periods available for this asset.")
            else:
                # Period selector
                period_options = {
                    p.period_start.strftime("%Y-%m-%d %H:%M"): i for i, p in enumerate(periods)
                }

                selected_period_label = st.selectbox("Select Period", list(period_options.keys()))
                selected_period_idx = period_options[selected_period_label]
                period = periods[selected_period_idx]

                # Period metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("PnL", format_currency(period.total_pnl))
                with col2:
                    st.metric("Position", f"{period.final_position:.2f}")
                with col3:
                    st.metric("Fills", period.n_fills)
                with col4:
                    st.metric("Volume", format_number(period.volume))
                with col5:
                    st.metric("Trades", format_number(period.n_trades))

                # Charts
                if period.timeseries:
                    # Combined chart
                    st.subheader("📈 Price, Position & PnL")
                    fig = create_combined_period_chart(period)
                    st.plotly_chart(fig, use_container_width=True)

                    # Individual charts in tabs
                    tab1, tab2, tab3, tab4 = st.tabs(
                        ["Price Chart", "Position", "PnL", "Fills"]
                    )

                    with tab1:
                        fig = create_price_chart(period.timeseries, height=500)
                        st.plotly_chart(fig, use_container_width=True)

                    with tab2:
                        fig = create_position_chart(period.timeseries, height=350)
                        st.plotly_chart(fig, use_container_width=True)

                    with tab3:
                        fig = create_pnl_chart(period.timeseries, height=350)
                        st.plotly_chart(fig, use_container_width=True)

                    with tab4:
                        fig = create_fills_chart(period.timeseries, height=350)
                        st.plotly_chart(fig, use_container_width=True)

                    # Fills table
                    st.subheader("📋 Fills")
                    from backtest.timeseries import timeseries_to_dataframe

                    ts_df = timeseries_to_dataframe(period.timeseries)
                    fills_df = ts_df[ts_df["fill_side"] != 0].copy()

                    if fills_df.empty:
                        st.info("No fills in this period.")
                    else:
                        fills_df["side"] = fills_df["fill_side"].map({1: "BUY", -1: "SELL"})
                        fills_df["time"] = fills_df["timestamp"].dt.strftime("%H:%M:%S.%f")
                        display_fills = fills_df[
                            ["time", "side", "fill_price", "fill_size", "position"]
                        ].copy()
                        display_fills.columns = ["Time", "Side", "Price", "Size", "Position After"]
                        st.dataframe(
                            display_fills.style.format(
                                {"Price": "${:.4f}", "Size": "{:.2f}", "Position After": "{:.2f}"}
                            ),
                            use_container_width=True,
                        )
                else:
                    st.warning("No timeseries data available for this period.")

    # Footer
    st.divider()
    st.caption(f"Run ID: {report.run_id} | Duration: {report.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
