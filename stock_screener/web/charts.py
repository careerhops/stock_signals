from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from uuid import uuid4


def _add_signal_highlight(fig: go.Figure, signal_date: pd.Timestamp, color: str) -> None:
    start = signal_date - pd.Timedelta(days=2)
    end = signal_date + pd.Timedelta(days=2)
    fig.add_vrect(
        x0=start,
        x1=end,
        fillcolor=color,
        opacity=0.18,
        line_width=0,
        layer="below",
    )


def build_signal_chart(strategy_output: pd.DataFrame, exchange: str, symbol: str, height: int = 620) -> str:
    if strategy_output.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{exchange}:{symbol} - No candle data available")
        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    frame = strategy_output.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    buy_rows = frame[frame["final_buy"]]
    sell_rows = frame[frame["final_sell"]]

    fig = go.Figure()

    for row in buy_rows.itertuples():
        _add_signal_highlight(fig, row.date, "#00b879")

    for row in sell_rows.itertuples():
        _add_signal_highlight(fig, row.date, "#ff0055")

    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["close"],
            mode="lines",
            name="Weekly Close",
            line={"color": "#17202a", "width": 2.5},
            hovertemplate="Date: %{x|%d %b %Y}<br>Close: %{y:.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["upper_level"],
            mode="lines",
            name="Structural Ceiling",
            line={"color": "rgba(185, 28, 28, 0.38)", "width": 1, "dash": "dot"},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["lower_level"],
            mode="lines",
            name="Structural Floor",
            line={"color": "rgba(4, 120, 87, 0.38)", "width": 1, "dash": "dot"},
        )
    )

    if not buy_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_rows["date"],
                y=buy_rows["close"],
                mode="markers+text",
                name="BUY",
                text=["BUY"] * len(buy_rows),
                textposition="bottom center",
                textfont={"color": "#047857", "size": 15, "family": "Arial Black, Arial, sans-serif"},
                marker={
                    "symbol": "triangle-up",
                    "size": 24,
                    "color": "#00b879",
                    "line": {"color": "#004d35", "width": 3},
                },
            )
        )

        fig.add_trace(
            go.Scatter(
                x=buy_rows["date"],
                y=buy_rows["demand_zone"],
                mode="markers",
                name="Demand Zone",
                marker={
                    "symbol": "circle",
                    "size": 9,
                    "color": "rgba(4, 120, 87, 0.55)",
                },
            )
        )

    if not sell_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_rows["date"],
                y=sell_rows["close"],
                mode="markers+text",
                name="SELL",
                text=["SELL"] * len(sell_rows),
                textposition="top center",
                textfont={"color": "#be123c", "size": 15, "family": "Arial Black, Arial, sans-serif"},
                marker={
                    "symbol": "triangle-down",
                    "size": 24,
                    "color": "#ff0055",
                    "line": {"color": "#6f0027", "width": 3},
                },
            )
        )

        fig.add_trace(
            go.Scatter(
                x=sell_rows["date"],
                y=sell_rows["supply_zone"],
                mode="markers",
                name="Supply Zone",
                marker={
                    "symbol": "circle",
                    "size": 9,
                    "color": "rgba(185, 28, 28, 0.55)",
                },
            )
        )

    if not frame.empty:
        latest_date = frame["date"].max()
        default_start = latest_date - pd.Timedelta(weeks=104)
        if default_start < frame["date"].min():
            default_start = frame["date"].min()
        default_range = [default_start, latest_date + pd.Timedelta(days=7)]
    else:
        default_range = None

    for row in buy_rows.itertuples():
        fig.add_vline(
            x=row.date,
            line_width=2,
            line_dash="solid",
            line_color="rgba(0, 184, 121, 0.72)",
        )

    for row in sell_rows.itertuples():
        fig.add_vline(
            x=row.date,
            line_width=2,
            line_dash="solid",
            line_color="rgba(255, 0, 85, 0.72)",
        )

    chart_width = max(1400, len(frame) * 18)

    fig.update_layout(
        title=f"{exchange}:{symbol} Weekly Buy/Sell Signal Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 50, "r": 28, "t": 80, "b": 44},
        height=height,
        width=chart_width,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )

    weekly_ticks = frame["date"].drop_duplicates().sort_values()

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(217, 225, 234, 0.7)",
        rangeslider={"visible": False},
        tickmode="array",
        tickvals=weekly_ticks,
        ticktext=[date.strftime("%d %b %Y") for date in weekly_ticks],
        tickformat="%d %b %Y",
        tickangle=-45,
        range=default_range,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(217, 225, 234, 0.7)",
        title="Price",
        tickformat=".2f",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
    )

    chart_id = f"chart-scroll-{uuid4().hex}"
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displaylogo": False, "responsive": False})
    return (
        f'<div id="{chart_id}" class="wide-chart-scroll">'
        f'<div class="wide-chart-inner">{chart_html}</div>'
        "</div>"
        "<script>"
        f'const el = document.getElementById("{chart_id}");'
        "if (el) { requestAnimationFrame(() => { el.parentElement.scrollLeft = el.parentElement.scrollWidth; }); }"
        "</script>"
    )


def latest_signal_summary(strategy_output: pd.DataFrame) -> dict[str, Any]:
    if strategy_output.empty:
        return {"signal": "NONE", "date": "", "close": ""}

    signals = strategy_output[strategy_output["signal"].isin(["BUY", "SELL"])].copy()
    if signals.empty:
        return {"signal": "NONE", "date": "", "close": ""}

    latest = signals.sort_values("date").iloc[-1]
    return {
        "signal": latest["signal"],
        "date": latest["date"],
        "close": latest["close"],
    }
