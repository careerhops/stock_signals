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


def _yes_no(value: Any) -> str:
    return "OK" if bool(value) else "NO"


def _check_mark(value: Any) -> str:
    return "Y" if bool(value) else "N"


def _format_float(value: Any, decimals: int = 2, suffix: str = "") -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.{decimals}f}{suffix}"


def _peak_speed_bucket(days_to_peak: Any) -> str:
    days = pd.to_numeric(pd.Series([days_to_peak]), errors="coerce").iloc[0]
    if pd.isna(days):
        return "NA"
    if days <= 30:
        return "Within 30 days"
    if days <= 60:
        return "31-60 days"
    if days <= 90:
        return "61-90 days"
    if days <= 180:
        return "91-180 days"
    if days <= 365:
        return "181-365 days"
    return "Over 1 year"


def _buy_quality_text(row: pd.Series) -> str:
    return (
        "BUY"
        f"<br>Vol {_check_mark(row.get('volume_confirmation', False))}"
        f" Trend {_check_mark(row.get('trend_confirmation', False))}"
        f"<br>Med1 {_format_float(row.get('prior_pair_return_last_1_pct'), 1, '%')}"
        f" Med3 {_format_float(row.get('median_pair_return_last_3_pct'), 1, '%')}"
    )


def _sell_quality_text(row: pd.Series) -> str:
    return f"SELL<br>Pair {_format_float(row.get('sell_pair_return_pct'), 1, '%')}"


def _quality_customdata(rows: pd.DataFrame) -> list[list[str]]:
    customdata: list[list[str]] = []
    for _, row in rows.iterrows():
        customdata.append(
            [
                _yes_no(row.get("volume_confirmation", False)),
                _format_float(row.get("volume_confirmation_ratio"), 2, "x"),
                _yes_no(row.get("trend_confirmation", False)),
                _format_float(row.get("prior_pair_return_last_1_pct"), 2, "%"),
                _format_float(row.get("median_pair_return_last_3_pct"), 2, "%"),
                _format_float(row.get("sell_pair_return_pct"), 2, "%"),
            ]
        )
    return customdata


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
            y=frame["close"],
            mode="lines",
            name="Right Price Axis",
            line={"color": "rgba(0,0,0,0)", "width": 0},
            hoverinfo="skip",
            showlegend=False,
            yaxis="y2",
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

    if "ema_20" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["ema_20"],
                mode="lines",
                name="20-week EMA",
                line={"color": "rgba(79, 142, 207, 0.72)", "width": 1.4},
            )
        )

    if "ema_50" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["ema_50"],
                mode="lines",
                name="50-week EMA",
                line={"color": "rgba(122, 139, 150, 0.72)", "width": 1.4},
            )
        )

    if not buy_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_rows["date"],
                y=buy_rows["close"],
                mode="markers+text",
                name="BUY",
                text=[_buy_quality_text(row) for _, row in buy_rows.iterrows()],
                textposition="bottom center",
                textfont={"color": "#047857", "size": 12, "family": "Arial, sans-serif"},
                marker={
                    "symbol": "triangle-up",
                    "size": 24,
                    "color": "#00b879",
                    "line": {"color": "#004d35", "width": 3},
                },
                customdata=_quality_customdata(buy_rows),
                hovertemplate=(
                    "Date: %{x|%d %b %Y}<br>"
                    "Close: %{y:.2f}<br>"
                    "Volume confirmation: %{customdata[0]} (%{customdata[1]})<br>"
                    "Trend confirmation: %{customdata[2]}<br>"
                    "Prior BUY-SELL return: %{customdata[3]}<br>"
                    "Median last 3 BUY-SELL returns: %{customdata[4]}<extra></extra>"
                ),
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
                text=[_sell_quality_text(row) for _, row in sell_rows.iterrows()],
                textposition="top center",
                textfont={"color": "#be123c", "size": 12, "family": "Arial, sans-serif"},
                marker={
                    "symbol": "triangle-down",
                    "size": 24,
                    "color": "#ff0055",
                    "line": {"color": "#6f0027", "width": 3},
                },
                customdata=_quality_customdata(sell_rows),
                hovertemplate=(
                    "Date: %{x|%d %b %Y}<br>"
                    "Close: %{y:.2f}<br>"
                    "Completed BUY-SELL return: %{customdata[5]}<extra></extra>"
                ),
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
        yaxis2={
            "title": "Price",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "tickformat": ".2f",
            "matches": "y",
        },
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 50, "r": 28, "t": 92, "b": 44},
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


def build_gtt_opportunity_chart(stock_stats: pd.DataFrame, height: int = 540) -> str:
    if stock_stats.empty:
        return ""

    frame = stock_stats.copy()
    numeric_columns = [
        "valid_pairs",
        "hit_10pct_rate_pct",
        "median_max_gain_pct",
        "avg_max_gain_pct",
        "best_max_gain_pct",
        "median_days_to_peak",
        "suggested_conservative_gtt_pct",
        "suggested_moderate_gtt_pct",
    ]
    for column in numeric_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame[
        frame["valid_pairs"].fillna(0).gt(0)
        & frame["hit_10pct_rate_pct"].notna()
        & frame["median_max_gain_pct"].notna()
    ].copy()
    if frame.empty:
        return ""

    median_gain_score = frame["median_max_gain_pct"].clip(lower=0, upper=50) / 50 * 100
    sample_score = frame["valid_pairs"].clip(lower=0, upper=10) / 10 * 100
    speed_score = (180 - frame["median_days_to_peak"].clip(lower=0, upper=180)) / 180 * 100
    speed_score = speed_score.fillna(0)
    frame["opportunity_score"] = (
        frame["hit_10pct_rate_pct"].clip(lower=0, upper=100) * 0.4
        + median_gain_score * 0.3
        + sample_score * 0.2
        + speed_score * 0.1
    )
    frame = frame.sort_values("opportunity_score", ascending=False).copy()
    frame["peak_speed_bucket"] = frame.get("peak_speed_bucket", pd.Series("NA", index=frame.index))
    frame["peak_speed_bucket"] = frame.apply(
        lambda row: row["peak_speed_bucket"]
        if pd.notna(row.get("peak_speed_bucket"))
        and str(row.get("peak_speed_bucket")).strip().upper() not in {"", "NA", "NAN", "NONE"}
        else _peak_speed_bucket(row.get("median_days_to_peak")),
        axis=1,
    )

    bucket_colors = {
        "Within 30 days": "#15866f",
        "31-60 days": "#4f8ecf",
        "61-90 days": "#d5a84b",
        "91-180 days": "#c77d48",
        "181-365 days": "#c65a62",
        "Over 1 year": "#8a607d",
        "NA": "#94a3b8",
    }
    bucket_order = ["Within 30 days", "31-60 days", "61-90 days", "91-180 days", "181-365 days", "Over 1 year", "NA"]
    frame["peak_speed_bucket"] = frame["peak_speed_bucket"].where(
        frame["peak_speed_bucket"].isin(bucket_order),
        "NA",
    )

    bucket_rows: list[dict[str, Any]] = []
    for bucket in bucket_order:
        rows = frame[frame["peak_speed_bucket"] == bucket]
        if rows.empty:
            bucket_rows.append(
                {
                    "bucket": bucket,
                    "stock_count": 0,
                    "median_hit_10pct": pd.NA,
                    "median_max_gain_pct": pd.NA,
                    "median_days_to_peak": pd.NA,
                    "median_valid_pairs": pd.NA,
                    "top_symbols": "",
                    "color": bucket_colors[bucket],
                }
            )
            continue

        top_symbols = rows.sort_values("opportunity_score", ascending=False).head(12)["symbol"].astype(str).tolist()
        bucket_rows.append(
            {
                "bucket": bucket,
                "stock_count": len(rows),
                "median_hit_10pct": rows["hit_10pct_rate_pct"].median(),
                "median_max_gain_pct": rows["median_max_gain_pct"].median(),
                "median_days_to_peak": rows["median_days_to_peak"].median(),
                "median_valid_pairs": rows["valid_pairs"].median(),
                "top_symbols": ", ".join(top_symbols),
                "color": bucket_colors[bucket],
            }
        )

    bucket_frame = pd.DataFrame(bucket_rows)
    customdata = [
        [
            _format_float(row.get("median_hit_10pct"), 2, "%"),
            _format_float(row.get("median_max_gain_pct"), 2, "%"),
            _format_float(row.get("median_days_to_peak"), 0),
            _format_float(row.get("median_valid_pairs"), 1),
            row.get("top_symbols") or "NA",
        ]
        for _, row in bucket_frame.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bucket_frame["bucket"],
            y=bucket_frame["stock_count"],
            text=bucket_frame["stock_count"],
            textposition="outside",
            marker={"color": bucket_frame["color"], "line": {"color": "#ffffff", "width": 1.5}},
            customdata=customdata,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Stocks: %{y}<br>"
                "Median hit 10%: %{customdata[0]}<br>"
                "Median max gain: %{customdata[1]}<br>"
                "Median days to peak: %{customdata[2]}<br>"
                "Median valid pairs: %{customdata[3]}<br>"
                "Top symbols: %{customdata[4]}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="GTT Peak Speed Buckets: Fresh Weekly BUY + EMA Trend",
        xaxis_title="Median BUY-to-peak time bucket",
        yaxis_title="Stocks in bucket",
        hovermode="x",
        height=height,
        margin={"l": 58, "r": 28, "t": 82, "b": 86},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=False,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(217, 225, 234, 0.8)",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(217, 225, 234, 0.8)",
        zeroline=False,
        rangemode="tozero",
    )

    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displaylogo": False, "responsive": True})
    return f'<div class="opportunity-chart-frame">{chart_html}</div>'


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
