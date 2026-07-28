from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from uuid import uuid4

from stock_screener.adx_di_study import calculate_adx_di
from stock_screener.data.storage import Storage
from stock_screener.resample import resample_daily_to_weekly


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


def build_rotation_group_chart(
    data_root: Path,
    config: dict[str, Any],
    group_id: str,
    group_members: pd.DataFrame,
    height: int = 460,
    lookback_weeks: int = 78,
) -> str:
    if group_members.empty:
        return ""

    storage = Storage(data_root)
    strategy_cfg = config.get("strategy", {})
    weekly_anchor = strategy_cfg.get("weekly_anchor", "W-FRI")
    use_completed_weeks_only = bool(strategy_cfg.get("use_completed_weeks_only", True))
    fig = go.Figure()
    member_rows = group_members.copy()
    member_rows["symbol"] = member_rows["symbol"].astype(str)
    member_rows["movement_status"] = member_rows.get("movement_status", "").astype(str)

    status_colors = {
        "Leader": "#11825f",
        "Catch-up Candidate": "#d98f1d",
        "Lagging": "#7a5af8",
        "In Sync": "#4f8ecf",
    }
    status_rank = {"Leader": 3, "Catch-up Candidate": 2, "In Sync": 1, "Lagging": 0}
    member_rows["_status_rank"] = member_rows["movement_status"].map(status_rank).fillna(-1)
    member_rows = member_rows.sort_values(["_status_rank", "recent_return_8w_pct", "symbol"], ascending=[False, False, True])

    plotted = 0
    for _, row in member_rows.iterrows():
        symbol = str(row.get("symbol", "")).upper()
        exchange = str(row.get("exchange", "NSE")).upper()
        if not symbol:
            continue
        daily = storage.load_candles(exchange, symbol, "1D")
        if daily.empty:
            continue
        weekly = resample_daily_to_weekly(daily, weekly_anchor, use_completed_weeks_only)
        if weekly.empty or "close" not in weekly.columns:
            continue
        weekly = weekly.copy()
        weekly["date"] = pd.to_datetime(weekly["date"], errors="coerce")
        weekly = weekly.sort_values("date").dropna(subset=["date"]).tail(lookback_weeks)
        closes = pd.to_numeric(weekly["close"], errors="coerce")
        valid = closes.dropna()
        if valid.empty:
            continue
        base = float(valid.iloc[0])
        if base == 0:
            continue
        normalized = (closes / base) * 100.0
        status = str(row.get("movement_status", "In Sync"))
        latest_signal = str(row.get("latest_week_signal", "NONE"))
        latest_signal_date = pd.to_datetime(row.get("latest_week_signal_date"), errors="coerce")
        latest_signal_is_fresh = str(row.get("latest_week_signal_is_fresh", "")).strip().lower() in {"1", "true", "yes", "y"}
        line_color = status_colors.get(status, "#64748b")
        line_dash = "solid" if latest_signal_is_fresh else "dot"
        hover_name = str(row.get("name", symbol))
        fig.add_trace(
            go.Scatter(
                x=weekly["date"],
                y=normalized,
                mode="lines",
                name=f"{symbol} · {status}",
                line={"color": line_color, "width": 2.6 if status in {"Leader", "Catch-up Candidate"} else 1.8, "dash": line_dash},
                hovertemplate=(
                    f"<b>{symbol}</b> · {hover_name}<br>"
                    f"Status: {status}<br>"
                    f"Latest weekly signal: {latest_signal}<br>"
                    f"Recent 8W return: {_format_float(row.get('recent_return_8w_pct'), 2, '%')}<br>"
                    f"Group 8W return: {_format_float(row.get('group_return_8w_pct'), 2, '%')}<br>"
                    f"Catch-up gap: {_format_float(row.get('catch_up_gap_8w_pct'), 2, '%')}<br>"
                    "Date: %{x|%d %b %Y}<br>"
                    "Normalized price: %{y:.2f}<extra></extra>"
                ),
            )
        )
        if latest_signal_is_fresh and latest_signal in {"BUY", "SELL"} and pd.notna(latest_signal_date):
            weekly_lookup = pd.Series(normalized.values, index=weekly["date"])
            if latest_signal_date in weekly_lookup.index:
                signal_value = float(weekly_lookup.loc[latest_signal_date])
                fig.add_trace(
                    go.Scatter(
                        x=[latest_signal_date],
                        y=[signal_value],
                        mode="markers+text",
                        name=f"{symbol} {latest_signal}",
                        text=[latest_signal],
                        textposition="top center" if latest_signal == "SELL" else "bottom center",
                        marker={
                            "symbol": "triangle-up" if latest_signal == "BUY" else "triangle-down",
                            "size": 14,
                            "color": "#00b879" if latest_signal == "BUY" else "#ff0055",
                            "line": {"color": "#073b30" if latest_signal == "BUY" else "#6f0027", "width": 1.5},
                        },
                        showlegend=False,
                        hovertemplate=f"{symbol} {latest_signal}<br>%{{x|%d %b %Y}}<br>Normalized price: %{{y:.2f}}<extra></extra>",
                    )
                )
        plotted += 1

    if plotted == 0:
        return ""

    fig.update_layout(
        title=f"Rotation Group {group_id}: normalized price trends and weekly signal state",
        xaxis_title="Date",
        yaxis_title="Normalized Price (Start = 100)",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 52, "r": 24, "t": 78, "b": 52},
        height=height,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(217, 225, 234, 0.8)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(217, 225, 234, 0.8)")
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displaylogo": False, "responsive": True})
    return f'<div class="opportunity-chart-frame">{chart_html}</div>'


def build_adx_di_chart(
    daily_frame: pd.DataFrame,
    exchange: str,
    symbol: str,
    *,
    length: int = 14,
    threshold: float = 20.0,
    bars: int = 140,
    height: int = 760,
) -> str:
    frame = calculate_adx_di(daily_frame, length=length, threshold=threshold)
    if frame.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{exchange}:{symbol} - No daily candle data available")
        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    frame = frame.tail(max(int(bars), 40)).copy()
    di_plus_cross_rows = frame[frame["di_plus_crossed_above_di_minus"].fillna(False)].copy()
    cross_rows = frame[frame["adx_bullish_cross_above_di_minus"].fillna(False)].copy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": True}], [{}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=frame["date"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="Price",
            increasing_line_color="#20c997",
            decreasing_line_color="#ff4d6d",
            increasing_fillcolor="#20c997",
            decreasing_fillcolor="#ff4d6d",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    volume_colors = [
        "rgba(32, 201, 151, 0.48)" if close_value >= open_value else "rgba(255, 77, 109, 0.48)"
        for close_value, open_value in zip(frame["close"], frame["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=frame["date"],
            y=frame["volume"],
            name="Volume",
            marker_color=volume_colors,
            opacity=0.55,
            hovertemplate="Date: %{x|%d %b %Y}<br>Volume: %{y:.0f}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["di_plus"],
            mode="lines",
            name="DI+",
            line={"color": "#39d353", "width": 2},
            hovertemplate="Date: %{x|%d %b %Y}<br>DI+: %{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["di_minus"],
            mode="lines",
            name="DI-",
            line={"color": "#ff4d4f", "width": 2},
            hovertemplate="Date: %{x|%d %b %Y}<br>DI-: %{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["adx"],
            mode="lines",
            name="ADX",
            line={"color": "#f8fafc", "width": 2.3},
            hovertemplate="Date: %{x|%d %b %Y}<br>ADX: %{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["threshold"],
            mode="lines",
            name=f"Threshold {float(threshold):.0f}",
            line={"color": "#facc15", "width": 1.4, "dash": "dot"},
            hovertemplate="Date: %{x|%d %b %Y}<br>Threshold: %{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    if not di_plus_cross_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=di_plus_cross_rows["date"],
                y=di_plus_cross_rows["di_plus"],
                mode="markers",
                name="DI+ crossed above DI-",
                marker={"size": 9, "color": "#39d353", "line": {"color": "#052e16", "width": 1.3}},
                hovertemplate=(
                    "Date: %{x|%d %b %Y}<br>"
                    "DI+ crossed above DI-<br>"
                    "DI+: %{y:.2f}<br>"
                    "DI-: %{customdata[0]:.2f}<br>"
                    "ADX: %{customdata[1]:.2f}<extra></extra>"
                ),
                customdata=di_plus_cross_rows[["di_minus", "adx"]].to_numpy(),
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        title=f"{exchange}:{symbol} ADX / DI Daily Chart",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 54, "r": 30, "t": 88, "b": 40},
        height=height,
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font={"color": "#e5e7eb"},
        xaxis_rangeslider_visible=False,
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.14)",
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Price",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.14)",
        zeroline=False,
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Volume",
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.update_yaxes(
        title_text="ADX / DI",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.14)",
        zeroline=False,
        row=2,
        col=1,
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displaylogo": False, "responsive": True})


def build_sector_mix_pie_chart(
    sector_summary: pd.DataFrame,
    *,
    title: str = "Sector Mix",
    height: int = 420,
) -> str:
    if sector_summary.empty or "sector_label" not in sector_summary.columns or "stock_count" not in sector_summary.columns:
        return ""

    labels = sector_summary["sector_label"].astype(str).tolist()
    values = pd.to_numeric(sector_summary["stock_count"], errors="coerce").fillna(0).tolist()
    if not any(value > 0 for value in values):
        return ""

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.38,
                sort=False,
                textinfo="label+percent",
                hovertemplate="%{label}<br>Stocks: %{value}<br>Share: %{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=height,
        margin={"l": 20, "r": 20, "t": 64, "b": 20},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.16, "xanchor": "left", "x": 0},
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displaylogo": False, "responsive": True})


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
