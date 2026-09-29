from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage


AGENT_COLUMNS: tuple[str, ...] = (
    "agent_rank",
    "agent_grade",
    "agent_action",
    "agent_total_score",
    "setup_agent_score",
    "confirmation_agent_score",
    "trend_agent_score",
    "risk_reward_agent_score",
    "backtest_agent_score",
    "liquidity_agent_score",
    "operator_activity_agent_score",
    "operator_activity_score",
    "operator_activity_label",
    "operator_activity_rationale",
    "operator_activity_warning",
    "operator_accumulation_days_20",
    "operator_absorption_days_20",
    "operator_distribution_days_20",
    "operator_high_volume_tight_range_days_20",
    "operator_volume_ratio_20",
    "operator_price_change_20d_pct",
    "operator_obv_delta_20d_pct_volume",
    "operator_cmf20",
    "operator_effort_vs_result_ratio",
    "agent_rationale",
    "agent_risks",
    "agent_entry_rule",
    "agent_exit_rule",
    "agent_position_sizing",
    "agent_reward_risk_ratio",
    "agent_remaining_upside_pct",
    "agent_downside_to_stop_pct",
    "agent_return_from_entry_pct",
    "agent_median_turnover_20d_cr",
    "agent_atr14_pct",
    "agent_latest_close_vs_sma50_pct",
    "agent_latest_close_vs_sma200_pct",
    "agent_sma50_slope_20d_pct",
    "agent_sma200_slope_20d_pct",
    "agent_historical_strategy",
    "agent_historical_trades",
    "agent_historical_win_rate_pct",
    "agent_historical_median_return_pct",
    "agent_historical_avg_return_pct",
    "agent_historical_profit_factor",
    "agent_similar_trades",
    "agent_similar_win_rate_pct",
    "agent_similar_median_return_pct",
    "agent_similar_avg_return_pct",
)


def enrich_ema20_candidates_with_research_agents(
    candidates: pd.DataFrame,
    *,
    stock_stats: pd.DataFrame,
    trades: pd.DataFrame,
    storage: Storage,
    as_of_date: Any | None = None,
) -> pd.DataFrame:
    """Rank EMA20 IB/EB candidates with deterministic research-agent scores."""
    if candidates.empty:
        return candidates.copy()

    working = candidates.drop(columns=list(AGENT_COLUMNS), errors="ignore").copy()
    stock_stats_prepared = _prepare_stock_stats(stock_stats)
    trades_prepared = _prepare_trades(trades)
    default_as_of_ts = _coerce_ts(as_of_date)

    rows: list[dict[str, Any]] = []
    for _, row in working.iterrows():
        exchange = _text(row.get("exchange") or "NSE").upper() or "NSE"
        symbol = _text(row.get("symbol")).upper()
        daily = _prepare_daily(storage.load_candles(exchange, symbol, "1D"))
        row_as_of_ts = default_as_of_ts or _coerce_ts(row.get("as_of_date"))
        if row_as_of_ts is not None and not daily.empty:
            daily = daily[daily["date"].dt.normalize() <= row_as_of_ts].copy()

        strategy = _candidate_strategy(row)
        historical = _historical_edge(row, strategy, stock_stats_prepared, trades_prepared)
        similar = _similar_context_edge(row, strategy, trades_prepared)
        market = _market_context(daily)
        operator = _operator_activity_context(daily)
        scores = _score_candidate(row, historical, similar, market)
        scores.update(_operator_score(operator))
        plan = _trade_plan(row, market)
        decision = _decision(scores, row, historical, plan, operator)

        enriched = row.to_dict()
        enriched.update(scores)
        enriched.update(historical)
        enriched.update(similar)
        enriched.update(market)
        enriched.update(operator)
        enriched.update(plan)
        enriched.update(decision)
        rows.append(enriched)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(
        ["agent_total_score", "confirmation_count", "candidate_score", "sessions_since_signal", "symbol"],
        ascending=[False, False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    result["agent_rank"] = range(1, len(result) + 1)
    return result


def _score_candidate(
    row: pd.Series,
    historical: dict[str, Any],
    similar: dict[str, Any],
    market: dict[str, Any],
) -> dict[str, Any]:
    setup_score = _clamp(_num(row.get("candidate_score"), 0.0), 0.0, 100.0)
    confirmation_score = _confirmation_score(row)
    trend_score = _trend_score(row, market)
    risk_score = _risk_reward_score(row)
    backtest_score = _backtest_score(historical, similar)
    liquidity_score = _liquidity_score(market)

    total = (
        setup_score * 0.20
        + confirmation_score * 0.17
        + trend_score * 0.14
        + risk_score * 0.19
        + backtest_score * 0.17
        + liquidity_score * 0.06
    )
    return {
        "agent_total_score": round(_clamp(total, 0.0, 100.0), 2),
        "setup_agent_score": round(setup_score, 2),
        "confirmation_agent_score": round(confirmation_score, 2),
        "trend_agent_score": round(trend_score, 2),
        "risk_reward_agent_score": round(risk_score, 2),
        "backtest_agent_score": round(backtest_score, 2),
        "liquidity_agent_score": round(liquidity_score, 2),
    }


def _decision(
    scores: dict[str, Any],
    row: pd.Series,
    historical: dict[str, Any],
    plan: dict[str, Any],
    operator: dict[str, Any],
) -> dict[str, Any]:
    operator_score = _num(scores.get("operator_activity_agent_score"), 0.0)
    total = _clamp(_num(scores.get("agent_total_score"), 0.0) + operator_score * 0.07, 0.0, 100.0)
    scores["agent_total_score"] = round(total, 2)
    risk_score = _num(scores.get("risk_reward_agent_score"), 0.0)
    backtest_score = _num(scores.get("backtest_agent_score"), 0.0)
    confirmation_score = _num(scores.get("confirmation_agent_score"), 0.0)
    reward_risk = _num(plan.get("agent_reward_risk_ratio"), np.nan)
    active_risk = str(plan.get("agent_plan_status") or "")
    weekly_sell = _truthy(row.get("weekly_sell_signal"))
    trades = int(_num(historical.get("agent_historical_trades"), 0.0))

    if active_risk == "invalidated":
        grade = "Avoid"
        action = "Avoid: setup invalidated by price"
    elif active_risk == "extended":
        grade = "Wait"
        action = "Wait: entry is extended versus target/stop"
    elif weekly_sell:
        grade = "Conflict"
        action = "Avoid or size very small: weekly SELL conflict"
    elif total >= 78 and risk_score >= 60 and backtest_score >= 55 and confirmation_score >= 55 and reward_risk >= 1.4:
        grade = "A+"
        action = "Top research candidate"
    elif total >= 68 and risk_score >= 50 and backtest_score >= 45 and reward_risk >= 1.2:
        grade = "A"
        action = "Research candidate"
    elif total >= 58 and risk_score >= 40:
        grade = "B"
        action = "Watchlist: needs confirmation"
    else:
        grade = "C"
        action = "Low priority / wait"

    rationale = _build_rationale(row, scores, historical, plan, operator)
    risks = _build_risks(row, historical, plan, trades, operator)
    return {
        "agent_grade": grade,
        "agent_action": action,
        "agent_rationale": rationale,
        "agent_risks": risks,
    }


def _confirmation_score(row: pd.Series) -> float:
    score = 35.0
    if _truthy(row.get("weekly_buy_signal")):
        score += 18.0
    if _truthy(row.get("adx_crossover_pass")):
        score += 18.0
    if _truthy(row.get("minervini_quality_pass")):
        score += 18.0
    if _truthy(row.get("knox_envelope_pass")):
        score += 14.0
    if _truthy(row.get("weekly_sell_signal")):
        score -= 35.0
    count = _num(row.get("confirmation_count"), 0.0)
    if count <= 0:
        score -= 8.0
    return _clamp(score, 0.0, 100.0)


def _trend_score(row: pd.Series, market: dict[str, Any]) -> float:
    score = 45.0
    trend_label = str(row.get("ema_trend_20d_label") or "")
    if trend_label == "Rising":
        score += 20.0
    elif trend_label == "Flat":
        score += 10.0
    elif trend_label == "Falling":
        score -= 10.0

    sma50_slope = _num(market.get("agent_sma50_slope_20d_pct"), np.nan)
    sma200_slope = _num(market.get("agent_sma200_slope_20d_pct"), np.nan)
    close_vs_sma50 = _num(market.get("agent_latest_close_vs_sma50_pct"), np.nan)
    close_vs_sma200 = _num(market.get("agent_latest_close_vs_sma200_pct"), np.nan)
    if np.isfinite(sma50_slope):
        score += 10.0 if sma50_slope > 0 else -6.0
    if np.isfinite(sma200_slope):
        score += 8.0 if sma200_slope > 0 else -8.0
    if np.isfinite(close_vs_sma50):
        score += 6.0 if close_vs_sma50 > -5.0 else -5.0
    if np.isfinite(close_vs_sma200):
        score += 6.0 if close_vs_sma200 > 0.0 else -8.0
    return _clamp(score, 0.0, 100.0)


def _risk_reward_score(row: pd.Series) -> float:
    entry = _num(row.get("planned_entry_price"), np.nan)
    stop = _num(row.get("planned_stop_price"), np.nan)
    target = _num(row.get("planned_target_price"), np.nan)
    close = _num(row.get("as_of_close"), np.nan)
    if not all(np.isfinite(value) for value in (entry, stop, target)) or entry <= 0 or stop >= entry:
        return 10.0

    per_share_risk = entry - stop
    reward = target - entry
    reward_risk = reward / per_share_risk if per_share_risk > 0 else np.nan
    score = 35.0
    if np.isfinite(reward_risk):
        if reward_risk >= 2.0:
            score += 35.0
        elif reward_risk >= 1.5:
            score += 25.0
        elif reward_risk >= 1.2:
            score += 15.0
        elif reward_risk >= 1.0:
            score += 5.0
        else:
            score -= 20.0

    if np.isfinite(close):
        if close <= stop:
            return 0.0
        remaining_upside = (target / close - 1.0) * 100.0 if close > 0 else np.nan
        downside = (close / stop - 1.0) * 100.0 if stop > 0 else np.nan
        if np.isfinite(remaining_upside) and remaining_upside < 1.0:
            score -= 35.0
        elif np.isfinite(remaining_upside) and remaining_upside < 3.0:
            score -= 15.0
        if np.isfinite(downside) and downside < 2.0:
            score -= 12.0
    return _clamp(score, 0.0, 100.0)


def _backtest_score(historical: dict[str, Any], similar: dict[str, Any]) -> float:
    trades = _num(historical.get("agent_historical_trades"), 0.0)
    win_rate = _num(historical.get("agent_historical_win_rate_pct"), np.nan)
    median_return = _num(historical.get("agent_historical_median_return_pct"), np.nan)
    profit_factor = _num(historical.get("agent_historical_profit_factor"), np.nan)

    similar_trades = _num(similar.get("agent_similar_trades"), 0.0)
    similar_win = _num(similar.get("agent_similar_win_rate_pct"), np.nan)
    similar_median = _num(similar.get("agent_similar_median_return_pct"), np.nan)

    score = 40.0
    if trades >= 8:
        score += 12.0
    elif trades >= 4:
        score += 6.0
    elif trades <= 1:
        score -= 8.0

    if np.isfinite(win_rate):
        score += _clamp((win_rate - 35.0) * 0.55, -15.0, 18.0)
    if np.isfinite(median_return):
        score += _clamp(median_return * 4.0, -18.0, 18.0)
    if np.isfinite(profit_factor):
        if profit_factor >= 1.8:
            score += 18.0
        elif profit_factor >= 1.3:
            score += 10.0
        elif profit_factor < 1.0:
            score -= 15.0

    if similar_trades >= 30:
        if np.isfinite(similar_win):
            score += _clamp((similar_win - 35.0) * 0.25, -8.0, 10.0)
        if np.isfinite(similar_median):
            score += _clamp(similar_median * 2.0, -8.0, 10.0)
    return _clamp(score, 0.0, 100.0)


def _liquidity_score(market: dict[str, Any]) -> float:
    turnover = _num(market.get("agent_median_turnover_20d_cr"), np.nan)
    if not np.isfinite(turnover):
        return 35.0
    if turnover >= 25.0:
        return 95.0
    if turnover >= 10.0:
        return 85.0
    if turnover >= 5.0:
        return 72.0
    if turnover >= 1.0:
        return 55.0
    if turnover >= 0.25:
        return 35.0
    return 15.0


def _operator_score(operator: dict[str, Any]) -> dict[str, Any]:
    score = _num(operator.get("operator_activity_score"), 0.0)
    return {"operator_activity_agent_score": round(_clamp(score, 0.0, 100.0), 2)}


def _trade_plan(row: pd.Series, market: dict[str, Any]) -> dict[str, Any]:
    entry = _num(row.get("planned_entry_price"), np.nan)
    stop = _num(row.get("planned_stop_price"), np.nan)
    target = _num(row.get("planned_target_price"), np.nan)
    close = _num(row.get("as_of_close"), np.nan)
    signal_high = _num(row.get("high"), np.nan)
    signal_low = _num(row.get("low"), np.nan)
    atr_pct = _num(market.get("agent_atr14_pct"), np.nan)

    reward_risk = np.nan
    if all(np.isfinite(value) for value in (entry, stop, target)) and entry > stop:
        reward_risk = (target - entry) / (entry - stop)

    remaining_upside = (target / close - 1.0) * 100.0 if np.isfinite(target) and np.isfinite(close) and close > 0 else np.nan
    downside_to_stop = (close / stop - 1.0) * 100.0 if np.isfinite(stop) and np.isfinite(close) and stop > 0 else np.nan
    return_from_entry = (close / entry - 1.0) * 100.0 if np.isfinite(close) and np.isfinite(entry) and entry > 0 else np.nan

    if np.isfinite(close) and np.isfinite(stop) and close <= stop:
        status = "invalidated"
    elif np.isfinite(remaining_upside) and remaining_upside < 1.0:
        status = "extended"
    else:
        status = "valid"

    breakout_level = signal_high * 1.001 if np.isfinite(signal_high) else np.nan
    if str(row.get("entry_status") or "").lower().startswith("entry already"):
        entry_rule = "Entry already triggered by the backtest rule; fresh entry only on controlled pullback or reclaim above signal high."
    elif np.isfinite(breakout_level):
        entry_rule = f"Buy only if price trades above signal high plus buffer near {breakout_level:.2f}; skip if it gaps more than 1 ATR."
    else:
        entry_rule = "Buy only on next-session confirmation; skip if the entry price is far above the planned stop."

    atr_stop_text = ""
    if np.isfinite(atr_pct):
        atr_stop_text = f" ATR14 is {atr_pct:.1f}%."
    exit_rule = (
        "Initial stop is the tighter of fixed risk and pattern-low buffer; first target is the configured profit target; "
        "also exit on EMA-close reclaim or max-hold per the backtest."
        + atr_stop_text
    )
    sizing = "Risk a fixed fraction of capital per trade, e.g. 0.5%-1.0%; shares = capital risk / (entry - stop)."

    return {
        "agent_plan_status": status,
        "agent_reward_risk_ratio": round(float(reward_risk), 2) if np.isfinite(reward_risk) else np.nan,
        "agent_remaining_upside_pct": round(float(remaining_upside), 2) if np.isfinite(remaining_upside) else np.nan,
        "agent_downside_to_stop_pct": round(float(downside_to_stop), 2) if np.isfinite(downside_to_stop) else np.nan,
        "agent_return_from_entry_pct": round(float(return_from_entry), 2) if np.isfinite(return_from_entry) else np.nan,
        "agent_entry_rule": entry_rule,
        "agent_exit_rule": exit_rule,
        "agent_position_sizing": sizing,
        "agent_signal_high": round(float(signal_high), 2) if np.isfinite(signal_high) else np.nan,
        "agent_signal_low": round(float(signal_low), 2) if np.isfinite(signal_low) else np.nan,
    }


def _historical_edge(
    row: pd.Series,
    strategy: str,
    stock_stats: pd.DataFrame,
    trades: pd.DataFrame,
) -> dict[str, Any]:
    exchange = _text(row.get("exchange") or "NSE").upper()
    symbol = _text(row.get("symbol")).upper()
    chosen = pd.Series(dtype="object")
    if not stock_stats.empty:
        exact = stock_stats[
            stock_stats["exchange"].eq(exchange)
            & stock_stats["symbol"].eq(symbol)
            & stock_stats["strategy"].eq(strategy)
        ]
        if exact.empty:
            exact = stock_stats[
                stock_stats["exchange"].eq(exchange)
                & stock_stats["symbol"].eq(symbol)
                & stock_stats["strategy"].eq("Any EMA20 Band Bullish Pattern")
            ]
        if not exact.empty:
            chosen = exact.sort_values(["trades", "profit_factor", "median_return_pct"], ascending=[False, False, False], na_position="last").iloc[0]

    if chosen.empty and not trades.empty:
        sample = trades[trades["strategy"].eq(strategy)].copy()
        if sample.empty:
            sample = trades[trades["strategy"].eq("Any EMA20 Band Bullish Pattern")].copy()
        chosen = _stats_from_trades(sample, strategy)

    return {
        "agent_historical_strategy": _text(chosen.get("strategy") if not chosen.empty else strategy) or strategy,
        "agent_historical_trades": int(_num(chosen.get("trades") if not chosen.empty else 0, 0.0)),
        "agent_historical_win_rate_pct": _clean_float(chosen.get("win_rate_pct") if not chosen.empty else np.nan),
        "agent_historical_median_return_pct": _clean_float(chosen.get("median_return_pct") if not chosen.empty else np.nan),
        "agent_historical_avg_return_pct": _clean_float(chosen.get("avg_return_pct") if not chosen.empty else np.nan),
        "agent_historical_profit_factor": _clean_float(chosen.get("profit_factor") if not chosen.empty else np.nan),
    }


def _similar_context_edge(row: pd.Series, strategy: str, trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "agent_similar_trades": 0,
            "agent_similar_win_rate_pct": np.nan,
            "agent_similar_median_return_pct": np.nan,
            "agent_similar_avg_return_pct": np.nan,
        }
    sample = trades[trades["strategy"].eq(strategy)].copy()
    if sample.empty:
        sample = trades.copy()
    if "pattern" in sample.columns and _text(row.get("pattern")):
        pattern_sample = sample[sample["pattern"].astype(str).eq(_text(row.get("pattern")))]
        if len(pattern_sample) >= 20:
            sample = pattern_sample
    if "ema_trend_20d_label" in sample.columns and _text(row.get("ema_trend_20d_label")):
        trend_sample = sample[sample["ema_trend_20d_label"].astype(str).eq(_text(row.get("ema_trend_20d_label")))]
        if len(trend_sample) >= 20:
            sample = trend_sample
    candidate_bucket = _distance_bucket(row.get("distance_to_ema_close_pct"))
    if "distance_to_ema_close_pct" in sample.columns and candidate_bucket:
        buckets = sample["distance_to_ema_close_pct"].map(_distance_bucket)
        bucket_sample = sample[buckets.eq(candidate_bucket)]
        if len(bucket_sample) >= 20:
            sample = bucket_sample
    stats = _stats_from_trades(sample, strategy)
    return {
        "agent_similar_trades": int(_num(stats.get("trades"), 0.0)),
        "agent_similar_win_rate_pct": _clean_float(stats.get("win_rate_pct")),
        "agent_similar_median_return_pct": _clean_float(stats.get("median_return_pct")),
        "agent_similar_avg_return_pct": _clean_float(stats.get("avg_return_pct")),
    }


def _market_context(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty or len(daily) < 20:
        return {
            "agent_median_turnover_20d_cr": np.nan,
            "agent_atr14_pct": np.nan,
            "agent_latest_close_vs_sma50_pct": np.nan,
            "agent_latest_close_vs_sma200_pct": np.nan,
            "agent_sma50_slope_20d_pct": np.nan,
            "agent_sma200_slope_20d_pct": np.nan,
        }
    frame = daily.copy().reset_index(drop=True)
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    volume = pd.to_numeric(frame.get("volume", pd.Series(np.nan, index=frame.index)), errors="coerce")
    latest_close = _num(close.iloc[-1], np.nan)
    turnover = (close * volume).tail(20).median() / 10_000_000.0

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.rolling(14, min_periods=5).mean().iloc[-1]
    atr_pct = atr14 / latest_close * 100.0 if np.isfinite(atr14) and np.isfinite(latest_close) and latest_close > 0 else np.nan

    sma50 = close.rolling(50, min_periods=20).mean()
    sma200 = close.rolling(200, min_periods=80).mean()
    close_vs_sma50 = _distance_pct(latest_close, sma50.iloc[-1] if len(sma50) else np.nan)
    close_vs_sma200 = _distance_pct(latest_close, sma200.iloc[-1] if len(sma200) else np.nan)
    sma50_slope = _distance_pct(sma50.iloc[-1], sma50.iloc[-21]) if len(sma50) > 21 else np.nan
    sma200_slope = _distance_pct(sma200.iloc[-1], sma200.iloc[-21]) if len(sma200) > 21 else np.nan
    return {
        "agent_median_turnover_20d_cr": _clean_float(turnover),
        "agent_atr14_pct": _clean_float(atr_pct),
        "agent_latest_close_vs_sma50_pct": _clean_float(close_vs_sma50),
        "agent_latest_close_vs_sma200_pct": _clean_float(close_vs_sma200),
        "agent_sma50_slope_20d_pct": _clean_float(sma50_slope),
        "agent_sma200_slope_20d_pct": _clean_float(sma200_slope),
    }


def _operator_activity_context(daily: pd.DataFrame) -> dict[str, Any]:
    empty = {
        "operator_activity_score": np.nan,
        "operator_activity_label": "Insufficient data",
        "operator_activity_rationale": "",
        "operator_activity_warning": "Daily OHLCV cannot prove manipulation or intent; use this only as a footprint screen.",
        "operator_accumulation_days_20": 0,
        "operator_absorption_days_20": 0,
        "operator_distribution_days_20": 0,
        "operator_high_volume_tight_range_days_20": 0,
        "operator_volume_ratio_20": np.nan,
        "operator_price_change_20d_pct": np.nan,
        "operator_obv_delta_20d_pct_volume": np.nan,
        "operator_cmf20": np.nan,
        "operator_effort_vs_result_ratio": np.nan,
    }
    if daily.empty or len(daily) < 45:
        return empty

    frame = daily.copy().reset_index(drop=True)
    close = pd.to_numeric(frame["close"], errors="coerce")
    open_ = pd.to_numeric(frame.get("open", close), errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    volume = pd.to_numeric(frame.get("volume", pd.Series(np.nan, index=frame.index)), errors="coerce")
    if close.tail(25).isna().all() or volume.tail(25).isna().all():
        return empty

    previous_close = close.shift(1)
    returns_pct = close.pct_change() * 100.0
    candle_range = (high - low).replace(0, np.nan)
    range_pct = candle_range / close.replace(0, np.nan) * 100.0
    close_location = (close - low) / candle_range * 100.0
    lower_wick_pct = (pd.concat([open_, close], axis=1).min(axis=1) - low) / candle_range * 100.0

    volume_median_20 = volume.rolling(20, min_periods=10).median().replace(0, np.nan)
    volume_median_60 = volume.rolling(60, min_periods=20).median().replace(0, np.nan)
    volume_ratio = volume / volume_median_20
    recent = frame.tail(20).index
    range_baseline = range_pct.tail(60).median()
    if not np.isfinite(range_baseline) or range_baseline <= 0:
        range_baseline = range_pct.tail(20).median()

    high_volume = volume_ratio >= 1.5
    quiet_price = returns_pct.abs() <= 1.25
    tight_range = range_pct <= max(float(range_baseline) * 0.85, 0.35) if np.isfinite(range_baseline) else pd.Series(False, index=frame.index)
    accumulation_day = high_volume & quiet_price & (close_location >= 50.0)
    absorption_day = high_volume & (low < previous_close) & (close_location >= 60.0) & (returns_pct >= -1.5) & (lower_wick_pct >= 30.0)
    distribution_day = high_volume & (close_location <= 35.0) & (returns_pct <= 0.0)
    high_volume_tight_day = high_volume & tight_range

    accumulation_days = int(accumulation_day.loc[recent].fillna(False).sum())
    absorption_days = int(absorption_day.loc[recent].fillna(False).sum())
    distribution_days = int(distribution_day.loc[recent].fillna(False).sum())
    high_volume_tight_days = int(high_volume_tight_day.loc[recent].fillna(False).sum())

    latest_close = _num(close.iloc[-1], np.nan)
    close_20_ago = _num(close.iloc[-21], np.nan) if len(close) > 21 else np.nan
    price_change_20d = _distance_pct(latest_close, close_20_ago)
    recent_volume_median = _num(volume.tail(20).median(), np.nan)
    older_volume_median = _num(volume.tail(60).median(), np.nan)
    volume_ratio_20 = recent_volume_median / older_volume_median if np.isfinite(recent_volume_median) and np.isfinite(older_volume_median) and older_volume_median > 0 else np.nan

    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume.fillna(0.0)).cumsum()
    obv_delta = _num(obv.iloc[-1] - obv.iloc[-21], np.nan) if len(obv) > 21 else np.nan
    volume_sum_20 = _num(volume.tail(20).sum(), np.nan)
    obv_delta_pct_volume = obv_delta / volume_sum_20 * 100.0 if np.isfinite(obv_delta) and np.isfinite(volume_sum_20) and volume_sum_20 > 0 else np.nan

    money_flow_multiplier = ((close - low) - (high - close)) / candle_range
    money_flow_volume = money_flow_multiplier.replace([np.inf, -np.inf], np.nan) * volume
    cmf_denominator = _num(volume.tail(20).sum(), np.nan)
    cmf20 = _num(money_flow_volume.tail(20).sum(), np.nan) / cmf_denominator if np.isfinite(cmf_denominator) and cmf_denominator > 0 else np.nan
    effort_vs_result = volume_ratio_20 / max(abs(price_change_20d), 1.0) if np.isfinite(volume_ratio_20) and np.isfinite(price_change_20d) else np.nan

    score = 25.0
    if accumulation_days >= 5:
        score += 20.0
    elif accumulation_days >= 3:
        score += 12.0
    elif accumulation_days >= 1:
        score += 5.0

    if absorption_days >= 3:
        score += 18.0
    elif absorption_days >= 2:
        score += 10.0
    elif absorption_days >= 1:
        score += 5.0

    if high_volume_tight_days >= 4:
        score += 16.0
    elif high_volume_tight_days >= 2:
        score += 8.0

    if np.isfinite(obv_delta_pct_volume):
        if obv_delta_pct_volume >= 20.0 and np.isfinite(price_change_20d) and -6.0 <= price_change_20d <= 8.0:
            score += 18.0
        elif obv_delta_pct_volume >= 10.0:
            score += 10.0
        elif obv_delta_pct_volume <= -20.0:
            score -= 12.0

    if np.isfinite(cmf20):
        if cmf20 >= 0.15:
            score += 12.0
        elif cmf20 >= 0.05:
            score += 6.0
        elif cmf20 <= -0.15:
            score -= 14.0

    if np.isfinite(volume_ratio_20) and np.isfinite(price_change_20d):
        if volume_ratio_20 >= 1.2 and abs(price_change_20d) <= 5.0:
            score += 10.0
        elif volume_ratio_20 >= 1.5 and price_change_20d < -8.0:
            score -= 15.0

    if distribution_days >= 4 and (not np.isfinite(cmf20) or cmf20 < 0.0):
        score -= 25.0
    elif distribution_days >= 3:
        score -= 12.0

    score = _clamp(score, 0.0, 100.0)
    if distribution_days >= 4 and score < 55.0:
        label = "Churn / possible distribution"
    elif score >= 75.0:
        label = "Strong stealth accumulation footprint"
    elif score >= 60.0:
        label = "Possible accumulation footprint"
    elif score >= 45.0:
        label = "Watch: mild absorption footprint"
    else:
        label = "No clear operator footprint"

    rationale_parts = [
        f"{accumulation_days} accumulation days",
        f"{absorption_days} absorption days",
        f"{high_volume_tight_days} high-volume tight-range days",
    ]
    if np.isfinite(price_change_20d):
        rationale_parts.append(f"20d price {price_change_20d:.1f}%")
    if np.isfinite(volume_ratio_20):
        rationale_parts.append(f"20d volume ratio {volume_ratio_20:.2f}x")
    if np.isfinite(obv_delta_pct_volume):
        rationale_parts.append(f"OBV delta {obv_delta_pct_volume:.1f}% of 20d volume")
    if np.isfinite(cmf20):
        rationale_parts.append(f"CMF20 {cmf20:.2f}")

    return {
        "operator_activity_score": round(score, 2),
        "operator_activity_label": label,
        "operator_activity_rationale": "; ".join(rationale_parts),
        "operator_activity_warning": "Daily OHLCV cannot prove manipulation or intent; confirm with delivery data, order book, block deals, and broader market context.",
        "operator_accumulation_days_20": accumulation_days,
        "operator_absorption_days_20": absorption_days,
        "operator_distribution_days_20": distribution_days,
        "operator_high_volume_tight_range_days_20": high_volume_tight_days,
        "operator_volume_ratio_20": _clean_float(volume_ratio_20),
        "operator_price_change_20d_pct": _clean_float(price_change_20d),
        "operator_obv_delta_20d_pct_volume": _clean_float(obv_delta_pct_volume),
        "operator_cmf20": _clean_float(cmf20),
        "operator_effort_vs_result_ratio": _clean_float(effort_vs_result),
    }


def _build_rationale(
    row: pd.Series,
    scores: dict[str, Any],
    historical: dict[str, Any],
    plan: dict[str, Any],
    operator: dict[str, Any],
) -> str:
    pieces = [
        f"{_text(row.get('pattern')) or 'EMA20 pattern'} score {_num(row.get('candidate_score'), 0.0):.1f}",
    ]
    confirmations = _text(row.get("confirmation_summary"))
    if confirmations:
        pieces.append(f"confirmed by {confirmations}")
    else:
        pieces.append("no secondary confirmation yet")
    trades = int(_num(historical.get("agent_historical_trades"), 0.0))
    if trades:
        pieces.append(
            f"historical edge {trades} trades, win {_num(historical.get('agent_historical_win_rate_pct'), 0.0):.1f}%, median {_num(historical.get('agent_historical_median_return_pct'), 0.0):.2f}%"
        )
    rr = _num(plan.get("agent_reward_risk_ratio"), np.nan)
    if np.isfinite(rr):
        pieces.append(f"planned R/R {rr:.2f}")
    operator_score = _num(operator.get("operator_activity_score"), np.nan)
    operator_label = _text(operator.get("operator_activity_label"))
    if np.isfinite(operator_score) and operator_score >= 55.0:
        pieces.append(f"operator footprint {operator_label} ({operator_score:.0f})")
    pieces.append(f"agent score {_num(scores.get('agent_total_score'), 0.0):.1f}")
    return "; ".join(pieces)


def _build_risks(
    row: pd.Series,
    historical: dict[str, Any],
    plan: dict[str, Any],
    trades: int,
    operator: dict[str, Any],
) -> str:
    risks: list[str] = []
    if _truthy(row.get("weekly_sell_signal")):
        risks.append("weekly SELL conflict")
    if trades < 5:
        risks.append("thin stock-specific backtest sample")
    if str(row.get("ema_trend_20d_label") or "") == "Falling":
        risks.append("EMA20 trend is falling")
    if str(plan.get("agent_plan_status")) == "invalidated":
        risks.append("current price is at or below planned stop")
    if str(plan.get("agent_plan_status")) == "extended":
        risks.append("little upside remains to planned target")
    downside = _num(plan.get("agent_downside_to_stop_pct"), np.nan)
    if np.isfinite(downside) and downside < 2.0:
        risks.append("stop is very close")
    operator_label = _text(operator.get("operator_activity_label"))
    if "distribution" in operator_label.lower() or "churn" in operator_label.lower():
        risks.append("operator footprint looks like churn/distribution, not clean accumulation")
    elif _num(operator.get("operator_activity_score"), 0.0) >= 55.0:
        risks.append("operator footprint is inferential, not proof of manipulation")
    return "; ".join(risks) if risks else "No major rule-based risk flag"


def _prepare_stock_stats(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    for column in ("exchange", "symbol", "strategy"):
        if column in working.columns:
            working[column] = working[column].fillna("").astype(str).str.upper().str.strip() if column != "strategy" else working[column].fillna("").astype(str).str.strip()
    return working


def _prepare_trades(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    if "strategy" in working.columns:
        working["strategy"] = working["strategy"].fillna("").astype(str).str.strip()
    return working


def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    frame = daily.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "date" not in frame.columns or "close" not in frame.columns:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
    for column in ("open", "high", "low", "close", "volume"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _stats_from_trades(frame: pd.DataFrame, strategy: str) -> pd.Series:
    if frame.empty or "net_return_pct" not in frame.columns:
        return pd.Series({"strategy": strategy, "trades": 0})
    returns = pd.to_numeric(frame["net_return_pct"], errors="coerce").dropna()
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    profit_factor = np.nan
    if len(wins) and len(losses) and losses.sum() != 0:
        profit_factor = float(wins.sum() / abs(losses.sum()))
    return pd.Series(
        {
            "strategy": strategy,
            "trades": int(len(returns)),
            "win_rate_pct": float(len(wins) / len(returns) * 100.0) if len(returns) else np.nan,
            "median_return_pct": float(returns.median()) if len(returns) else np.nan,
            "avg_return_pct": float(returns.mean()) if len(returns) else np.nan,
            "profit_factor": profit_factor,
        }
    )


def _candidate_strategy(row: pd.Series) -> str:
    pattern = _text(row.get("pattern"))
    if pattern == "Bullish Engulfing Bar":
        return "Bullish Engulfing Bar Below EMA20 Band"
    if pattern == "Bullish Inside Bar":
        return "Bullish Inside Bar Below EMA20 Band"
    return "Any EMA20 Band Bullish Pattern"


def _distance_bucket(value: Any) -> str:
    distance = _num(value, np.nan)
    if not np.isfinite(distance):
        return ""
    if distance <= -12.0:
        return "deep"
    if distance <= -6.0:
        return "medium"
    if distance <= 0.0:
        return "near"
    return "reclaimed"


def _distance_pct(numerator: Any, denominator: Any) -> float:
    top = _num(numerator, np.nan)
    bottom = _num(denominator, np.nan)
    if not np.isfinite(top) or not np.isfinite(bottom) or bottom == 0:
        return np.nan
    return (top / bottom - 1.0) * 100.0


def _coerce_ts(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _num(value: Any, default: float = np.nan) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def _clean_float(value: Any) -> float:
    numeric = _num(value, np.nan)
    return float(numeric) if np.isfinite(numeric) else np.nan


def _clamp(value: float, minimum: float, maximum: float) -> float:
    if not np.isfinite(value):
        return minimum
    return float(min(max(value, minimum), maximum))
