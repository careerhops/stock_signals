from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.weekly_buy_tracker_study import _emit_progress, _load_name_map, _to_float


DEFAULT_KNOX_LOOKBACK = 100
DEFAULT_RSI_LENGTH = 14
DEFAULT_MOMENTUM_LENGTH = 20
DEFAULT_SIGNAL_LOOKBACK = 1
DEFAULT_ENVELOPE_LENGTH = 20
DEFAULT_ENVELOPE_PERCENT = 12.0
DEFAULT_ENVELOPE_MA_TYPE = "SMA"
DEFAULT_ENVELOPE_MODE = "lower_support"
DEFAULT_ENVELOPE_PROXIMITY_PCT = 2.0
DEFAULT_SIGNAL_DIRECTION = "bullish"
DEFAULT_CMF_LENGTH = 20
DEFAULT_CMF_CONDITION = "disabled"
DEFAULT_CONFIRMATION_MODE = "strong_bullish_close"
DEFAULT_CONFIRMATION_WINDOW = 3
DEFAULT_CONFIRMATION_CLOSE_LOCATION_PCT = 80.0
DEFAULT_USE_SHARPE_FILTER = False
DEFAULT_SHARPE_LOOKBACK_DAYS = 252
DEFAULT_ANNUAL_RISK_FREE_RATE_PCT = 0.0
DEFAULT_MIN_SHARPE_RATIO: float | None = None
KNOX_ENVELOPE_LOGIC_VERSION = "completed_candles_optional_sharpe_v7"

SIGNAL_DIRECTIONS = {"bullish", "bearish", "either"}
ENVELOPE_MA_TYPES = {"SMA", "EMA"}
CMF_CONDITIONS = {"greater_than_zero", "less_than_zero", "disabled"}
CONFIRMATION_MODES = {"strong_bullish_close", "disabled"}
ENVELOPE_MODES = {
    "lower_support",
    "close_below_lower",
    "upper_resistance",
    "close_above_upper",
    "inside_envelope",
}


@dataclass(frozen=True)
class KnoxEnvelopeStudyResult:
    summary: dict[str, Any]
    stock_stats: pd.DataFrame


def calculate_knox_envelope(
    frame: pd.DataFrame,
    *,
    knox_lookback: int = DEFAULT_KNOX_LOOKBACK,
    rsi_length: int = DEFAULT_RSI_LENGTH,
    momentum_length: int = DEFAULT_MOMENTUM_LENGTH,
    envelope_length: int = DEFAULT_ENVELOPE_LENGTH,
    envelope_percent: float = DEFAULT_ENVELOPE_PERCENT,
    envelope_ma_type: str = DEFAULT_ENVELOPE_MA_TYPE,
    envelope_proximity_pct: float = DEFAULT_ENVELOPE_PROXIMITY_PCT,
    cmf_length: int = DEFAULT_CMF_LENGTH,
    confirmation_close_location_pct: float = DEFAULT_CONFIRMATION_CLOSE_LOCATION_PCT,
) -> pd.DataFrame:
    """Calculate the published Knoxville logic and TradingView ENV bands bar by bar."""
    prepared = frame.copy()
    prepared["date"] = pd.to_datetime(prepared.get("date"), errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = (
        prepared.dropna(subset=["date", "high", "low", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    if prepared.empty:
        return prepared

    knox_lookback = max(int(knox_lookback), 5)
    rsi_length = max(int(rsi_length), 1)
    momentum_length = max(int(momentum_length), 1)
    envelope_length = max(int(envelope_length), 1)
    envelope_percent = max(float(envelope_percent), 0.0)
    envelope_proximity_pct = max(float(envelope_proximity_pct), 0.0)
    cmf_length = max(int(cmf_length), 1)
    confirmation_close_location_pct = min(
        max(float(confirmation_close_location_pct), 0.0),
        100.0,
    )
    envelope_ma_type = str(envelope_ma_type or DEFAULT_ENVELOPE_MA_TYPE).upper()
    if envelope_ma_type not in ENVELOPE_MA_TYPES:
        envelope_ma_type = DEFAULT_ENVELOPE_MA_TYPE

    close = prepared["close"].astype(float)
    high = prepared["high"].astype(float)
    low = prepared["low"].astype(float)
    open_ = pd.to_numeric(prepared.get("open", close), errors="coerce")
    momentum = close - close.shift(momentum_length)
    rsi = _pine_rsi(close, rsi_length)

    if envelope_ma_type == "EMA":
        basis = close.ewm(span=envelope_length, adjust=False, min_periods=envelope_length).mean()
    else:
        basis = close.rolling(envelope_length, min_periods=envelope_length).mean()
    envelope_fraction = envelope_percent / 100.0
    upper = basis * (1.0 + envelope_fraction)
    lower = basis * (1.0 - envelope_fraction)

    volume = pd.to_numeric(
        prepared.get("volume", pd.Series(0.0, index=prepared.index)),
        errors="coerce",
    ).fillna(0.0)
    candle_range = (high - low).replace(0.0, np.nan)
    money_flow_multiplier = (((close - low) - (high - close)) / candle_range).fillna(0.0)
    cmf = (
        (money_flow_multiplier * volume).rolling(cmf_length, min_periods=cmf_length).sum()
        / volume.rolling(cmf_length, min_periods=cmf_length).sum().replace(0.0, np.nan)
    )
    close_location_pct = ((close - low) / candle_range * 100.0).clip(0.0, 100.0)
    strong_bullish_close = (
        (close > open_)
        & (close > close.shift(1))
        & (close_location_pct >= confirmation_close_location_pct)
    ).fillna(False)

    bullish = np.zeros(len(prepared), dtype=bool)
    bearish = np.zeros(len(prepared), dtype=bool)
    reference_bars = np.zeros(len(prepared), dtype=int)

    for row_index in range(len(prepared)):
        if row_index < max(knox_lookback - 1, momentum_length):
            continue
        window_start = row_index - knox_lookback + 1
        is_highest = bool(high.iloc[row_index] == high.iloc[window_start : row_index + 1].max())
        is_lowest = bool(low.iloc[row_index] == low.iloc[window_start : row_index + 1].min())

        bar_up = 0
        bar_down = 0
        current_momentum = momentum.iloc[row_index]
        if pd.notna(current_momentum):
            for offset in range(5, knox_lookback + 1):
                reference_index = row_index - offset
                if reference_index < 0:
                    break
                reference_momentum = momentum.iloc[reference_index]
                if pd.isna(reference_momentum):
                    continue
                if float(current_momentum) < float(reference_momentum):
                    bar_up = offset
                if float(current_momentum) > float(reference_momentum):
                    bar_down = offset

        if bar_up > 0 and is_highest:
            was_overbought = _rsi_extreme_between(
                rsi,
                row_index,
                bar_up,
                threshold=70.0,
                above=True,
            )
            bearish[row_index] = bool(
                was_overbought and high.iloc[row_index] > high.iloc[row_index - bar_up]
            )
            if bearish[row_index]:
                reference_bars[row_index] = bar_up

        if bar_down > 0 and is_lowest:
            was_oversold = _rsi_extreme_between(
                rsi,
                row_index,
                bar_down,
                threshold=30.0,
                above=False,
            )
            bullish[row_index] = bool(
                was_oversold and low.iloc[row_index] < low.iloc[row_index - bar_down]
            )
            if bullish[row_index]:
                reference_bars[row_index] = bar_down

    safe_lower = lower.replace(0.0, np.nan)
    safe_upper = upper.replace(0.0, np.nan)
    lower_distance_pct = (close - lower) / safe_lower * 100.0
    upper_distance_pct = (upper - close) / safe_upper * 100.0
    low_distance_from_lower_pct = (low - lower) / safe_lower * 100.0
    high_distance_from_upper_pct = (high - upper) / safe_upper * 100.0
    lower_support = (
        lower.notna()
        & (low_distance_from_lower_pct.abs() <= envelope_proximity_pct)
    )
    upper_resistance = (
        upper.notna()
        & (high_distance_from_upper_pct.abs() <= envelope_proximity_pct)
    )

    prepared["momentum"] = momentum
    prepared["rsi"] = rsi
    prepared["knox_reference_bars"] = reference_bars
    prepared["knox_bullish"] = bullish
    prepared["knox_bearish"] = bearish
    prepared["envelope_basis"] = basis
    prepared["envelope_upper"] = upper
    prepared["envelope_lower"] = lower
    prepared["cmf"] = cmf
    prepared["close_location_pct"] = close_location_pct
    prepared["strong_bullish_close"] = strong_bullish_close
    prepared["distance_from_lower_pct"] = lower_distance_pct
    prepared["distance_from_upper_pct"] = upper_distance_pct
    prepared["low_distance_from_lower_pct"] = low_distance_from_lower_pct
    prepared["high_distance_from_upper_pct"] = high_distance_from_upper_pct
    prepared["envelope_lower_support"] = lower_support.fillna(False)
    prepared["envelope_close_below_lower"] = (lower.notna() & (close <= lower)).fillna(False)
    prepared["envelope_upper_resistance"] = upper_resistance.fillna(False)
    prepared["envelope_close_above_upper"] = (upper.notna() & (close >= upper)).fillna(False)
    prepared["envelope_inside"] = (lower.notna() & upper.notna() & close.between(lower, upper)).fillna(False)
    return prepared


def run_knox_envelope_study(
    storage: Storage,
    exchange: str = "NSE",
    *,
    symbols: list[str] | None = None,
    knox_lookback: int = DEFAULT_KNOX_LOOKBACK,
    rsi_length: int = DEFAULT_RSI_LENGTH,
    momentum_length: int = DEFAULT_MOMENTUM_LENGTH,
    signal_lookback_bars: int = DEFAULT_SIGNAL_LOOKBACK,
    signal_direction: str = DEFAULT_SIGNAL_DIRECTION,
    envelope_length: int = DEFAULT_ENVELOPE_LENGTH,
    envelope_percent: float = DEFAULT_ENVELOPE_PERCENT,
    envelope_ma_type: str = DEFAULT_ENVELOPE_MA_TYPE,
    envelope_mode: str = DEFAULT_ENVELOPE_MODE,
    envelope_proximity_pct: float = DEFAULT_ENVELOPE_PROXIMITY_PCT,
    cmf_length: int = DEFAULT_CMF_LENGTH,
    cmf_condition: str = DEFAULT_CMF_CONDITION,
    confirmation_mode: str = DEFAULT_CONFIRMATION_MODE,
    confirmation_window_bars: int = DEFAULT_CONFIRMATION_WINDOW,
    confirmation_close_location_pct: float = DEFAULT_CONFIRMATION_CLOSE_LOCATION_PCT,
    use_sharpe_filter: bool = DEFAULT_USE_SHARPE_FILTER,
    sharpe_lookback_days: int = DEFAULT_SHARPE_LOOKBACK_DAYS,
    annual_risk_free_rate_pct: float = DEFAULT_ANNUAL_RISK_FREE_RATE_PCT,
    min_sharpe_ratio: float | None = DEFAULT_MIN_SHARPE_RATIO,
    as_of_date: date | str | pd.Timestamp | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> KnoxEnvelopeStudyResult:
    signal_direction = str(signal_direction or DEFAULT_SIGNAL_DIRECTION).lower()
    if signal_direction not in SIGNAL_DIRECTIONS:
        signal_direction = DEFAULT_SIGNAL_DIRECTION
    envelope_mode = str(envelope_mode or DEFAULT_ENVELOPE_MODE).lower()
    if envelope_mode not in ENVELOPE_MODES:
        envelope_mode = DEFAULT_ENVELOPE_MODE
    envelope_ma_type = str(envelope_ma_type or DEFAULT_ENVELOPE_MA_TYPE).upper()
    if envelope_ma_type not in ENVELOPE_MA_TYPES:
        envelope_ma_type = DEFAULT_ENVELOPE_MA_TYPE
    signal_lookback_bars = max(int(signal_lookback_bars), 1)
    cmf_length = max(int(cmf_length), 1)
    cmf_condition = str(cmf_condition or DEFAULT_CMF_CONDITION).lower()
    if cmf_condition not in CMF_CONDITIONS:
        cmf_condition = DEFAULT_CMF_CONDITION
    confirmation_mode = str(confirmation_mode or DEFAULT_CONFIRMATION_MODE).lower()
    if confirmation_mode not in CONFIRMATION_MODES:
        confirmation_mode = DEFAULT_CONFIRMATION_MODE
    confirmation_window_bars = max(int(confirmation_window_bars), 0)
    confirmation_close_location_pct = min(
        max(float(confirmation_close_location_pct), 0.0),
        100.0,
    )
    use_sharpe_filter = bool(use_sharpe_filter)
    sharpe_lookback_days = max(int(sharpe_lookback_days), 2)
    annual_risk_free_rate_pct = max(float(annual_risk_free_rate_pct), -99.99)
    if min_sharpe_ratio is not None:
        min_sharpe_ratio = float(min_sharpe_ratio)

    if symbols is None:
        candidates = sorted(
            path.stem
            for path in (storage.data_root / "candles" / exchange / "1D").glob("*.csv")
            if not _is_excluded_symbol(path.stem)
        )
    else:
        candidates = sorted(
            {
                str(symbol or "").strip().upper()
                for symbol in symbols
                if not _is_excluded_symbol(str(symbol or "").strip().upper())
            }
        )
    name_map = _load_name_map(storage, exchange)
    rows: list[dict[str, Any]] = []
    min_history = max(
        int(knox_lookback) + int(momentum_length) + 1,
        int(envelope_length),
        cmf_length,
    )

    _emit_progress(
        progress_callback,
        phase="Scanning Knoxville and Envelope confluence",
        completed=0,
        total=len(candidates),
        current_symbol="",
        current_exchange=exchange,
    )
    for completed, symbol in enumerate(candidates, start=1):
        daily = storage.load_candles(exchange, symbol, "1D")
        if as_of_date is not None and not daily.empty:
            cutoff = pd.to_datetime(as_of_date, errors="coerce")
            daily_dates = pd.to_datetime(daily.get("date"), errors="coerce")
            if pd.notna(cutoff):
                daily = daily[daily_dates.dt.normalize() <= pd.Timestamp(cutoff).normalize()].copy()
        _emit_progress(
            progress_callback,
            phase="Scanning Knoxville and Envelope confluence",
            completed=completed,
            total=len(candidates),
            current_symbol=symbol,
            current_exchange=exchange,
        )
        if daily.empty or len(daily) < min_history:
            continue
        calculated = calculate_knox_envelope(
            daily,
            knox_lookback=knox_lookback,
            rsi_length=rsi_length,
            momentum_length=momentum_length,
            envelope_length=envelope_length,
            envelope_percent=envelope_percent,
            envelope_ma_type=envelope_ma_type,
            envelope_proximity_pct=envelope_proximity_pct,
            cmf_length=cmf_length,
            confirmation_close_location_pct=confirmation_close_location_pct,
        )
        if calculated.empty:
            continue

        envelope_column = {
            "lower_support": "envelope_lower_support",
            "close_below_lower": "envelope_close_below_lower",
            "upper_resistance": "envelope_upper_resistance",
            "close_above_upper": "envelope_close_above_upper",
            "inside_envelope": "envelope_inside",
        }[envelope_mode]
        if signal_direction == "bullish":
            direction_mask = calculated["knox_bullish"].fillna(False)
        elif signal_direction == "bearish":
            direction_mask = calculated["knox_bearish"].fillna(False)
        else:
            direction_mask = calculated["knox_bullish"].fillna(False) | calculated["knox_bearish"].fillna(False)
        if cmf_condition == "greater_than_zero":
            cmf_mask = calculated["cmf"] > 0.0
        elif cmf_condition == "less_than_zero":
            cmf_mask = calculated["cmf"] < 0.0
        else:
            cmf_mask = pd.Series(True, index=calculated.index)
        setup_mask = (
            direction_mask
            & calculated[envelope_column].fillna(False)
            & cmf_mask.fillna(False)
        )
        if confirmation_mode == "strong_bullish_close":
            setup_mask &= calculated["knox_bullish"].fillna(False)
            confluence_mask, setup_indexes = _confirmation_matches(
                setup_mask,
                calculated["strong_bullish_close"].fillna(False),
                confirmation_window_bars,
            )
        else:
            confluence_mask = setup_mask.to_numpy(dtype=bool)
            setup_indexes = np.where(confluence_mask, np.arange(len(calculated)), -1)
        recent = calculated.iloc[-signal_lookback_bars:]
        latest = calculated.iloc[-1]
        recent_mask = confluence_mask[-signal_lookback_bars:]
        recent_match_positions = np.flatnonzero(recent_mask)
        confirmation_match = (
            recent.iloc[int(recent_match_positions[-1])]
            if len(recent_match_positions)
            else None
        )
        technical_match = confirmation_match is not None
        if use_sharpe_filter:
            sharpe_ratio, sharpe_observations = _annualized_sharpe(
                calculated["close"],
                lookback_days=sharpe_lookback_days,
                annual_risk_free_rate_pct=annual_risk_free_rate_pct,
            )
        else:
            sharpe_ratio, sharpe_observations = float("nan"), 0
        sharpe_available = bool(np.isfinite(sharpe_ratio))
        sharpe_pass = bool(
            not use_sharpe_filter
            or min_sharpe_ratio is None
            or (sharpe_available and sharpe_ratio >= min_sharpe_ratio)
        )
        combined_match = technical_match and sharpe_pass
        confirmation_index = int(confirmation_match.name) if confirmation_match is not None else -1
        setup_index = int(setup_indexes[confirmation_index]) if confirmation_index >= 0 else -1
        latest_match = calculated.iloc[setup_index] if setup_index >= 0 else None
        match_side = ""
        if latest_match is not None:
            if bool(latest_match.get("knox_bullish")):
                match_side = "BULLISH"
            elif bool(latest_match.get("knox_bearish")):
                match_side = "BEARISH"
        signal_date = pd.to_datetime(latest_match.get("date"), errors="coerce") if latest_match is not None else pd.NaT
        signal_age = (
            int(len(calculated) - 1 - setup_index)
            if latest_match is not None
            else pd.NA
        )
        confirmation_age = (
            int(len(calculated) - 1 - confirmation_index)
            if confirmation_match is not None
            else pd.NA
        )
        reference_row = None
        if latest_match is not None:
            reference_bars = int(latest_match.get("knox_reference_bars") or 0)
            reference_index = int(latest_match.name) - reference_bars
            if reference_bars > 0 and reference_index >= 0:
                reference_row = calculated.iloc[reference_index]

        rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "name": name_map.get(symbol, symbol),
                "latest_date": latest.get("date"),
                "latest_close": _to_float(latest.get("close")),
                "technical_match": technical_match,
                "sharpe_ratio": _to_float(sharpe_ratio),
                "sharpe_observations": sharpe_observations,
                "sharpe_available": sharpe_available,
                "sharpe_pass": sharpe_pass,
                "combined_match": combined_match,
                "match_side": match_side,
                "signal_date": signal_date,
                "signal_age_bars": signal_age,
                "confirmation_date": confirmation_match.get("date") if confirmation_match is not None else pd.NaT,
                "confirmation_age_bars": confirmation_age,
                "confirmation_delay_bars": confirmation_index - setup_index if setup_index >= 0 else pd.NA,
                "confirmation_close": _to_float(confirmation_match.get("close")) if confirmation_match is not None else pd.NA,
                "confirmation_close_location_pct": _to_float(confirmation_match.get("close_location_pct")) if confirmation_match is not None else pd.NA,
                "confirmation_pass": (
                    confirmation_mode == "disabled"
                    or bool(confirmation_match.get("strong_bullish_close"))
                ) if confirmation_match is not None else False,
                "signal_close": _to_float(latest_match.get("close")) if latest_match is not None else pd.NA,
                "signal_low": _to_float(latest_match.get("low")) if latest_match is not None else pd.NA,
                "signal_high": _to_float(latest_match.get("high")) if latest_match is not None else pd.NA,
                "signal_rsi": _to_float(latest_match.get("rsi")) if latest_match is not None else pd.NA,
                "signal_momentum": _to_float(latest_match.get("momentum")) if latest_match is not None else pd.NA,
                "signal_cmf": _to_float(latest_match.get("cmf")) if latest_match is not None else pd.NA,
                "cmf_pass": bool(cmf_mask.loc[latest_match.name]) if latest_match is not None else False,
                "knox_reference_bars": _to_float(latest_match.get("knox_reference_bars")) if latest_match is not None else pd.NA,
                "reference_date": reference_row.get("date") if reference_row is not None else pd.NaT,
                "reference_low": _to_float(reference_row.get("low")) if reference_row is not None else pd.NA,
                "reference_high": _to_float(reference_row.get("high")) if reference_row is not None else pd.NA,
                "envelope_basis": _to_float(latest_match.get("envelope_basis")) if latest_match is not None else _to_float(latest.get("envelope_basis")),
                "envelope_lower": _to_float(latest_match.get("envelope_lower")) if latest_match is not None else _to_float(latest.get("envelope_lower")),
                "envelope_upper": _to_float(latest_match.get("envelope_upper")) if latest_match is not None else _to_float(latest.get("envelope_upper")),
                "distance_from_lower_pct": _to_float(latest_match.get("distance_from_lower_pct")) if latest_match is not None else _to_float(latest.get("distance_from_lower_pct")),
                "distance_from_upper_pct": _to_float(latest_match.get("distance_from_upper_pct")) if latest_match is not None else _to_float(latest.get("distance_from_upper_pct")),
                "low_distance_from_lower_pct": _to_float(latest_match.get("low_distance_from_lower_pct")) if latest_match is not None else _to_float(latest.get("low_distance_from_lower_pct")),
                "high_distance_from_upper_pct": _to_float(latest_match.get("high_distance_from_upper_pct")) if latest_match is not None else _to_float(latest.get("high_distance_from_upper_pct")),
            }
        )

    stock_stats = pd.DataFrame(rows)
    if not stock_stats.empty:
        stock_stats["latest_date"] = pd.to_datetime(stock_stats["latest_date"], errors="coerce")
        stock_stats["signal_date"] = pd.to_datetime(stock_stats["signal_date"], errors="coerce")
        stock_stats["reference_date"] = pd.to_datetime(stock_stats["reference_date"], errors="coerce")
        stock_stats = stock_stats.sort_values(
            ["combined_match", "signal_date", "symbol"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)

    summary = _build_summary(
        exchange,
        len(candidates),
        stock_stats,
        knox_lookback=knox_lookback,
        rsi_length=rsi_length,
        momentum_length=momentum_length,
        signal_lookback_bars=signal_lookback_bars,
        signal_direction=signal_direction,
        envelope_length=envelope_length,
        envelope_percent=envelope_percent,
        envelope_ma_type=envelope_ma_type,
        envelope_mode=envelope_mode,
        envelope_proximity_pct=envelope_proximity_pct,
        cmf_length=cmf_length,
        cmf_condition=cmf_condition,
        confirmation_mode=confirmation_mode,
        confirmation_window_bars=confirmation_window_bars,
        confirmation_close_location_pct=confirmation_close_location_pct,
        use_sharpe_filter=use_sharpe_filter,
        sharpe_lookback_days=sharpe_lookback_days,
        annual_risk_free_rate_pct=annual_risk_free_rate_pct,
        min_sharpe_ratio=min_sharpe_ratio,
    )
    return KnoxEnvelopeStudyResult(summary=summary, stock_stats=stock_stats)


def _rsi_extreme_between(
    rsi: pd.Series,
    row_index: int,
    reference_bars: int,
    *,
    threshold: float,
    above: bool,
) -> bool:
    """Check RSI only from the Knoxville reference candle through the endpoint."""
    start = max(0, int(row_index) - int(reference_bars))
    window = pd.to_numeric(rsi.iloc[start : int(row_index) + 1], errors="coerce")
    return bool((window > threshold).any() if above else (window < threshold).any())


def _annualized_sharpe(
    close: pd.Series,
    *,
    lookback_days: int = DEFAULT_SHARPE_LOOKBACK_DAYS,
    annual_risk_free_rate_pct: float = DEFAULT_ANNUAL_RISK_FREE_RATE_PCT,
    trading_days_per_year: int = 252,
) -> tuple[float, int]:
    """Return trailing annualized Sharpe and the daily-return observations used."""
    lookback_days = max(int(lookback_days), 2)
    trading_days_per_year = max(int(trading_days_per_year), 1)
    prices = pd.to_numeric(close, errors="coerce").replace([np.inf, -np.inf], np.nan)
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    returns = returns.tail(lookback_days)
    observations = int(len(returns))
    minimum_observations = min(60, lookback_days)
    if observations < max(2, minimum_observations):
        return float("nan"), observations

    annual_rate = max(float(annual_risk_free_rate_pct), -99.99) / 100.0
    daily_risk_free_rate = (1.0 + annual_rate) ** (1.0 / trading_days_per_year) - 1.0
    excess_returns = returns - daily_risk_free_rate
    volatility = float(excess_returns.std(ddof=1))
    if not np.isfinite(volatility) or volatility <= 0.0:
        return float("nan"), observations
    sharpe = float(excess_returns.mean() / volatility * np.sqrt(trading_days_per_year))
    return sharpe, observations


def _confirmation_matches(
    setup_mask: pd.Series | np.ndarray,
    confirmation_mask: pd.Series | np.ndarray,
    window_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    setup_values = np.asarray(setup_mask, dtype=bool)
    confirmation_values = np.asarray(confirmation_mask, dtype=bool)
    matches = np.zeros(len(setup_values), dtype=bool)
    setup_indexes = np.full(len(setup_values), -1, dtype=int)
    window_bars = max(int(window_bars), 0)
    for setup_index in np.flatnonzero(setup_values):
        end = min(int(setup_index) + window_bars, len(setup_values) - 1)
        candidates = np.flatnonzero(
            confirmation_values[int(setup_index) : end + 1]
        )
        if not len(candidates):
            continue
        confirmation_index = int(setup_index) + int(candidates[0])
        matches[confirmation_index] = True
        setup_indexes[confirmation_index] = int(setup_index)
    return matches, setup_indexes


def save_knox_envelope_outputs(result: KnoxEnvelopeStudyResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "latest_summary.csv"
    stock_stats_path = output_dir / "latest_stock_stats.csv"
    pd.DataFrame([result.summary]).to_csv(summary_path, index=False)
    result.stock_stats.to_csv(stock_stats_path, index=False)
    return {"summary": summary_path, "stock_stats": stock_stats_path}


def load_knox_envelope_outputs(output_dir: Path) -> KnoxEnvelopeStudyResult:
    summary: dict[str, Any] = {}
    summary_path = output_dir / "latest_summary.csv"
    if summary_path.exists():
        try:
            frame = pd.read_csv(summary_path)
            if not frame.empty:
                summary = frame.iloc[0].to_dict()
        except pd.errors.EmptyDataError:
            pass
    stock_stats_path = output_dir / "latest_stock_stats.csv"
    try:
        stock_stats = pd.read_csv(stock_stats_path) if stock_stats_path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        stock_stats = pd.DataFrame()
    return KnoxEnvelopeStudyResult(summary=summary, stock_stats=stock_stats)


def _pine_rsi(close: pd.Series, length: int) -> pd.Series:
    delta = pd.to_numeric(close, errors="coerce").diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = _pine_rma(gains, length)
    avg_loss = _pine_rma(losses, length)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask(avg_loss == 0.0, 100.0)
    rsi = rsi.mask((avg_gain == 0.0) & (avg_loss == 0.0), 50.0)
    return rsi


def _pine_rma(values: pd.Series, length: int) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.dropna()
    if len(valid) < int(length) or int(length) < 1:
        return result
    seed_end = valid.index[int(length) - 1]
    seed = float(valid.iloc[: int(length)].mean())
    result.loc[seed_end] = seed
    previous = seed
    start_position = series.index.get_loc(seed_end) + 1
    for position in range(start_position, len(series)):
        value = series.iloc[position]
        if pd.isna(value):
            result.iloc[position] = previous
            continue
        previous = ((previous * (float(length) - 1.0)) + float(value)) / float(length)
        result.iloc[position] = previous
    return result


def _is_excluded_symbol(symbol: str) -> bool:
    value = str(symbol or "").strip().upper()
    if not value or "-" in value or "NIFTY" in value or "BEES" in value or value.endswith("ETF"):
        return True
    return any(character.isdigit() for character in value)


def _build_summary(
    exchange: str,
    symbols_processed: int,
    stock_stats: pd.DataFrame,
    **settings: Any,
) -> dict[str, Any]:
    latest_date = ""
    if not stock_stats.empty and "latest_date" in stock_stats.columns:
        dates = pd.to_datetime(stock_stats["latest_date"], errors="coerce")
        if dates.notna().any():
            latest_date = dates.max().date().isoformat()
    matches = (
        int(stock_stats["combined_match"].fillna(False).astype(bool).sum())
        if not stock_stats.empty and "combined_match" in stock_stats.columns
        else 0
    )
    return {
        "logic_version": KNOX_ENVELOPE_LOGIC_VERSION,
        "exchange": exchange,
        "symbols_processed": int(symbols_processed),
        "stocks_with_history": int(len(stock_stats)),
        "combined_matches": matches,
        "latest_date": latest_date,
        **settings,
    }
