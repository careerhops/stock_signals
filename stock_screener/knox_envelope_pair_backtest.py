from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.knox_envelope_study import _is_excluded_symbol, _pine_rsi


DEFAULT_START_DATE = "2016-08-20"
DEFAULT_END_DATE = "2026-08-20"
DEFAULT_PROXIMITY_PCT = 5.0
DEFAULT_TARGET_PCT = 10.0
DEFAULT_ROUND_TRIP_COST_PCT = 0.35


@dataclass(frozen=True)
class PairStrategyParameters:
    knox_lookback: int = 100
    rsi_length: int = 14
    momentum_length: int = 20
    envelope_length: int = 100
    envelope_percent: float = 14.0
    envelope_ma_type: str = "SMA"

    @property
    def name(self) -> str:
        return (
            f"K{self.knox_lookback}_R{self.rsi_length}_M{self.momentum_length}"
            f"_E{self.envelope_length}_P{self.envelope_percent:g}"
        )


@dataclass(frozen=True)
class KnoxEnvelopePairBacktestResult:
    summary: dict[str, Any]
    parameter_stats: pd.DataFrame
    period_stats: pd.DataFrame
    baseline_trades: pd.DataFrame
    recommended_trades: pd.DataFrame
    open_positions: pd.DataFrame


BASELINE_PARAMETERS = PairStrategyParameters()


def default_search_parameters() -> tuple[PairStrategyParameters, ...]:
    """A bounded sensitivity grid containing the requested settings.

    It varies every parameter independently around the baseline and adds selected
    short/medium/long-horizon interactions. This keeps the search statistically
    inspectable instead of mining hundreds of nearly identical combinations.
    """
    baseline = BASELINE_PARAMETERS
    candidates = [baseline]
    candidates.extend(
        PairStrategyParameters(value, 14, 20, 100, 14.0)
        for value in (50, 75, 150, 200)
    )
    candidates.extend(
        PairStrategyParameters(100, value, 20, 100, 14.0)
        for value in (10, 20)
    )
    candidates.extend(
        PairStrategyParameters(100, 14, value, 100, 14.0)
        for value in (10, 30)
    )
    candidates.extend(
        PairStrategyParameters(100, 14, 20, value, 14.0)
        for value in (50, 75, 150, 200)
    )
    candidates.extend(
        PairStrategyParameters(100, 14, 20, 100, value)
        for value in (8.0, 10.0, 12.0, 16.0, 18.0)
    )
    candidates.extend(
        [
            PairStrategyParameters(50, 10, 10, 50, 10.0),
            PairStrategyParameters(50, 14, 20, 50, 10.0),
            PairStrategyParameters(75, 14, 20, 75, 12.0),
            PairStrategyParameters(150, 14, 20, 150, 16.0),
            PairStrategyParameters(200, 20, 30, 200, 18.0),
            PairStrategyParameters(50, 10, 20, 100, 14.0),
            PairStrategyParameters(100, 10, 10, 50, 14.0),
            PairStrategyParameters(100, 14, 10, 200, 10.0),
        ]
    )
    return tuple(dict.fromkeys(candidates))


def run_knox_envelope_pair_backtest(
    storage: Storage,
    *,
    exchange: str = "NSE",
    symbols: list[str] | None = None,
    start_date: str | pd.Timestamp = DEFAULT_START_DATE,
    end_date: str | pd.Timestamp = DEFAULT_END_DATE,
    parameters: Iterable[PairStrategyParameters] | None = None,
    include_baseline: bool = True,
    proximity_pct: float = DEFAULT_PROXIMITY_PCT,
    target_pct: float = DEFAULT_TARGET_PCT,
    round_trip_cost_pct: float = DEFAULT_ROUND_TRIP_COST_PCT,
    entry_cmf_length: int | None = None,
    min_entry_cmf: float = 0.0,
    min_entry_rvol20: float | None = None,
    entry_obv_accumulation_days: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> KnoxEnvelopePairBacktestResult:
    """Backtest paired bullish/bearish Knoxville line endpoints without look-ahead signals.

    A bullish Knoxville endpoint whose low is near the lower Envelope creates an entry
    signal. The fill is the following session's high. A later bearish Knoxville endpoint
    whose high is near the upper Envelope creates an exit signal, filled at the following
    session's low. Only one position per symbol may be open at a time.
    """
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    proximity_pct = max(float(proximity_pct), 0.0)
    target_pct = max(float(target_pct), 0.0)
    round_trip_cost_pct = max(float(round_trip_cost_pct), 0.0)
    entry_cmf_length = max(int(entry_cmf_length), 1) if entry_cmf_length else None
    min_entry_rvol20 = (
        max(float(min_entry_rvol20), 0.0) if min_entry_rvol20 is not None else None
    )
    entry_obv_accumulation_days = (
        max(int(entry_obv_accumulation_days), 1)
        if entry_obv_accumulation_days
        else None
    )
    parameter_grid = tuple(parameters or default_search_parameters())
    if include_baseline and BASELINE_PARAMETERS not in parameter_grid:
        parameter_grid = (BASELINE_PARAMETERS, *parameter_grid)
    parameter_grid = tuple(dict.fromkeys(parameter_grid))

    candidates = _candidate_symbols(storage, exchange, symbols)
    all_trades: list[pd.DataFrame] = []
    all_open_positions: list[dict[str, Any]] = []
    symbols_with_history = 0
    observed_start: pd.Timestamp | None = None
    observed_end: pd.Timestamp | None = None

    knox_keys = sorted(
        {
            (item.knox_lookback, item.rsi_length, item.momentum_length)
            for item in parameter_grid
        }
    )
    envelope_keys = sorted(
        {(item.envelope_length, item.envelope_percent, item.envelope_ma_type) for item in parameter_grid}
    )
    minimum_history = max(
        max(item.knox_lookback + item.momentum_length for item in parameter_grid),
        max(item.envelope_length for item in parameter_grid),
    ) + 2

    for completed, symbol in enumerate(candidates, start=1):
        daily = _prepare_daily(storage.load_candles(exchange, symbol, "1D"))
        daily = daily[daily["date"] <= end_ts].reset_index(drop=True)
        if len(daily) < minimum_history or daily["date"].max() < start_ts:
            _emit_progress(progress_callback, completed, len(candidates), symbol)
            continue
        symbols_with_history += 1
        first_date = pd.Timestamp(daily["date"].min())
        last_date = pd.Timestamp(daily["date"].max())
        observed_start = first_date if observed_start is None else min(observed_start, first_date)
        observed_end = last_date if observed_end is None else max(observed_end, last_date)

        rsi_cache = {
            length: _pine_rsi(daily["close"], length)
            for length in sorted({key[1] for key in knox_keys})
        }
        momentum_cache = {
            length: daily["close"] - daily["close"].shift(length)
            for length in sorted({key[2] for key in knox_keys})
        }
        knox_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
        for knox_lookback, rsi_length, momentum_length in knox_keys:
            knox_cache[(knox_lookback, rsi_length, momentum_length)] = _knoxville_endpoints(
                daily["high"],
                daily["low"],
                momentum_cache[momentum_length],
                rsi_cache[rsi_length],
                knox_lookback,
            )

        envelope_cache: dict[tuple[int, float, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for envelope_length, envelope_percent, envelope_ma_type in envelope_keys:
            envelope_cache[(envelope_length, envelope_percent, envelope_ma_type)] = _envelope_proximity(
                daily,
                envelope_length=envelope_length,
                envelope_percent=envelope_percent,
                envelope_ma_type=envelope_ma_type,
                proximity_pct=proximity_pct,
            )
        entry_quality_mask = _entry_quality_mask(
            daily,
            cmf_length=entry_cmf_length,
            min_cmf=min_entry_cmf,
            min_rvol20=min_entry_rvol20,
            obv_accumulation_days=entry_obv_accumulation_days,
        )

        for item in parameter_grid:
            bullish, bearish = knox_cache[
                (item.knox_lookback, item.rsi_length, item.momentum_length)
            ]
            near_lower, near_upper, lower_distance, upper_distance = envelope_cache[
                (item.envelope_length, item.envelope_percent, item.envelope_ma_type)
            ]
            entry_signals = bullish & near_lower & entry_quality_mask
            exit_signals = bearish & near_upper
            trades, open_position = simulate_paired_signals(
                daily,
                entry_signals=entry_signals,
                exit_signals=exit_signals,
                lower_distance_pct=lower_distance,
                upper_distance_pct=upper_distance,
                symbol=symbol,
                exchange=exchange,
                parameter_name=item.name,
                start_ts=start_ts,
                end_ts=end_ts,
                target_pct=target_pct,
                round_trip_cost_pct=round_trip_cost_pct,
            )
            if not trades.empty:
                all_trades.append(trades)
            if open_position is not None:
                all_open_positions.append(open_position)
        _emit_progress(progress_callback, completed, len(candidates), symbol)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else _empty_trades()
    if not trades.empty:
        # Exit-date cohorts prevent a validation entry from using a holdout-period exit.
        trades["period"] = trades["exit_date"].map(_period_name)
    open_positions = pd.DataFrame(all_open_positions)
    period_stats = _aggregate_trades(trades, ["parameter_name", "period"])
    parameter_stats = _build_parameter_stats(period_stats, parameter_grid)
    recommended_name = _select_recommended_parameter(parameter_stats)
    baseline_trades = trades.loc[trades.get("parameter_name", pd.Series(dtype=str)) == BASELINE_PARAMETERS.name].copy()
    recommended_trades = trades.loc[
        trades.get("parameter_name", pd.Series(dtype=str)) == recommended_name
    ].copy()
    baseline_stats = _overall_metrics(baseline_trades)
    recommended_stats = _overall_metrics(recommended_trades)
    baseline_open = open_positions.loc[
        open_positions.get("parameter_name", pd.Series(dtype=str)) == BASELINE_PARAMETERS.name
    ].copy()
    recommended_open = open_positions.loc[
        open_positions.get("parameter_name", pd.Series(dtype=str)) == recommended_name
    ].copy()
    baseline_marked = _mark_to_market_metrics(baseline_trades, baseline_open)
    recommended_marked = _mark_to_market_metrics(recommended_trades, recommended_open)
    recommended_parameters = next(
        (item for item in parameter_grid if item.name == recommended_name),
        parameter_grid[0] if parameter_grid else BASELINE_PARAMETERS,
    )
    summary = {
        "exchange": exchange,
        "requested_start_date": start_ts.date().isoformat(),
        "requested_end_date": end_ts.date().isoformat(),
        "observed_start_date": observed_start.date().isoformat() if observed_start is not None else "",
        "observed_end_date": observed_end.date().isoformat() if observed_end is not None else "",
        "symbols_considered": len(candidates),
        "symbols_with_sufficient_history": symbols_with_history,
        "parameter_combinations": len(parameter_grid),
        "proximity_pct": proximity_pct,
        "target_measure_pct": target_pct,
        "round_trip_cost_pct": round_trip_cost_pct,
        "entry_cmf_length": entry_cmf_length or "",
        "min_entry_cmf": min_entry_cmf if entry_cmf_length else "",
        "min_entry_rvol20": min_entry_rvol20 if min_entry_rvol20 is not None else "",
        "entry_obv_accumulation_days": entry_obv_accumulation_days or "",
        "baseline_parameter_name": BASELINE_PARAMETERS.name if include_baseline else "",
        "recommended_parameter_name": recommended_name,
        "recommended_knox_lookback": recommended_parameters.knox_lookback,
        "recommended_rsi_length": recommended_parameters.rsi_length,
        "recommended_momentum_length": recommended_parameters.momentum_length,
        "recommended_envelope_length": recommended_parameters.envelope_length,
        "recommended_envelope_percent": recommended_parameters.envelope_percent,
        "selection_rule": (
            "Selected only from trades exited in 2022-2023; ranked by 10% reach rate, "
            "then realized expectancy and profit factor. Exits from 2024 onward are untouched holdout reporting."
        ),
        **{f"baseline_{key}": value for key, value in baseline_stats.items()},
        **{f"baseline_marked_{key}": value for key, value in baseline_marked.items()},
        **{f"recommended_{key}": value for key, value in recommended_stats.items()},
        **{f"recommended_marked_{key}": value for key, value in recommended_marked.items()},
    }
    return KnoxEnvelopePairBacktestResult(
        summary=summary,
        parameter_stats=parameter_stats,
        period_stats=period_stats,
        baseline_trades=baseline_trades.reset_index(drop=True),
        recommended_trades=recommended_trades.reset_index(drop=True),
        open_positions=open_positions.reset_index(drop=True),
    )


def simulate_paired_signals(
    frame: pd.DataFrame,
    *,
    entry_signals: np.ndarray | pd.Series,
    exit_signals: np.ndarray | pd.Series,
    lower_distance_pct: np.ndarray | pd.Series | None = None,
    upper_distance_pct: np.ndarray | pd.Series | None = None,
    symbol: str = "",
    exchange: str = "NSE",
    parameter_name: str = "",
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
    target_pct: float = DEFAULT_TARGET_PCT,
    round_trip_cost_pct: float = DEFAULT_ROUND_TRIP_COST_PCT,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Pair each eligible bullish endpoint with the next eligible bearish endpoint."""
    prepared = _prepare_daily(frame)
    if prepared.empty:
        return _empty_trades(), None
    entry_values = np.asarray(entry_signals, dtype=bool)
    exit_values = np.asarray(exit_signals, dtype=bool)
    if len(entry_values) != len(prepared) or len(exit_values) != len(prepared):
        raise ValueError("Signal arrays must match the candle frame length.")
    lower_values = _optional_float_array(lower_distance_pct, len(prepared))
    upper_values = _optional_float_array(upper_distance_pct, len(prepared))
    start_ts = pd.Timestamp(start_ts or prepared["date"].min()).normalize()
    end_ts = pd.Timestamp(end_ts or prepared["date"].max()).normalize()

    entry_indexes = np.flatnonzero(
        entry_values & prepared["date"].between(start_ts, end_ts).to_numpy(dtype=bool)
    )
    exit_indexes = np.flatnonzero(exit_values)
    rows: list[dict[str, Any]] = []
    entry_pointer = 0
    next_signal_index = 0
    open_position: dict[str, Any] | None = None
    while entry_pointer < len(entry_indexes):
        signal_index = int(entry_indexes[entry_pointer])
        if signal_index < next_signal_index:
            entry_pointer += 1
            continue
        entry_index = signal_index + 1
        if entry_index >= len(prepared) or prepared.iloc[entry_index]["date"] > end_ts:
            break
        entry_price = _finite_float(prepared.iloc[entry_index]["high"])
        if entry_price is None or entry_price <= 0:
            entry_pointer += 1
            continue

        exit_pointer = int(np.searchsorted(exit_indexes, entry_index, side="left"))
        exit_signal_index = None
        exit_index = None
        while exit_pointer < len(exit_indexes):
            candidate_signal = int(exit_indexes[exit_pointer])
            candidate_exit = candidate_signal + 1
            if candidate_exit >= len(prepared) or prepared.iloc[candidate_exit]["date"] > end_ts:
                break
            exit_signal_index = candidate_signal
            exit_index = candidate_exit
            break
        if exit_signal_index is None or exit_index is None:
            latest = prepared.iloc[-1]
            holding_window = prepared.iloc[entry_index:]
            maximum_high = float(holding_window["high"].max())
            target_level = entry_price * (1.0 + float(target_pct) / 100.0)
            target_indexes = holding_window.index[holding_window["high"] >= target_level]
            previous_close = prepared["close"].shift(1).iloc[entry_index:]
            overnight_ratio = holding_window["open"] / previous_close.replace(0.0, np.nan)
            discontinuity = bool(
                ((overnight_ratio < 0.55) | (overnight_ratio > 1.80)).fillna(False).any()
            )
            latest_close = _finite_float(latest["close"])
            unrealized_gross = (
                (latest_close - entry_price) / entry_price * 100.0
                if latest_close is not None
                else np.nan
            )
            open_position = {
                "exchange": exchange,
                "symbol": symbol,
                "parameter_name": parameter_name,
                "entry_signal_date": prepared.iloc[signal_index]["date"],
                "entry_date": prepared.iloc[entry_index]["date"],
                "entry_price": entry_price,
                "latest_date": latest["date"],
                "latest_close": latest_close,
                "unrealized_gross_return_pct": unrealized_gross,
                "unrealized_net_return_pct": unrealized_gross - float(round_trip_cost_pct),
                "bars_open": int(len(prepared) - entry_index),
                "target_10_hit": len(target_indexes) > 0,
                "bars_to_target": (
                    int(target_indexes[0] - entry_index + 1) if len(target_indexes) > 0 else np.nan
                ),
                "mfe_pct": (maximum_high - entry_price) / entry_price * 100.0,
                "entry_low_distance_pct": lower_values[signal_index],
                "data_quality_pass": not discontinuity,
                "data_quality_reason": "" if not discontinuity else "overnight_price_discontinuity",
            }
            break

        exit_price = _finite_float(prepared.iloc[exit_index]["low"])
        if exit_price is None or exit_price <= 0:
            next_signal_index = exit_index
            entry_pointer += 1
            continue
        holding_window = prepared.iloc[entry_index : exit_signal_index + 1]
        execution_window = prepared.iloc[entry_index : exit_index + 1]
        previous_close = prepared["close"].shift(1).iloc[entry_index : exit_index + 1]
        overnight_ratio = execution_window["open"] / previous_close.replace(0.0, np.nan)
        discontinuity = bool(((overnight_ratio < 0.55) | (overnight_ratio > 1.80)).fillna(False).any())
        gross_return_pct = (exit_price - entry_price) / entry_price * 100.0
        net_return_pct = gross_return_pct - float(round_trip_cost_pct)
        maximum_high = float(holding_window["high"].max()) if not holding_window.empty else entry_price
        minimum_low = float(holding_window["low"].min()) if not holding_window.empty else entry_price
        target_level = entry_price * (1.0 + float(target_pct) / 100.0)
        target_indexes = holding_window.index[holding_window["high"] >= target_level]
        rows.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "parameter_name": parameter_name,
                "entry_signal_date": prepared.iloc[signal_index]["date"],
                "entry_date": prepared.iloc[entry_index]["date"],
                "entry_price": entry_price,
                "exit_signal_date": prepared.iloc[exit_signal_index]["date"],
                "exit_date": prepared.iloc[exit_index]["date"],
                "exit_price": exit_price,
                "bars_held": int(exit_index - entry_index + 1),
                "calendar_days_held": int(
                    (prepared.iloc[exit_index]["date"] - prepared.iloc[entry_index]["date"]).days
                ),
                "gross_return_pct": gross_return_pct,
                "net_return_pct": net_return_pct,
                "win_flag": net_return_pct > 0.0,
                "target_10_hit": maximum_high >= entry_price * (1.0 + float(target_pct) / 100.0),
                "target_10_hit_date": (
                    prepared.iloc[int(target_indexes[0])]["date"] if len(target_indexes) > 0 else pd.NaT
                ),
                "bars_to_target": (
                    int(target_indexes[0] - entry_index + 1) if len(target_indexes) > 0 else np.nan
                ),
                "mfe_pct": (maximum_high - entry_price) / entry_price * 100.0,
                "mae_pct": (minimum_low - entry_price) / entry_price * 100.0,
                "entry_low_distance_pct": lower_values[signal_index],
                "exit_high_distance_pct": upper_values[exit_signal_index],
                "data_quality_pass": not discontinuity,
                "data_quality_reason": "" if not discontinuity else "overnight_price_discontinuity",
            }
        )
        next_signal_index = exit_index
        entry_pointer += 1

    return (pd.DataFrame(rows) if rows else _empty_trades()), open_position


def save_knox_envelope_pair_backtest(
    result: KnoxEnvelopePairBacktestResult,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "summary.csv",
        "parameter_stats": output_dir / "parameter_stats.csv",
        "period_stats": output_dir / "period_stats.csv",
        "baseline_trades": output_dir / "baseline_trades.csv",
        "recommended_trades": output_dir / "recommended_trades.csv",
        "open_positions": output_dir / "open_positions.csv",
    }
    pd.DataFrame([result.summary]).to_csv(paths["summary"], index=False)
    result.parameter_stats.to_csv(paths["parameter_stats"], index=False)
    result.period_stats.to_csv(paths["period_stats"], index=False)
    result.baseline_trades.to_csv(paths["baseline_trades"], index=False)
    result.recommended_trades.to_csv(paths["recommended_trades"], index=False)
    result.open_positions.to_csv(paths["open_positions"], index=False)
    return paths


def _knoxville_endpoints(
    high: pd.Series,
    low: pd.Series,
    momentum: pd.Series,
    rsi: pd.Series,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return published bullish and bearish Knoxville endpoint flags."""
    lookback = max(int(lookback), 5)
    values_high = high.to_numpy(dtype=float)
    values_low = low.to_numpy(dtype=float)
    values_momentum = momentum.to_numpy(dtype=float)
    values_rsi = rsi.to_numpy(dtype=float)
    rolling_high = high.rolling(lookback, min_periods=lookback).max().to_numpy(dtype=float)
    rolling_low = low.rolling(lookback, min_periods=lookback).min().to_numpy(dtype=float)
    bullish = np.zeros(len(low), dtype=bool)
    bearish = np.zeros(len(high), dtype=bool)
    candidates = np.flatnonzero(
        (np.isfinite(rolling_high) & np.isclose(values_high, rolling_high, rtol=0.0, atol=1e-10))
        | (np.isfinite(rolling_low) & np.isclose(values_low, rolling_low, rtol=0.0, atol=1e-10))
    )
    for index in candidates:
        current_momentum = values_momentum[index]
        if not np.isfinite(current_momentum):
            continue
        bar_up = 0
        bar_down = 0
        for offset in range(5, lookback + 1):
            reference = index - offset
            if reference < 0:
                break
            reference_momentum = values_momentum[reference]
            if not np.isfinite(reference_momentum):
                continue
            if current_momentum < reference_momentum:
                bar_up = offset
            if current_momentum > reference_momentum:
                bar_down = offset
        if (
            bar_down > 0
            and np.isfinite(rolling_low[index])
            and np.isclose(values_low[index], rolling_low[index], rtol=0.0, atol=1e-10)
            and values_low[index] < values_low[index - bar_down]
        ):
            rsi_start = max(0, index - (bar_down + 1))
            bullish[index] = bool(np.any(values_rsi[rsi_start : index + 1] < 30.0))
        if (
            bar_up > 0
            and np.isfinite(rolling_high[index])
            and np.isclose(values_high[index], rolling_high[index], rtol=0.0, atol=1e-10)
            and values_high[index] > values_high[index - bar_up]
        ):
            rsi_start = max(0, index - (bar_up + 1))
            bearish[index] = bool(np.any(values_rsi[rsi_start : index + 1] > 70.0))
    return bullish, bearish


def _envelope_proximity(
    frame: pd.DataFrame,
    *,
    envelope_length: int,
    envelope_percent: float,
    envelope_ma_type: str,
    proximity_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = frame["close"].astype(float)
    ma_type = str(envelope_ma_type or "SMA").upper()
    if ma_type == "EMA":
        basis = close.ewm(span=envelope_length, adjust=False, min_periods=envelope_length).mean()
    else:
        basis = close.rolling(envelope_length, min_periods=envelope_length).mean()
    lower = basis * (1.0 - float(envelope_percent) / 100.0)
    upper = basis * (1.0 + float(envelope_percent) / 100.0)
    lower_distance = (frame["low"] - lower) / lower.replace(0.0, np.nan) * 100.0
    upper_distance = (frame["high"] - upper) / upper.replace(0.0, np.nan) * 100.0
    near_lower = lower.notna() & lower_distance.abs().le(float(proximity_pct))
    near_upper = upper.notna() & upper_distance.abs().le(float(proximity_pct))
    return (
        near_lower.fillna(False).to_numpy(dtype=bool),
        near_upper.fillna(False).to_numpy(dtype=bool),
        lower_distance.to_numpy(dtype=float),
        upper_distance.to_numpy(dtype=float),
    )


def _entry_quality_mask(
    frame: pd.DataFrame,
    *,
    cmf_length: int | None,
    min_cmf: float,
    min_rvol20: float | None,
    obv_accumulation_days: int | None = None,
) -> np.ndarray:
    mask = pd.Series(True, index=frame.index)
    volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    if cmf_length:
        candle_range = (frame["high"] - frame["low"]).replace(0.0, np.nan)
        multiplier = (
            ((frame["close"] - frame["low"]) - (frame["high"] - frame["close"]))
            / candle_range
        ).fillna(0.0)
        cmf = (
            (multiplier * volume).rolling(cmf_length, min_periods=cmf_length).sum()
            / volume.rolling(cmf_length, min_periods=cmf_length).sum().replace(0.0, np.nan)
        )
        mask &= cmf > float(min_cmf)
    if min_rvol20 is not None:
        prior_average = volume.shift(1).rolling(20, min_periods=20).mean()
        rvol20 = volume / prior_average.replace(0.0, np.nan)
        mask &= rvol20 >= float(min_rvol20)
    if obv_accumulation_days:
        direction = pd.to_numeric(frame["close"], errors="coerce").diff()
        obv_flow = pd.Series(0.0, index=frame.index)
        obv_flow.loc[direction > 0.0] = volume.loc[direction > 0.0]
        obv_flow.loc[direction < 0.0] = -volume.loc[direction < 0.0]
        obv = obv_flow.cumsum()
        mask &= obv > obv.shift(int(obv_accumulation_days))
    return mask.fillna(False).to_numpy(dtype=bool)


def _aggregate_trades(trades: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in trades.groupby(groups, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(groups, key_values))
        row.update(_overall_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _overall_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty or "net_return_pct" not in trades.columns:
        return {
            "trades": 0,
            "win_rate_pct": np.nan,
            "target_10_hit_rate_pct": np.nan,
            "avg_gross_return_pct": np.nan,
            "avg_net_return_pct": np.nan,
            "median_net_return_pct": np.nan,
            "profit_factor": np.nan,
            "avg_bars_held": np.nan,
            "median_bars_held": np.nan,
            "avg_mfe_pct": np.nan,
            "avg_mae_pct": np.nan,
            "median_bars_to_target": np.nan,
            "sum_net_return_pct": np.nan,
            "excluded_data_quality_trades": 0,
        }
    evaluated = trades
    excluded_count = 0
    if "data_quality_pass" in trades.columns:
        quality = trades["data_quality_pass"].fillna(False).astype(bool)
        excluded_count = int((~quality).sum())
        evaluated = trades.loc[quality]
    if evaluated.empty:
        empty = _overall_metrics(pd.DataFrame())
        empty["excluded_data_quality_trades"] = excluded_count
        return empty
    net = pd.to_numeric(evaluated["net_return_pct"], errors="coerce").dropna()
    gross = pd.to_numeric(evaluated["gross_return_pct"], errors="coerce").dropna()
    wins = net[net > 0.0]
    losses = net[net <= 0.0]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    return {
        "trades": int(len(net)),
        "win_rate_pct": float((net > 0.0).mean() * 100.0),
        "target_10_hit_rate_pct": float(evaluated["target_10_hit"].fillna(False).mean() * 100.0),
        "avg_gross_return_pct": float(gross.mean()),
        "avg_net_return_pct": float(net.mean()),
        "median_net_return_pct": float(net.median()),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else np.nan,
        "avg_bars_held": float(pd.to_numeric(evaluated["bars_held"], errors="coerce").mean()),
        "median_bars_held": float(pd.to_numeric(evaluated["bars_held"], errors="coerce").median()),
        "avg_mfe_pct": float(pd.to_numeric(evaluated["mfe_pct"], errors="coerce").mean()),
        "avg_mae_pct": float(pd.to_numeric(evaluated["mae_pct"], errors="coerce").mean()),
        "median_bars_to_target": float(
            pd.to_numeric(evaluated.get("bars_to_target"), errors="coerce").dropna().median()
        ),
        "sum_net_return_pct": float(net.sum()),
        "excluded_data_quality_trades": excluded_count,
    }


def _mark_to_market_metrics(closed: pd.DataFrame, open_positions: pd.DataFrame) -> dict[str, Any]:
    closed_quality = closed
    if not closed.empty and "data_quality_pass" in closed.columns:
        closed_quality = closed.loc[closed["data_quality_pass"].fillna(False).astype(bool)]
    open_quality = open_positions
    if not open_positions.empty and "data_quality_pass" in open_positions.columns:
        open_quality = open_positions.loc[
            open_positions["data_quality_pass"].fillna(False).astype(bool)
        ]
    closed_returns = pd.to_numeric(
        closed_quality.get("net_return_pct", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    open_returns = pd.to_numeric(
        open_quality.get("unrealized_net_return_pct", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    combined = pd.concat([closed_returns, open_returns], ignore_index=True)
    closed_targets = closed_quality.get("target_10_hit", pd.Series(dtype=bool)).fillna(False)
    open_targets = open_quality.get("target_10_hit", pd.Series(dtype=bool)).fillna(False)
    targets = pd.concat([closed_targets, open_targets], ignore_index=True)
    return {
        "closed_positions": int(len(closed_returns)),
        "open_positions": int(len(open_returns)),
        "positions": int(len(combined)),
        "win_rate_pct": float((combined > 0.0).mean() * 100.0) if not combined.empty else np.nan,
        "target_10_hit_rate_pct": float(targets.mean() * 100.0) if not targets.empty else np.nan,
        "avg_net_return_pct": float(combined.mean()) if not combined.empty else np.nan,
        "median_net_return_pct": float(combined.median()) if not combined.empty else np.nan,
    }


def _build_parameter_stats(
    period_stats: pd.DataFrame,
    parameters: tuple[PairStrategyParameters, ...],
) -> pd.DataFrame:
    parameter_map = {item.name: item for item in parameters}
    rows: list[dict[str, Any]] = []
    if period_stats.empty:
        return pd.DataFrame()
    for name, group in period_stats.groupby("parameter_name"):
        item = parameter_map[str(name)]
        by_period = {str(row["period"]): row for row in group.to_dict(orient="records")}
        row: dict[str, Any] = {
            "parameter_name": name,
            "knox_lookback": item.knox_lookback,
            "rsi_length": item.rsi_length,
            "momentum_length": item.momentum_length,
            "envelope_length": item.envelope_length,
            "envelope_percent": item.envelope_percent,
        }
        for period in ("DEVELOPMENT", "VALIDATION", "TEST"):
            stats = by_period.get(period, {})
            prefix = period.lower()
            for metric in (
                "trades",
                "win_rate_pct",
                "target_10_hit_rate_pct",
                "avg_net_return_pct",
                "median_net_return_pct",
                "profit_factor",
                "avg_bars_held",
                "excluded_data_quality_trades",
            ):
                row[f"{prefix}_{metric}"] = stats.get(metric, 0 if metric == "trades" else np.nan)
        validation_trades = int(row.get("validation_trades") or 0)
        validation_average = _finite_float(row.get("validation_avg_net_return_pct"))
        validation_factor = _finite_float(row.get("validation_profit_factor"))
        row["selection_eligible"] = bool(
            validation_trades >= 30
            and validation_average is not None
            and validation_average > 0.0
            and validation_factor is not None
            and validation_factor > 1.0
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [
            "selection_eligible",
            "validation_target_10_hit_rate_pct",
            "validation_avg_net_return_pct",
            "validation_profit_factor",
            "validation_trades",
            "parameter_name",
        ],
        ascending=[False, False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def _select_recommended_parameter(parameter_stats: pd.DataFrame) -> str:
    if parameter_stats.empty:
        return BASELINE_PARAMETERS.name
    eligible = parameter_stats.loc[parameter_stats["selection_eligible"].fillna(False)]
    if not eligible.empty:
        return str(eligible.iloc[0]["parameter_name"])
    sufficiently_sampled = parameter_stats.loc[parameter_stats["validation_trades"] >= 30]
    if not sufficiently_sampled.empty:
        return str(sufficiently_sampled.iloc[0]["parameter_name"])
    return BASELINE_PARAMETERS.name


def _candidate_symbols(storage: Storage, exchange: str, symbols: list[str] | None) -> list[str]:
    if symbols is None:
        values = [
            path.stem
            for path in (storage.data_root / "candles" / exchange / "1D").glob("*.csv")
        ]
    else:
        values = [str(symbol or "").strip().upper() for symbol in symbols]
    eligible = {value for value in values if not _is_excluded_symbol(value)}
    instruments = storage.load_instruments()
    required = {"exchange", "tradingsymbol", "name"}
    if not instruments.empty and required.issubset(instruments.columns):
        listed = instruments.loc[
            instruments["exchange"].astype(str).str.upper() == exchange.upper(),
            ["tradingsymbol", "name"],
        ].copy()
        listed["tradingsymbol"] = listed["tradingsymbol"].astype(str).str.upper().str.strip()
        normalized_name = listed["name"].fillna("").astype(str).str.upper()
        fund_name = (
            normalized_name.str.contains(r"\bETF\b|EXCHANGE[ -]TRADED FUND", regex=True)
            | normalized_name.str.contains(r"AMC\s*-", regex=True)
            | normalized_name.str.endswith(" GOLD FUND")
        )
        eligible.difference_update(listed.loc[fund_name, "tradingsymbol"])
    return sorted(eligible)


def _prepare_daily(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    prepared = frame.copy()
    prepared["date"] = pd.to_datetime(prepared.get("date"), errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        prepared[column] = pd.to_numeric(prepared.get(column), errors="coerce")
    return (
        prepared.dropna(subset=["date", "open", "high", "low", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )


def _period_name(value: Any) -> str:
    year = pd.Timestamp(value).year
    if year <= 2021:
        return "DEVELOPMENT"
    if year <= 2023:
        return "VALIDATION"
    return "TEST"


def _optional_float_array(values: np.ndarray | pd.Series | None, length: int) -> np.ndarray:
    if values is None:
        return np.full(length, np.nan, dtype=float)
    result = np.asarray(values, dtype=float)
    if len(result) != length:
        raise ValueError("Distance arrays must match the candle frame length.")
    return result


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    completed: int,
    total: int,
    symbol: str,
) -> None:
    if callback is not None:
        callback(
            {
                "phase": "Backtesting Knoxville line pairs and Envelope proximity",
                "completed": completed,
                "total": total,
                "current_symbol": symbol,
            }
        )


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "exchange",
            "symbol",
            "parameter_name",
            "entry_signal_date",
            "entry_date",
            "entry_price",
            "exit_signal_date",
            "exit_date",
            "exit_price",
            "bars_held",
            "calendar_days_held",
            "gross_return_pct",
            "net_return_pct",
            "win_flag",
            "target_10_hit",
            "target_10_hit_date",
            "bars_to_target",
            "mfe_pct",
            "mae_pct",
            "entry_low_distance_pct",
            "exit_high_distance_pct",
            "data_quality_pass",
            "data_quality_reason",
        ]
    )
