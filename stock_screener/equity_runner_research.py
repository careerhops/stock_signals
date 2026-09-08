from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.strategy.technical_ratings import _adx


FEATURE_COLUMNS = [
    "return_3d_pct",
    "return_20d_pct",
    "distance_sma20_pct",
    "distance_sma50_pct",
    "atr14_pct",
    "rsi14",
    "relative_volume20",
    "adx14",
    "di_spread",
    "range_position20_pct",
    "close_location_pct",
]


@dataclass(frozen=True)
class EquityRunnerReport:
    published_metrics: dict[str, float]
    open_positions: pd.DataFrame
    closed_trades: pd.DataFrame
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class RunnerRule:
    return_20d_min: float
    return_20d_max: float
    distance_sma50_min: float
    distance_sma50_max: float
    return_3d_min: float
    return_3d_max: float
    atr14_min: float
    atr14_max: float

    @property
    def name(self) -> str:
        return (
            f"R20_{self.return_20d_min:g}_{self.return_20d_max:g}__"
            f"SMA50_{self.distance_sma50_min:g}_{self.distance_sma50_max:g}__"
            f"R3_{self.return_3d_min:g}_{self.return_3d_max:g}__"
            f"ATR_{self.atr14_min:g}_{self.atr14_max:g}"
        )


@dataclass(frozen=True)
class EquityRunnerResearchResult:
    summary: dict[str, Any]
    friend_entry_audit: pd.DataFrame
    selection_profile: pd.DataFrame
    rule_search: pd.DataFrame
    split_metrics: pd.DataFrame
    latest_candidates: pd.DataFrame
    selected_rule: RunnerRule | None


def parse_equity_runner_report(path: str | Path) -> EquityRunnerReport:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    open_columns = [
        "Symbol",
        "Entered",
        "Entry",
        "Last",
        "Size (% of book)",
        "P&L",
        "Of book",
        "Days",
        "Target",
        "Stop",
        "Exit by",
    ]
    closed_columns = [
        "Symbol",
        "Entered",
        "Exited",
        "Entry",
        "Exit",
        "Size (% of book)",
        "P&L",
        "Of book",
        "Held",
        "Exit reason",
    ]
    open_header = "\t".join(open_columns)
    closed_header = "\t".join(closed_columns)
    try:
        open_start = lines.index(open_header) + 1
        closed_marker = lines.index("Closed trades")
        closed_start = lines.index(closed_header) + 1
    except ValueError as exc:
        raise ValueError("The attachment does not contain the expected Equity Runner tables") from exc

    open_positions = _parse_tab_rows(lines[open_start:closed_marker], open_columns)
    closed_trades = _parse_tab_rows(lines[closed_start:], closed_columns)
    if closed_trades.empty:
        raise ValueError("The Equity Runner report has no closed trades")

    _normalize_report_frame(open_positions, open_position=True)
    _normalize_report_frame(closed_trades, open_position=False)
    symbols = tuple(sorted(set(open_positions["symbol"]) | set(closed_trades["symbol"])))
    published_metrics = _parse_published_metrics(lines[:open_start])
    return EquityRunnerReport(published_metrics, open_positions, closed_trades, symbols)


def _parse_tab_rows(lines: Iterable[str], columns: list[str]) -> pd.DataFrame:
    rows = [line.split("\t") for line in lines if len(line.split("\t")) == len(columns)]
    return pd.DataFrame(rows, columns=columns)


def _normalize_report_frame(frame: pd.DataFrame, *, open_position: bool) -> None:
    rename = {
        "Symbol": "symbol",
        "Entered": "entry_date",
        "Entry": "entry_price",
        "Size (% of book)": "position_size_pct",
        "P&L": "return_pct",
        "Of book": "book_contribution_pct",
    }
    if open_position:
        rename.update(
            {
                "Last": "last_price",
                "Days": "days_held",
                "Target": "target_price",
                "Stop": "stop_price",
                "Exit by": "exit_by",
            }
        )
    else:
        rename.update(
            {
                "Exited": "exit_date",
                "Exit": "exit_price",
                "Held": "days_held",
                "Exit reason": "exit_reason",
            }
        )
    frame.rename(columns=rename, inplace=True)
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    if open_position:
        frame["exit_by"] = pd.to_datetime(frame["exit_by"], errors="coerce")
        numeric = ["entry_price", "last_price", "target_price", "stop_price"]
        frame["days_held"] = pd.to_numeric(frame["days_held"], errors="coerce")
    else:
        frame["exit_date"] = pd.to_datetime(frame["exit_date"], errors="coerce")
        frame["days_held"] = pd.to_numeric(
            frame["days_held"].astype(str).str.replace("d", "", regex=False), errors="coerce"
        )
        numeric = ["entry_price", "exit_price"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("position_size_pct", "return_pct", "book_contribution_pct"):
        frame[column] = pd.to_numeric(
            frame[column].astype(str).str.replace("%", "", regex=False).str.replace("+", "", regex=False),
            errors="coerce",
        )


def _parse_published_metrics(lines: list[str]) -> dict[str, float]:
    names = {
        "Net return": "net_return_pct",
        "Gross return": "gross_return_pct",
        "Sharpe": "sharpe",
        "Max drawdown": "max_drawdown_pct",
        "Win rate": "win_rate_pct",
        "Profit factor": "profit_factor",
        "Avg win": "avg_win_pct",
        "Avg loss": "avg_loss_pct",
        "Expectancy": "expectancy_pct",
    }
    metrics: dict[str, float] = {}
    for index, line in enumerate(lines[:-1]):
        key = names.get(line.strip())
        if not key:
            continue
        raw = lines[index + 1].strip().replace("%", "").replace("+", "")
        try:
            metrics[key] = float(raw)
        except ValueError:
            continue
    return metrics


def calculate_runner_features(candles: pd.DataFrame) -> pd.DataFrame:
    frame = candles.copy()
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce", format="mixed")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame = (
        frame.dropna(subset=["date", "open", "high", "low", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if frame.empty:
        return frame

    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"].fillna(0.0)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    atr14 = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    frame["return_1d_pct"] = close.pct_change(fill_method=None) * 100.0
    frame["return_3d_pct"] = (close / close.shift(3) - 1.0) * 100.0
    frame["return_20d_pct"] = (close / close.shift(20) - 1.0) * 100.0
    frame["distance_sma20_pct"] = (close / close.rolling(20, min_periods=20).mean() - 1.0) * 100.0
    frame["distance_sma50_pct"] = (close / close.rolling(50, min_periods=50).mean() - 1.0) * 100.0
    frame["atr14_pct"] = atr14 / close * 100.0
    frame["relative_volume20"] = volume / volume.shift(1).rolling(20, min_periods=20).mean().replace(0.0, np.nan)

    delta = close.diff()
    average_gain = delta.clip(lower=0.0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    average_loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    frame["rsi14"] = 100.0 - (100.0 / (1.0 + relative_strength))

    di_plus, di_minus, adx14 = _adx(high, low, close, 14, 14)
    frame["di_plus"] = di_plus
    frame["di_minus"] = di_minus
    frame["di_spread"] = di_plus - di_minus
    frame["adx14"] = adx14

    prior_high20 = high.shift(1).rolling(20, min_periods=20).max()
    prior_low20 = low.shift(1).rolling(20, min_periods=20).min()
    prior_range20 = (prior_high20 - prior_low20).replace(0.0, np.nan)
    frame["range_position20_pct"] = (close - prior_low20) / prior_range20 * 100.0
    frame["close_location_pct"] = (close - low) / (high - low).replace(0.0, np.nan) * 100.0
    frame["average_traded_value20"] = (close * volume).shift(1).rolling(20, min_periods=20).mean()

    # Avoid learning from likely split/bonus discontinuities in unadjusted exchange history.
    frame["data_quality_pass"] = frame["return_1d_pct"].abs().rolling(20, min_periods=20).max() < 45.0
    return frame


def attach_forward_outcomes(
    features: pd.DataFrame,
    *,
    target_pct: float = 15.0,
    stop_pct: float = 5.0,
    max_hold_sessions: int = 7,
    trailing_stop: bool = True,
    round_trip_cost_pct: float = 0.15,
) -> pd.DataFrame:
    frame = features.copy().reset_index(drop=True)
    count = len(frame)
    returns = np.full(count, np.nan)
    holding_sessions = np.full(count, np.nan)
    exit_indexes = np.full(count, -1, dtype=int)
    exit_reasons = np.full(count, "", dtype=object)
    if count < 2:
        return _set_outcome_columns(frame, returns, holding_sessions, exit_indexes, exit_reasons)

    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)

    for signal_index in range(count - 1):
        entry_index = signal_index + 1
        entry_price = opens[entry_index]
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        target_price = entry_price * (1.0 + target_pct / 100.0)
        stop_floor = entry_price * (1.0 - stop_pct / 100.0)
        active_stop = stop_floor
        peak = entry_price
        last_index = min(count - 1, signal_index + max_hold_sessions)

        for bar_index in range(entry_index, last_index + 1):
            exit_price: float | None = None
            reason = ""
            if opens[bar_index] <= active_stop:
                exit_price, reason = opens[bar_index], "GAP_STOP"
            elif opens[bar_index] >= target_price:
                exit_price, reason = opens[bar_index], "GAP_TARGET"
            elif lows[bar_index] <= active_stop:
                # Daily bars cannot reveal whether target or stop came first. Use the stop.
                exit_price, reason = active_stop, "STOP"
            elif highs[bar_index] >= target_price:
                exit_price, reason = target_price, "TARGET"
            elif bar_index == last_index:
                exit_price, reason = closes[bar_index], "MAX_HOLD"

            if exit_price is not None:
                returns[signal_index] = (exit_price / entry_price - 1.0) * 100.0 - round_trip_cost_pct
                holding_sessions[signal_index] = bar_index - signal_index
                exit_indexes[signal_index] = bar_index
                exit_reasons[signal_index] = reason
                break

            if trailing_stop:
                peak = max(peak, highs[bar_index])
                active_stop = max(stop_floor, peak * (1.0 - stop_pct / 100.0))

    return _set_outcome_columns(frame, returns, holding_sessions, exit_indexes, exit_reasons)


def _set_outcome_columns(
    frame: pd.DataFrame,
    returns: np.ndarray,
    holding_sessions: np.ndarray,
    exit_indexes: np.ndarray,
    exit_reasons: np.ndarray,
) -> pd.DataFrame:
    frame["model_return_pct"] = returns
    frame["holding_sessions"] = holding_sessions
    frame["exit_index"] = exit_indexes
    frame["exit_reason"] = exit_reasons
    dates = frame["date"].to_numpy()
    frame["exit_date"] = [dates[index] if index >= 0 else pd.NaT for index in exit_indexes]
    return frame


def run_equity_runner_research(
    storage: Storage,
    report_path: str | Path,
    *,
    exchange: str = "NSE",
    validation_start: str = "2024-01-01",
    test_start: str = "2026-01-01",
    min_average_traded_value: float = 10_000_000.0,
    trailing_stop: bool = True,
) -> EquityRunnerResearchResult:
    report = parse_equity_runner_report(report_path)
    frames: list[pd.DataFrame] = []
    latest_dates: dict[str, pd.Timestamp] = {}
    missing_symbols: list[str] = []
    for symbol in report.symbols:
        candles = storage.load_candles(exchange, symbol, "1D")
        if candles.empty:
            missing_symbols.append(symbol)
            continue
        featured = calculate_runner_features(candles)
        if featured.empty:
            missing_symbols.append(symbol)
            continue
        featured = attach_forward_outcomes(featured, trailing_stop=trailing_stop)
        featured["symbol"] = symbol
        featured["bar_index"] = np.arange(len(featured), dtype=int)
        latest_dates[symbol] = pd.Timestamp(featured["date"].max()).normalize()
        frames.append(featured)
    if not frames:
        raise ValueError("No local NSE candle history was found for the attachment cohort")

    data = pd.concat(frames, ignore_index=True)
    base_mask = (
        data[FEATURE_COLUMNS].notna().all(axis=1)
        & data["model_return_pct"].notna()
        & data["data_quality_pass"].fillna(False)
        & (data["distance_sma20_pct"] > 0.0)
        & (data["di_spread"] > 0.0)
        & (data["rsi14"].between(40.0, 90.0))
        & (data["range_position20_pct"].between(20.0, 130.0))
        & (data["average_traded_value20"] >= float(min_average_traded_value))
    )
    eligible = data.loc[base_mask].copy()
    validation_date = pd.Timestamp(validation_start)
    test_date = pd.Timestamp(test_start)
    development = eligible.loc[eligible["date"] < validation_date].copy()
    validation = eligible.loc[eligible["date"].between(validation_date, test_date - pd.Timedelta(days=1))].copy()
    test = eligible.loc[eligible["date"] >= test_date].copy()

    rules = list(_candidate_rules())
    search_rows: list[dict[str, Any]] = []
    for rule in rules:
        development_rows = development.loc[_rule_mask(development, rule)]
        validation_rows = validation.loc[_rule_mask(validation, rule)]
        if len(development_rows) < 150 or len(validation_rows) < 60:
            continue
        development_metrics = _independent_metrics(development_rows)
        validation_metrics = _independent_metrics(validation_rows)
        robustness = _robustness_score(development_metrics, validation_metrics)
        search_rows.append(
            {
                "rule_name": rule.name,
                **rule.__dict__,
                "development_trades": development_metrics["trades"],
                "development_expectancy_pct": development_metrics["expectancy_pct"],
                "development_profit_factor": development_metrics["profit_factor"],
                "validation_trades": validation_metrics["trades"],
                "validation_expectancy_pct": validation_metrics["expectancy_pct"],
                "validation_profit_factor": validation_metrics["profit_factor"],
                "robustness_score": robustness,
            }
        )
    rule_search = pd.DataFrame(search_rows)
    if rule_search.empty:
        selected_rule = None
    else:
        rule_search = rule_search.sort_values(
            ["robustness_score", "validation_profit_factor", "validation_trades"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        selected_rule = _rule_from_row(rule_search.iloc[0])

    split_rows: list[dict[str, Any]] = []
    if selected_rule is not None:
        for split_name, split_frame in (
            ("DEVELOPMENT_BEFORE_2024", development),
            ("VALIDATION_2024_2025", validation),
            ("TEST_2026", test),
            ("TEST_FRIEND_PERIOD", test.loc[test["date"] >= pd.Timestamp("2026-06-04")]),
        ):
            selected = split_frame.loc[_rule_mask(split_frame, selected_rule)]
            selected = _select_non_overlapping_trades(selected)
            split_rows.append({"split": split_name, **_trade_metrics(selected)})
    split_metrics = pd.DataFrame(split_rows)

    friend_entry_audit = _friend_entry_audit(report, data)
    selection_profile = _friend_selection_profile(report, data)
    replay_metrics = _trade_metrics(friend_entry_audit.rename(columns={"replay_return_pct": "model_return_pct"}))
    latest_candidates = _latest_candidates(data, latest_dates, selected_rule)
    expansion_ready = _passes_expansion_gate(split_metrics)
    summary = {
        "attachment_closed_trades": int(len(report.closed_trades)),
        "attachment_open_positions": int(len(report.open_positions)),
        "attachment_unique_symbols": int(len(report.symbols)),
        "attachment_equal_weight_profit_factor": _profit_factor(report.closed_trades["return_pct"]),
        "attachment_position_weighted_profit_factor": _profit_factor(
            report.closed_trades["return_pct"] * report.closed_trades["position_size_pct"]
        ),
        "attachment_expectancy_pct": float(report.closed_trades["return_pct"].mean()),
        "attachment_win_rate_pct": float((report.closed_trades["return_pct"] > 0.0).mean() * 100.0),
        "replayed_trades": replay_metrics["trades"],
        "replayed_win_rate_pct": replay_metrics["win_rate_pct"],
        "replayed_expectancy_pct": replay_metrics["expectancy_pct"],
        "replayed_profit_factor": replay_metrics["profit_factor"],
        "exit_policy": "TRAILING_5_PERCENT" if trailing_stop else "FIXED_5_PERCENT",
        "symbols_with_candles": int(len(frames)),
        "missing_symbols": ",".join(missing_symbols),
        "candidate_rules_tested": int(len(rule_search)),
        "selected_rule": selected_rule.name if selected_rule else "",
        "latest_cohort_date": max(latest_dates.values()).strftime("%Y-%m-%d"),
        "latest_candidates": int(len(latest_candidates)),
        "expansion_ready": bool(expansion_ready),
        "expansion_gate": "Validation and 2026 test each require PF >= 1.30, expectancy >= 0.50%, and >= 30 trades.",
    }
    return EquityRunnerResearchResult(
        summary=summary,
        friend_entry_audit=friend_entry_audit,
        selection_profile=selection_profile,
        rule_search=rule_search,
        split_metrics=split_metrics,
        latest_candidates=latest_candidates,
        selected_rule=selected_rule,
    )


def _candidate_rules() -> Iterable[RunnerRule]:
    for values in itertools.product(
        (0.0, 5.0, 10.0, 15.0),
        (30.0, 40.0, 60.0),
        (0.0, 5.0, 10.0),
        (25.0, 35.0, 50.0),
        (-5.0, 0.0, 1.0),
        (6.0, 10.0, 20.0),
        (2.0, 3.0, 4.0),
        (7.0, 10.0, 15.0),
    ):
        rule = RunnerRule(*values)
        if (
            rule.return_20d_min < rule.return_20d_max
            and rule.distance_sma50_min < rule.distance_sma50_max
            and rule.return_3d_min < rule.return_3d_max
            and rule.atr14_min < rule.atr14_max
        ):
            yield rule


def _rule_mask(frame: pd.DataFrame, rule: RunnerRule) -> pd.Series:
    return (
        frame["return_20d_pct"].between(rule.return_20d_min, rule.return_20d_max)
        & frame["distance_sma50_pct"].between(rule.distance_sma50_min, rule.distance_sma50_max)
        & frame["return_3d_pct"].between(rule.return_3d_min, rule.return_3d_max)
        & frame["atr14_pct"].between(rule.atr14_min, rule.atr14_max)
    )


def _independent_metrics(frame: pd.DataFrame) -> dict[str, float]:
    returns = pd.to_numeric(frame["model_return_pct"], errors="coerce").dropna()
    return {
        "trades": float(len(returns)),
        "expectancy_pct": float(returns.mean()) if len(returns) else math.nan,
        "profit_factor": _profit_factor(returns),
    }


def _robustness_score(development: dict[str, float], validation: dict[str, float]) -> float:
    expectancies = [development["expectancy_pct"], validation["expectancy_pct"]]
    profit_factors = [development["profit_factor"], validation["profit_factor"]]
    if not all(np.isfinite(value) for value in expectancies + profit_factors):
        return -math.inf
    minimum_expectancy = min(expectancies)
    minimum_profit_factor = min(profit_factors)
    if minimum_expectancy <= 0.0 or minimum_profit_factor <= 1.0:
        return minimum_expectancy - abs(1.0 - minimum_profit_factor)
    count_penalty = min(1.0, math.sqrt(validation["trades"] / 100.0))
    stability_penalty = 1.0 / (1.0 + abs(expectancies[0] - expectancies[1]))
    return minimum_expectancy * minimum_profit_factor * count_penalty * stability_penalty


def _rule_from_row(row: pd.Series) -> RunnerRule:
    return RunnerRule(
        return_20d_min=float(row["return_20d_min"]),
        return_20d_max=float(row["return_20d_max"]),
        distance_sma50_min=float(row["distance_sma50_min"]),
        distance_sma50_max=float(row["distance_sma50_max"]),
        return_3d_min=float(row["return_3d_min"]),
        return_3d_max=float(row["return_3d_max"]),
        atr14_min=float(row["atr14_min"]),
        atr14_max=float(row["atr14_max"]),
    )


def _select_non_overlapping_trades(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    selected_indexes: list[int] = []
    for _, symbol_rows in frame.sort_values(["symbol", "bar_index"]).groupby("symbol", sort=False):
        blocked_through = -1
        for index, row in symbol_rows.iterrows():
            bar_index = int(row["bar_index"])
            if bar_index <= blocked_through:
                continue
            selected_indexes.append(index)
            blocked_through = int(row["exit_index"])
    return frame.loc[selected_indexes].sort_values(["date", "symbol"]).reset_index(drop=True)


def _trade_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    returns = pd.to_numeric(frame.get("model_return_pct"), errors="coerce").dropna()
    if returns.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": math.nan,
            "expectancy_pct": math.nan,
            "profit_factor": math.nan,
            "average_win_pct": math.nan,
            "average_loss_pct": math.nan,
        }
    wins = returns.loc[returns > 0.0]
    losses = returns.loc[returns <= 0.0]
    return {
        "trades": int(len(returns)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": float(len(wins) / len(returns) * 100.0),
        "expectancy_pct": float(returns.mean()),
        "profit_factor": _profit_factor(returns),
        "average_win_pct": float(wins.mean()) if len(wins) else math.nan,
        "average_loss_pct": float(losses.mean()) if len(losses) else math.nan,
    }


def _profit_factor(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    gross_profit = float(values.loc[values > 0.0].sum())
    gross_loss = abs(float(values.loc[values < 0.0].sum()))
    if gross_loss == 0.0:
        return math.inf if gross_profit > 0.0 else math.nan
    return gross_profit / gross_loss


def _friend_entry_audit(report: EquityRunnerReport, data: pd.DataFrame) -> pd.DataFrame:
    closed = report.closed_trades.copy()
    closed["friend_trade_id"] = np.arange(len(closed), dtype=int)
    prior_rows = data.sort_values(["symbol", "date"]).copy()
    prior_rows["entry_date"] = prior_rows.groupby("symbol")["date"].shift(-1)
    columns = [
        "symbol",
        "entry_date",
        "date",
        *FEATURE_COLUMNS,
        "model_return_pct",
        "holding_sessions",
        "exit_date",
        "exit_reason",
    ]
    audit = closed.merge(prior_rows[columns], on=["symbol", "entry_date"], how="left", validate="many_to_one")
    audit.rename(
        columns={
            "date": "feature_date",
            "return_pct": "friend_return_pct",
            "model_return_pct": "replay_return_pct",
            "holding_sessions": "replay_holding_sessions",
            "exit_date_y": "replay_exit_date",
            "exit_reason_y": "replay_exit_reason",
            "exit_date_x": "friend_exit_date",
            "exit_reason_x": "friend_exit_reason",
        },
        inplace=True,
    )
    audit["friend_win"] = audit["friend_return_pct"] > 0.0
    return audit.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def _friend_selection_profile(report: EquityRunnerReport, data: pd.DataFrame) -> pd.DataFrame:
    events = set(
        zip(
            pd.concat([report.closed_trades["symbol"], report.open_positions["symbol"]]),
            pd.concat([report.closed_trades["entry_date"], report.open_positions["entry_date"]]).dt.normalize(),
        )
    )
    comparison = data.sort_values(["symbol", "date"]).copy()
    comparison["next_date"] = comparison.groupby("symbol")["date"].shift(-1)
    comparison["friend_selected_next"] = [
        (symbol, pd.Timestamp(next_date).normalize()) in events if pd.notna(next_date) else False
        for symbol, next_date in zip(comparison["symbol"], comparison["next_date"])
    ]
    first_entry = min(date for _, date in events)
    last_entry = max(date for _, date in events)
    comparison = comparison.loc[comparison["next_date"].between(first_entry, last_entry)].copy()
    selected = comparison.loc[comparison["friend_selected_next"]]
    controls = comparison.loc[~comparison["friend_selected_next"]]

    rows: list[dict[str, Any]] = []
    for column in FEATURE_COLUMNS:
        rows.append(
            {
                "measure": f"median_{column}",
                "friend_value": float(pd.to_numeric(selected[column], errors="coerce").median()),
                "control_value": float(pd.to_numeric(controls[column], errors="coerce").median()),
            }
        )
    boolean_measures = {
        "close_above_sma20_pct": comparison["distance_sma20_pct"] > 0.0,
        "close_above_sma50_pct": comparison["distance_sma50_pct"] > 0.0,
        "di_plus_above_di_minus_pct": comparison["di_spread"] > 0.0,
        "return_20d_at_least_10_pct": comparison["return_20d_pct"] >= 10.0,
        "relative_volume_at_least_1_5_pct": comparison["relative_volume20"] >= 1.5,
        "adx_at_least_20_pct": comparison["adx14"] >= 20.0,
    }
    for name, mask in boolean_measures.items():
        rows.append(
            {
                "measure": name,
                "friend_value": float(mask.loc[selected.index].mean() * 100.0),
                "control_value": float(mask.loc[controls.index].mean() * 100.0),
            }
        )
    profile = pd.DataFrame(rows)
    profile["difference"] = profile["friend_value"] - profile["control_value"]
    return profile


def _latest_candidates(
    data: pd.DataFrame,
    latest_dates: dict[str, pd.Timestamp],
    rule: RunnerRule | None,
) -> pd.DataFrame:
    columns = [
        "symbol",
        "date",
        "close",
        *FEATURE_COLUMNS,
        "average_traded_value20",
    ]
    if rule is None or not latest_dates:
        return pd.DataFrame(columns=columns)
    cohort_latest = max(latest_dates.values())
    recent = data.loc[data["date"].dt.normalize() == cohort_latest].copy()
    recent = recent.loc[
        recent[FEATURE_COLUMNS].notna().all(axis=1)
        & recent["data_quality_pass"].fillna(False)
        & (recent["distance_sma20_pct"] > 0.0)
        & (recent["di_spread"] > 0.0)
        & (recent["rsi14"].between(40.0, 90.0))
        & (recent["range_position20_pct"].between(20.0, 130.0))
        & _rule_mask(recent, rule)
    ].copy()
    recent["next_session_entry"] = True
    return recent[[*columns, "next_session_entry"]].sort_values(
        ["return_20d_pct", "di_spread"], ascending=[False, False]
    ).reset_index(drop=True)


def _passes_expansion_gate(split_metrics: pd.DataFrame) -> bool:
    if split_metrics.empty:
        return False
    required = split_metrics.loc[split_metrics["split"].isin(["VALIDATION_2024_2025", "TEST_2026"])]
    if len(required) != 2:
        return False
    return bool(
        (required["trades"] >= 30).all()
        and (required["profit_factor"] >= 1.30).all()
        and (required["expectancy_pct"] >= 0.50).all()
    )


def save_equity_runner_research(result: EquityRunnerResearchResult, output_dir: str | Path) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "friend_entry_audit": directory / "friend_entry_audit.csv",
        "selection_profile": directory / "selection_profile.csv",
        "rule_search": directory / "rule_search.csv",
        "split_metrics": directory / "split_metrics.csv",
        "latest_candidates": directory / "latest_candidates.csv",
        "report": directory / "report.md",
    }
    result.friend_entry_audit.to_csv(paths["friend_entry_audit"], index=False)
    result.selection_profile.to_csv(paths["selection_profile"], index=False)
    result.rule_search.to_csv(paths["rule_search"], index=False)
    result.split_metrics.to_csv(paths["split_metrics"], index=False)
    result.latest_candidates.to_csv(paths["latest_candidates"], index=False)
    paths["report"].write_text(_markdown_report(result), encoding="utf-8")
    return paths


def _markdown_report(result: EquityRunnerResearchResult) -> str:
    summary = result.summary
    lines = [
        "# Equity Runner Cohort Research",
        "",
        "This report is research output, not an investment recommendation.",
        "",
        "## Attachment Audit",
        "",
        f"- Closed trades: {summary['attachment_closed_trades']}",
        f"- Open positions: {summary['attachment_open_positions']}",
        f"- Exact stock cohort: {summary['attachment_unique_symbols']}",
        f"- Equal-weight profit factor: {summary['attachment_equal_weight_profit_factor']:.3f}",
        f"- Position-weighted profit factor: {summary['attachment_position_weighted_profit_factor']:.3f}",
        f"- Expectancy: {summary['attachment_expectancy_pct']:.3f}% per trade",
        f"- Next-session replay policy: {summary['exit_policy']}",
        f"- Replayed trades with sufficient candles: {summary['replayed_trades']}",
        f"- Replayed win rate: {summary['replayed_win_rate_pct']:.2f}%",
        f"- Replayed expectancy: {summary['replayed_expectancy_pct']:.3f}%",
        f"- Replayed profit factor: {summary['replayed_profit_factor']:.3f}",
        "",
        "## Reverse-engineered Selection Profile",
        "",
        "Values below use only the completed candle before each reported entry date.",
        "",
        "```",
        result.selection_profile.to_string(index=False),
        "```",
        "",
        "## Best Transparent Candidate",
        "",
        f"- Selected rule: {summary['selected_rule'] or 'None'}",
        f"- Rules tested: {summary['candidate_rules_tested']}",
        f"- Expansion ready: {summary['expansion_ready']}",
        f"- Gate: {summary['expansion_gate']}",
        "",
        "The rule always also requires close above SMA20, +DI above -DI, RSI(14) between 40 and 90,",
        "20-day range position between 20% and 130%, at least Rs 1 crore average traded value,",
        "and no likely corporate-action discontinuity in the prior 20 sessions.",
        "",
        "## Walk-forward Results",
        "",
    ]
    if result.split_metrics.empty:
        lines.append("No rule met the minimum research sample sizes.")
    else:
        lines.extend(["```", result.split_metrics.to_string(index=False), "```"])
    lines.extend(
        [
            "",
            "## Latest Cohort Candidates",
            "",
        f"As of the latest available cohort date: {summary['latest_cohort_date']}",
            "",
        ]
    )
    if result.latest_candidates.empty:
        lines.append("No current cohort symbol meets the selected rule.")
    else:
        lines.append(", ".join(result.latest_candidates["symbol"].astype(str)))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and model the attached Equity Runner cohort")
    parser.add_argument("--report", required=True, help="Path to the pasted Equity Runner report")
    parser.add_argument("--data-root", default="data", help="Local candle data root")
    parser.add_argument("--output", default="data/research/equity_runner", help="Output directory")
    parser.add_argument(
        "--fixed-stop",
        action="store_true",
        help="Use the original fixed 5%% stop instead of a 5%% trailing stop",
    )
    args = parser.parse_args()
    result = run_equity_runner_research(
        Storage(Path(args.data_root)),
        args.report,
        trailing_stop=not args.fixed_stop,
    )
    paths = save_equity_runner_research(result, args.output)
    print(pd.Series(result.summary).to_string())
    print("\nWalk-forward metrics")
    print(result.split_metrics.to_string(index=False))
    print("\nLatest candidates")
    print(",".join(result.latest_candidates.get("symbol", pd.Series(dtype=str)).astype(str)))
    print(f"\nReport: {paths['report']}")


if __name__ == "__main__":
    main()
