from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.strategy.technical_ratings import _adx


BASELINE_PARAMETER_NAME = "K100_R14_M20_E100_P14"


def run_knox_alpha_filter_research(
    storage: Storage,
    backtest_dir: Path,
    *,
    exchange: str = "NSE",
    parameter_name: str = BASELINE_PARAMETER_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = _load_positions(backtest_dir, parameter_name)
    if positions.empty:
        return pd.DataFrame(), pd.DataFrame()

    benchmark = _feature_frame(storage.load_candles("NSE_INDEX", "NIFTY 50", "1D"))
    benchmark = benchmark[
        ["date", "momentum_12_1_pct", "above_sma200", "sma200_rising"]
    ].rename(
        columns={
            "momentum_12_1_pct": "benchmark_momentum_12_1_pct",
            "above_sma200": "benchmark_above_sma200",
            "sma200_rising": "benchmark_sma200_rising",
        }
    )

    feature_rows: list[pd.DataFrame] = []
    for symbol, group in positions.groupby("symbol"):
        daily = _feature_frame(storage.load_candles(exchange, str(symbol), "1D"))
        if daily.empty:
            continue
        dates = pd.to_datetime(group["entry_signal_date"], errors="coerce").dropna().unique()
        selected = daily.loc[daily["date"].isin(dates)].copy()
        if selected.empty:
            continue
        selected["symbol"] = symbol
        feature_rows.append(selected)
    features = pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame()
    if features.empty:
        return pd.DataFrame(), pd.DataFrame()

    enriched = positions.merge(
        features,
        left_on=["symbol", "entry_signal_date"],
        right_on=["symbol", "date"],
        how="left",
        validate="many_to_one",
    )
    enriched = pd.merge_asof(
        enriched.sort_values("entry_signal_date"),
        benchmark.sort_values("date"),
        left_on="entry_signal_date",
        right_on="date",
        direction="backward",
        suffixes=("", "_benchmark"),
    )
    enriched["relative_strength_12_1_pct"] = (
        enriched["momentum_12_1_pct"] - enriched["benchmark_momentum_12_1_pct"]
    )
    enriched["market_bullish"] = (
        enriched["benchmark_above_sma200"].fillna(False)
        & enriched["benchmark_sma200_rising"].fillna(False)
    )

    filters: dict[str, pd.Series] = {
        "No additional filter": pd.Series(True, index=enriched.index),
        "12-1 momentum > 0%": enriched["momentum_12_1_pct"] > 0.0,
        "12-1 relative strength > NIFTY": enriched["relative_strength_12_1_pct"] > 0.0,
        "Within 25% of 52-week high": enriched["high_52w_ratio"] >= 0.75,
        "Close above rising SMA200": enriched["above_sma200"] & enriched["sma200_rising"],
        "NIFTY above rising SMA200": enriched["market_bullish"],
        "OBV accumulating over 20 days": enriched["obv_accumulating"],
        "OBV above 13D SMA": enriched["obv_above_sma13"],
        "OBV crossed 13D SMA in last 5 days": enriched["obv_cross_sma13_recent"],
        "OBV above 13D SMA and accumulating": (
            enriched["obv_above_sma13"] & enriched["obv_accumulating"]
        ),
        "RVOL20 >= 1.2x": enriched["volume_ratio20"] >= 1.2,
        "RVOL20 >= 1.5x": enriched["volume_ratio20"] >= 1.5,
        "RVOL20 >= 2.0x": enriched["volume_ratio20"] >= 2.0,
        "RVOL20 >= 3.0x": enriched["volume_ratio20"] >= 3.0,
        "CMF20 > 0": enriched["cmf20"] > 0.0,
        "CMF20 > 0.10": enriched["cmf20"] > 0.10,
        "RVOL20 >= 3x and CMF20 > 0": (
            (enriched["volume_ratio20"] >= 3.0) & (enriched["cmf20"] > 0.0)
        ),
        "RVOL20 >= 1.2x and CMF20 > 0": (
            (enriched["volume_ratio20"] >= 1.2) & (enriched["cmf20"] > 0.0)
        ),
        "RVOL20 >= 1.5x and CMF20 > 0": (
            (enriched["volume_ratio20"] >= 1.5) & (enriched["cmf20"] > 0.0)
        ),
        "RVOL20 >= 2x and CMF20 > 0": (
            (enriched["volume_ratio20"] >= 2.0) & (enriched["cmf20"] > 0.0)
        ),
        "Volume <= 0.8x prior 20D": enriched["volume_ratio20"] <= 0.8,
        "+DI > -DI": enriched["di_plus"] > enriched["di_minus"],
        "ADX < 20": enriched["adx14"] < 20.0,
        "20D annualized volatility <= 40%": enriched["realized_vol20_pct"] <= 40.0,
        "RS > NIFTY and bullish market": (
            (enriched["relative_strength_12_1_pct"] > 0.0) & enriched["market_bullish"]
        ),
    }

    rows: list[dict[str, Any]] = []
    for name, mask in filters.items():
        selected = enriched.loc[mask.fillna(False)].copy()
        for cohort, cohort_frame in (
            ("VALIDATION_EXITS_2022_2023", selected.loc[selected["cohort"] == "VALIDATION"]),
            ("HOLDOUT_EXITS_2024_2026", selected.loc[selected["cohort"] == "TEST"]),
            ("ALL_MARKED", selected),
        ):
            rows.append({"filter": name, "cohort": cohort, **_metrics(cohort_frame)})
    stats = pd.DataFrame(rows)
    validation = stats.loc[stats["cohort"] == "VALIDATION_EXITS_2022_2023"].copy()
    baseline = validation.loc[validation["filter"] == "No additional filter"].iloc[0]
    validation["win_rate_uplift_pct"] = validation["win_rate_pct"] - baseline["win_rate_pct"]
    validation["target_uplift_pct"] = (
        validation["target_10_hit_rate_pct"] - baseline["target_10_hit_rate_pct"]
    )
    validation["median_return_uplift_pct"] = (
        validation["median_return_pct"] - baseline["median_return_pct"]
    )
    validation["retention_pct"] = validation["positions"] / baseline["positions"] * 100.0
    ranking = validation.sort_values(
        ["target_10_hit_rate_pct", "win_rate_pct", "median_return_pct", "positions"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return ranking, stats


def save_knox_alpha_filter_research(
    ranking: pd.DataFrame,
    stats: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ranking": output_dir / "filter_ranking.csv",
        "cohort_stats": output_dir / "cohort_stats.csv",
    }
    ranking.to_csv(paths["ranking"], index=False)
    stats.to_csv(paths["cohort_stats"], index=False)
    return paths


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["date"] = pd.to_datetime(result.get("date"), errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        result[column] = pd.to_numeric(result.get(column), errors="coerce")
    result = (
        result.dropna(subset=["date", "high", "low", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    close = result["close"]
    volume = result["volume"].fillna(0.0)
    sma200 = close.rolling(200, min_periods=200).mean()
    result["above_sma200"] = close > sma200
    result["sma200_rising"] = sma200 > sma200.shift(20)
    result["momentum_12_1_pct"] = (close.shift(21) / close.shift(252) - 1.0) * 100.0
    result["high_52w_ratio"] = close / result["high"].rolling(252, min_periods=252).max()
    prior_volume20 = volume.shift(1).rolling(20, min_periods=20).mean()
    result["volume_ratio20"] = volume / prior_volume20.replace(0.0, np.nan)
    candle_range = (result["high"] - result["low"]).replace(0.0, np.nan)
    money_flow_multiplier = (
        ((close - result["low"]) - (result["high"] - close)) / candle_range
    ).fillna(0.0)
    money_flow_volume = money_flow_multiplier * volume
    result["cmf20"] = (
        money_flow_volume.rolling(20, min_periods=20).sum()
        / volume.rolling(20, min_periods=20).sum().replace(0.0, np.nan)
    )
    direction = close.diff()
    obv_flow = pd.Series(0.0, index=result.index)
    obv_flow.loc[direction > 0] = volume.loc[direction > 0]
    obv_flow.loc[direction < 0] = -volume.loc[direction < 0]
    obv = obv_flow.cumsum()
    result["obv_accumulating"] = obv > obv.shift(20)
    obv_sma13 = obv.rolling(13, min_periods=13).mean()
    result["obv_above_sma13"] = obv > obv_sma13
    obv_cross_sma13 = (obv > obv_sma13) & (obv.shift(1) <= obv_sma13.shift(1))
    result["obv_cross_sma13_recent"] = (
        obv_cross_sma13.rolling(5, min_periods=1).max().fillna(0.0).astype(bool)
    )
    returns = close.pct_change(fill_method=None)
    result["realized_vol20_pct"] = returns.rolling(20, min_periods=20).std() * np.sqrt(252) * 100.0
    plus_di, minus_di, adx = _adx(result["high"], result["low"], close, 14, 14)
    result["di_plus"] = plus_di
    result["di_minus"] = minus_di
    result["adx14"] = adx
    return result


def _load_positions(backtest_dir: Path, parameter_name: str) -> pd.DataFrame:
    closed_path = backtest_dir / "baseline_trades.csv"
    open_path = backtest_dir / "open_positions.csv"
    closed = pd.read_csv(closed_path) if closed_path.exists() else pd.DataFrame()
    opened = pd.read_csv(open_path) if open_path.exists() else pd.DataFrame()
    if not closed.empty:
        closed = closed.loc[closed["parameter_name"] == parameter_name].copy()
        closed = closed.loc[_truthy(closed["data_quality_pass"])]
        closed["outcome_return_pct"] = pd.to_numeric(closed["net_return_pct"], errors="coerce")
        closed["outcome_bars"] = pd.to_numeric(closed["bars_held"], errors="coerce")
        exit_year = pd.to_datetime(closed["exit_date"], errors="coerce").dt.year
        closed["cohort"] = np.where(exit_year.between(2022, 2023), "VALIDATION", np.where(exit_year >= 2024, "TEST", "DEVELOPMENT"))
        closed["position_status"] = "CLOSED"
    if not opened.empty:
        opened = opened.loc[opened["parameter_name"] == parameter_name].copy()
        opened = opened.loc[_truthy(opened["data_quality_pass"])]
        opened["outcome_return_pct"] = pd.to_numeric(
            opened["unrealized_net_return_pct"], errors="coerce"
        )
        opened["outcome_bars"] = pd.to_numeric(opened["bars_open"], errors="coerce")
        opened["cohort"] = "OPEN"
        opened["position_status"] = "OPEN"
    columns = [
        "symbol",
        "parameter_name",
        "entry_signal_date",
        "entry_date",
        "outcome_return_pct",
        "outcome_bars",
        "target_10_hit",
        "bars_to_target",
        "cohort",
        "position_status",
    ]
    combined = pd.concat(
        [closed.reindex(columns=columns), opened.reindex(columns=columns)],
        ignore_index=True,
    )
    combined["entry_signal_date"] = pd.to_datetime(
        combined["entry_signal_date"], errors="coerce", format="mixed"
    )
    combined["target_10_hit"] = _truthy(combined["target_10_hit"])
    return combined.dropna(subset=["symbol", "entry_signal_date", "outcome_return_pct"])


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "positions": 0,
            "win_rate_pct": np.nan,
            "target_10_hit_rate_pct": np.nan,
            "median_return_pct": np.nan,
            "avg_return_pct": np.nan,
            "median_outcome_bars": np.nan,
            "target_within_20_rate_pct": np.nan,
        }
    returns = pd.to_numeric(frame["outcome_return_pct"], errors="coerce").dropna()
    bars_to_target = pd.to_numeric(frame["bars_to_target"], errors="coerce")
    return {
        "positions": int(len(returns)),
        "win_rate_pct": float((returns > 0.0).mean() * 100.0),
        "target_10_hit_rate_pct": float(frame["target_10_hit"].fillna(False).mean() * 100.0),
        "median_return_pct": float(returns.median()),
        "avg_return_pct": float(returns.mean()),
        "median_outcome_bars": float(pd.to_numeric(frame["outcome_bars"], errors="coerce").median()),
        "target_within_20_rate_pct": float(bars_to_target.le(20).fillna(False).mean() * 100.0),
    }


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
