from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from stock_screener.strategy.technical_ratings import _adx, _cci, _rsi


LONG_LABEL = 1
SHORT_LABEL = -1
NEUTRAL_LABEL = 0


@dataclass(frozen=True)
class LorentzianFeatureConfig:
    name: str
    param_a: int
    param_b: int = 1


@dataclass(frozen=True)
class LorentzianFilterSettings:
    use_volatility_filter: bool = True
    use_regime_filter: bool = True
    use_adx_filter: bool = False
    regime_threshold: float = -0.1
    adx_threshold: int = 20


@dataclass(frozen=True)
class LorentzianKernelSettings:
    use_kernel_filter: bool = True
    use_kernel_smoothing: bool = False
    lookback_window: int = 8
    relative_weighting: float = 8.0
    regression_level: int = 25
    lag: int = 2


@dataclass(frozen=True)
class LorentzianSettings:
    source: str = "close"
    neighbors_count: int = 8
    max_bars_back: int = 2000
    feature_count: int = 5
    color_compression: int = 1
    show_exits: bool = False
    use_dynamic_exits: bool = False
    use_worst_case: bool = False
    use_ema_filter: bool = False
    ema_period: int = 200
    use_sma_filter: bool = False
    sma_period: int = 200
    features: tuple[LorentzianFeatureConfig, ...] = field(
        default_factory=lambda: (
            LorentzianFeatureConfig("RSI", 14, 1),
            LorentzianFeatureConfig("WT", 10, 11),
            LorentzianFeatureConfig("CCI", 20, 1),
            LorentzianFeatureConfig("ADX", 20, 2),
            LorentzianFeatureConfig("RSI", 9, 1),
        )
    )
    filters: LorentzianFilterSettings = field(default_factory=LorentzianFilterSettings)
    kernel: LorentzianKernelSettings = field(default_factory=LorentzianKernelSettings)


def run_lorentzian_classification(
    candles: pd.DataFrame,
    settings: LorentzianSettings | None = None,
) -> pd.DataFrame:
    """Convert jdehorty's Lorentzian Classification Pine logic into Python.

    Notes:
    - The TradingView script depends on external Pine libraries (`MLExtensions`
      and `KernelFunctions`). Their exact source is not embedded in the user
      supplied script, so this Python port uses standard indicator-equivalent
      implementations that match the public function descriptions closely.
    - The ANN selection logic, label alignment, kernel gating, and trade stats
      flow follow the supplied Pine script directly.
    """

    cfg = settings or LorentzianSettings()
    frame = _prepare_candles(candles)
    if frame.empty:
        return _empty_lorentzian_frame(frame)

    source = _select_source(frame, cfg.source)
    hlc3 = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    ohlc4 = (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0
    frame["source"] = source

    feature_series = []
    for feature in cfg.features[: max(2, min(cfg.feature_count, len(cfg.features)))]:
        feature_series.append(
            _series_from(
                feature.name,
                close=frame["close"],
                high=frame["high"],
                low=frame["low"],
                hlc3=hlc3,
                param_a=feature.param_a,
                param_b=feature.param_b,
            )
        )

    while len(feature_series) < 5:
        feature_series.append(pd.Series(np.nan, index=frame.index, dtype="float64"))

    for idx, values in enumerate(feature_series, start=1):
        frame[f"f{idx}"] = values

    frame["label"] = np.select(
        [source.shift(4) < source, source.shift(4) > source],
        [LONG_LABEL, SHORT_LABEL],
        default=NEUTRAL_LABEL,
    ).astype(int)

    plus_di, minus_di, adx = _adx(frame["high"], frame["low"], frame["close"], 14, 14)
    frame["adx_14"] = adx
    frame["filter_volatility"] = _filter_volatility(frame["high"], frame["low"], frame["close"], cfg.filters)
    frame["filter_regime"] = _filter_regime(ohlc4, frame["high"], frame["low"], cfg.filters)
    frame["filter_adx"] = _filter_adx(frame["close"], frame["high"], frame["low"], cfg.filters)
    frame["filter_all"] = frame["filter_volatility"] & frame["filter_regime"] & frame["filter_adx"]

    if cfg.use_ema_filter:
        ema_filter = frame["close"].ewm(span=cfg.ema_period, adjust=False).mean()
        frame["ema_filter_value"] = ema_filter
        frame["is_ema_uptrend"] = frame["close"] > ema_filter
        frame["is_ema_downtrend"] = frame["close"] < ema_filter
    else:
        frame["ema_filter_value"] = np.nan
        frame["is_ema_uptrend"] = True
        frame["is_ema_downtrend"] = True

    if cfg.use_sma_filter:
        sma_filter = frame["close"].rolling(cfg.sma_period, min_periods=cfg.sma_period).mean()
        frame["sma_filter_value"] = sma_filter
        frame["is_sma_uptrend"] = frame["close"] > sma_filter
        frame["is_sma_downtrend"] = frame["close"] < sma_filter
    else:
        frame["sma_filter_value"] = np.nan
        frame["is_sma_uptrend"] = True
        frame["is_sma_downtrend"] = True

    max_bars_back_index = max(len(frame) - 1 - cfg.max_bars_back, 0) if len(frame) - 1 >= cfg.max_bars_back else 0
    frame["max_bars_back_index"] = max_bars_back_index

    feature_matrix = frame[[f"f{i}" for i in range(1, cfg.feature_count + 1)]].to_numpy(dtype=float, copy=False)
    labels = frame["label"].to_numpy(dtype=int, copy=False)

    predictions: list[float] = []
    signal_values: list[int] = []
    signal = NEUTRAL_LABEL

    for bar in range(len(frame)):
        prediction = 0.0
        if bar >= max_bars_back_index:
            prediction = _approximate_ann_prediction(feature_matrix, labels, bar, cfg)

        predictions.append(prediction)
        if prediction > 0 and bool(frame.at[bar, "filter_all"]):
            signal = LONG_LABEL
        elif prediction < 0 and bool(frame.at[bar, "filter_all"]):
            signal = SHORT_LABEL
        signal_values.append(signal)

    frame["prediction"] = predictions
    frame["signal"] = pd.Series(signal_values, index=frame.index, dtype="int64")
    frame["bars_held"] = _bars_held(frame["signal"])
    frame["is_held_four_bars"] = frame["bars_held"] == 4
    frame["is_held_less_than_four_bars"] = (frame["bars_held"] > 0) & (frame["bars_held"] < 4)
    frame["is_different_signal_type"] = frame["signal"].ne(frame["signal"].shift(1)).fillna(False)

    recent_changes = pd.concat([_shift_bool(frame["is_different_signal_type"], i) for i in (1, 2, 3)], axis=1)
    frame["is_early_signal_flip"] = frame["is_different_signal_type"] & recent_changes.any(axis=1)

    frame["is_buy_signal"] = (
        (frame["signal"] == LONG_LABEL) & frame["is_ema_uptrend"] & frame["is_sma_uptrend"]
    )
    frame["is_sell_signal"] = (
        (frame["signal"] == SHORT_LABEL) & frame["is_ema_downtrend"] & frame["is_sma_downtrend"]
    )
    frame["is_last_signal_buy"] = (
        frame["signal"].shift(4).eq(LONG_LABEL)
        & _shift_bool(frame["is_ema_uptrend"], 4)
        & _shift_bool(frame["is_sma_uptrend"], 4)
    )
    frame["is_last_signal_sell"] = (
        frame["signal"].shift(4).eq(SHORT_LABEL)
        & _shift_bool(frame["is_ema_downtrend"], 4)
        & _shift_bool(frame["is_sma_downtrend"], 4)
    )
    frame["is_new_buy_signal"] = frame["is_buy_signal"] & frame["is_different_signal_type"]
    frame["is_new_sell_signal"] = frame["is_sell_signal"] & frame["is_different_signal_type"]

    kernel_estimate = _rational_quadratic_kernel(source, cfg.kernel.lookback_window, cfg.kernel.relative_weighting, cfg.kernel.regression_level)
    gaussian_estimate = _gaussian_kernel(source, max(cfg.kernel.lookback_window - cfg.kernel.lag, 1), cfg.kernel.regression_level)
    frame["kernel_estimate"] = kernel_estimate
    frame["kernel_gaussian"] = gaussian_estimate

    frame["was_bearish_rate"] = kernel_estimate.shift(2) > kernel_estimate.shift(1)
    frame["was_bullish_rate"] = kernel_estimate.shift(2) < kernel_estimate.shift(1)
    frame["is_bearish_rate"] = kernel_estimate.shift(1) > kernel_estimate
    frame["is_bullish_rate"] = kernel_estimate.shift(1) < kernel_estimate
    frame["is_bearish_change"] = frame["is_bearish_rate"] & frame["was_bullish_rate"]
    frame["is_bullish_change"] = frame["is_bullish_rate"] & frame["was_bearish_rate"]
    frame["is_bullish_cross_alert"] = _crossover(gaussian_estimate, kernel_estimate)
    frame["is_bearish_cross_alert"] = _crossunder(gaussian_estimate, kernel_estimate)
    frame["is_bullish_smooth"] = gaussian_estimate >= kernel_estimate
    frame["is_bearish_smooth"] = gaussian_estimate <= kernel_estimate
    frame["alert_bullish"] = (
        frame["is_bullish_cross_alert"] if cfg.kernel.use_kernel_smoothing else frame["is_bullish_change"]
    )
    frame["alert_bearish"] = (
        frame["is_bearish_cross_alert"] if cfg.kernel.use_kernel_smoothing else frame["is_bearish_change"]
    )
    frame["is_bullish"] = (
        (frame["is_bullish_smooth"] if cfg.kernel.use_kernel_smoothing else frame["is_bullish_rate"])
        if cfg.kernel.use_kernel_filter
        else True
    )
    frame["is_bearish"] = (
        (frame["is_bearish_smooth"] if cfg.kernel.use_kernel_smoothing else frame["is_bearish_rate"])
        if cfg.kernel.use_kernel_filter
        else True
    )

    frame["start_long_trade"] = (
        frame["is_new_buy_signal"] & frame["is_bullish"] & frame["is_ema_uptrend"] & frame["is_sma_uptrend"]
    )
    frame["start_short_trade"] = (
        frame["is_new_sell_signal"] & frame["is_bearish"] & frame["is_ema_downtrend"] & frame["is_sma_downtrend"]
    )

    bars_since_start_long = _bars_since(frame["start_long_trade"])
    bars_since_start_short = _bars_since(frame["start_short_trade"])
    bars_since_alert_bullish = _bars_since(frame["alert_bullish"])
    bars_since_alert_bearish = _bars_since(frame["alert_bearish"])

    frame["is_valid_short_exit"] = bars_since_alert_bullish > bars_since_start_short
    frame["is_valid_long_exit"] = bars_since_alert_bearish > bars_since_start_long
    frame["end_long_trade_dynamic"] = frame["is_bearish_change"] & _shift_bool(frame["is_valid_long_exit"], 1)
    frame["end_short_trade_dynamic"] = frame["is_bullish_change"] & _shift_bool(frame["is_valid_short_exit"], 1)

    frame["end_long_trade_strict"] = (
        (
            (frame["is_held_four_bars"] & frame["is_last_signal_buy"])
            | (frame["is_held_less_than_four_bars"] & frame["is_new_sell_signal"] & frame["is_last_signal_buy"])
        )
        & _shift_bool(frame["start_long_trade"], 4)
    )
    frame["end_short_trade_strict"] = (
        (
            (frame["is_held_four_bars"] & frame["is_last_signal_sell"])
            | (frame["is_held_less_than_four_bars"] & frame["is_new_buy_signal"] & frame["is_last_signal_sell"])
        )
        & _shift_bool(frame["start_short_trade"], 4)
    )

    is_dynamic_exit_valid = not cfg.use_ema_filter and not cfg.use_sma_filter and not cfg.kernel.use_kernel_smoothing
    frame["end_long_trade"] = (
        frame["end_long_trade_dynamic"] if cfg.use_dynamic_exits and is_dynamic_exit_valid else frame["end_long_trade_strict"]
    )
    frame["end_short_trade"] = (
        frame["end_short_trade_dynamic"] if cfg.use_dynamic_exits and is_dynamic_exit_valid else frame["end_short_trade_strict"]
    )
    frame["market_price"] = np.where(
        cfg.use_worst_case,
        source,
        (frame["high"] + frame["low"] + frame["open"] + frame["open"]) / 4.0,
    )

    return frame


def lorentzian_trades(frame: pd.DataFrame, settings: LorentzianSettings | None = None) -> pd.DataFrame:
    cfg = settings or LorentzianSettings()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "side",
                "entry_date",
                "entry_price",
                "exit_date",
                "exit_price",
                "return_pct",
                "bars_in_trade",
                "had_early_signal_flip",
            ]
        )

    price_column = "market_price" if "market_price" in frame.columns else ("source" if cfg.use_worst_case and "source" in frame.columns else "close")

    trades: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None

    for idx, row in frame.iterrows():
        if active is None:
            if bool(row.get("start_long_trade", False)):
                active = {
                    "side": "LONG",
                    "entry_idx": idx,
                    "entry_date": row.get("date"),
                    "entry_price": float(row[price_column]),
                    "had_early_signal_flip": False,
                }
            elif bool(row.get("start_short_trade", False)):
                active = {
                    "side": "SHORT",
                    "entry_idx": idx,
                    "entry_date": row.get("date"),
                    "entry_price": float(row[price_column]),
                    "had_early_signal_flip": False,
                }
            continue

        if bool(row.get("is_early_signal_flip", False)):
            active["had_early_signal_flip"] = True

        should_exit = (
            active["side"] == "LONG" and bool(row.get("end_long_trade", False))
        ) or (
            active["side"] == "SHORT" and bool(row.get("end_short_trade", False))
        )

        if not should_exit:
            continue

        entry_price = float(active["entry_price"])
        exit_price = float(row[price_column])
        if active["side"] == "LONG":
            return_pct = ((exit_price - entry_price) / entry_price) * 100.0
        else:
            return_pct = ((entry_price - exit_price) / entry_price) * 100.0

        trades.append(
            {
                "side": active["side"],
                "entry_date": active["entry_date"],
                "entry_price": entry_price,
                "exit_date": row.get("date"),
                "exit_price": exit_price,
                "return_pct": return_pct,
                "bars_in_trade": idx - int(active["entry_idx"]),
                "had_early_signal_flip": bool(active["had_early_signal_flip"]),
            }
        )
        active = None

    return pd.DataFrame(trades)


def lorentzian_trade_stats(frame: pd.DataFrame, settings: LorentzianSettings | None = None) -> dict[str, Any]:
    cfg = settings or LorentzianSettings()
    trades = lorentzian_trades(frame, cfg)

    if frame.empty:
        return {
            "total_wins": 0,
            "total_losses": 0,
            "total_early_signal_flips": 0,
            "total_trades": 0,
            "trade_stats_header": "📈 Trade Stats",
            "win_loss_ratio": 0.0,
            "wins_over_losses_ratio": 0.0,
            "win_rate_raw": 0.0,
            "win_rate": 0.0,
            "trades": trades,
        }

    if "market_price" in frame.columns:
        market_price = pd.to_numeric(frame["market_price"], errors="coerce")
    elif cfg.use_worst_case and "source" in frame.columns:
        market_price = pd.to_numeric(frame["source"], errors="coerce")
    elif {"high", "low", "open"}.issubset(frame.columns):
        market_price = (
            pd.to_numeric(frame["high"], errors="coerce")
            + pd.to_numeric(frame["low"], errors="coerce")
            + pd.to_numeric(frame["open"], errors="coerce")
            + pd.to_numeric(frame["open"], errors="coerce")
        ) / 4.0
    else:
        market_price = pd.to_numeric(frame.get("close"), errors="coerce")

    if "max_bars_back_index" in frame.columns and not frame["max_bars_back_index"].empty:
        max_bars_back_index = int(pd.to_numeric(frame["max_bars_back_index"], errors="coerce").fillna(0).iloc[0])
    else:
        max_bars_back_index = 0

    start_long_trade_price = float(market_price.iloc[0]) if len(market_price) else 0.0
    start_short_trade_price = float(market_price.iloc[0]) if len(market_price) else 0.0
    total_wins = 0
    total_losses = 0
    total_early_signal_flips = 0

    for idx, row in frame.iterrows():
        if idx <= max_bars_back_index:
            continue

        price = float(market_price.iloc[idx]) if pd.notna(market_price.iloc[idx]) else 0.0
        start_long = bool(row.get("start_long_trade", False))
        end_long = bool(row.get("end_long_trade", False))
        start_short = bool(row.get("start_short_trade", False))
        end_short = bool(row.get("end_short_trade", False))
        early_flip = bool(row.get("is_early_signal_flip", False))

        if start_long:
            start_short_trade_price = 0.0
            total_early_signal_flips += 1 if early_flip else 0
            start_long_trade_price = price

        if end_long:
            delta = price - start_long_trade_price
            total_wins += 1 if delta > 0 else 0
            total_losses += 1 if delta < 0 else 0

        if start_short:
            start_long_trade_price = 0.0
            start_short_trade_price = price

        if end_short:
            total_early_signal_flips += 1 if early_flip else 0
            delta = start_short_trade_price - price
            total_wins += 1 if delta > 0 else 0
            total_losses += 1 if delta < 0 else 0

    total_trades = total_wins + total_losses
    pine_win_loss_ratio = (total_wins / total_trades) if total_trades else 0.0
    pine_win_rate_raw = (total_wins / (total_wins + total_losses)) if (total_wins + total_losses) else 0.0

    return {
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_early_signal_flips": total_early_signal_flips,
        "total_trades": total_trades,
        "trade_stats_header": "📈 Trade Stats",
        "win_loss_ratio": pine_win_loss_ratio,
        "wins_over_losses_ratio": (total_wins / total_losses) if total_losses else np.inf if total_wins else 0.0,
        "win_rate_raw": pine_win_rate_raw,
        "win_rate": pine_win_rate_raw * 100.0,
        "trades": trades,
    }


def _prepare_candles(candles: pd.DataFrame) -> pd.DataFrame:
    if candles.empty:
        return candles.copy()

    frame = candles.copy()
    required = {"open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required candle columns: {sorted(missing)}")

    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values("date")
    else:
        frame = frame.reset_index().rename(columns={"index": "date"})
    frame = frame.reset_index(drop=True)

    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _select_source(frame: pd.DataFrame, source_name: str) -> pd.Series:
    normalized = str(source_name).strip().lower()
    if normalized == "hlc3":
        return (frame["high"] + frame["low"] + frame["close"]) / 3.0
    if normalized == "ohlc4":
        return (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0
    if normalized in frame.columns:
        return pd.to_numeric(frame[normalized], errors="coerce")
    return pd.to_numeric(frame["close"], errors="coerce")


def _series_from(
    feature_name: str,
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    hlc3: pd.Series,
    param_a: int,
    param_b: int,
) -> pd.Series:
    normalized = str(feature_name).upper()
    if normalized == "RSI":
        rsi = _rsi(close, max(param_a, 1)).ewm(span=max(param_b, 1), adjust=False).mean()
        return _rescale_bounded(rsi, 0.0, 100.0, 0.0, 1.0)
    if normalized == "CCI":
        cci = _cci(high, low, close, max(param_a, 1)).ewm(span=max(param_b, 1), adjust=False).mean()
        return _normalize_unbounded(cci, 0.0, 1.0)
    if normalized == "ADX":
        adx = _n_adx(high, low, close, max(param_a, 1))
        return adx
    if normalized == "WT":
        wt = _wavetrend(hlc3, max(param_a, 1), max(param_b, 1))
        return _normalize_unbounded(wt, 0.0, 1.0)
    raise ValueError(f"Unsupported feature: {feature_name}")


def _rescale_bounded(src: pd.Series, old_min: float, old_max: float, new_min: float, new_max: float) -> pd.Series:
    denominator = old_max - old_min
    if denominator == 0:
        return pd.Series(np.nan, index=src.index, dtype="float64")
    return ((src - old_min) * (new_max - new_min) / denominator) + new_min


def _normalize_unbounded(src: pd.Series, target_low: float = 0.0, target_high: float = 100.0) -> pd.Series:
    historic_min = src.expanding(min_periods=1).min()
    historic_max = src.expanding(min_periods=1).max()
    denominator = (historic_max - historic_min).clip(lower=1e-10)
    return target_low + (target_high - target_low) * (src - historic_min) / denominator


def _wavetrend(src: pd.Series, channel_length: int, average_length: int) -> pd.Series:
    esa = src.ewm(span=channel_length, adjust=False).mean()
    deviation = (src - esa).abs().ewm(span=channel_length, adjust=False).mean()
    ci = (src - esa) / (0.015 * deviation.replace(0, np.nan))
    wt1 = ci.ewm(span=average_length, adjust=False).mean()
    wt2 = wt1.rolling(4, min_periods=1).mean()
    return wt1 - wt2


def _n_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    directional_movement_plus = np.where((high - prev_high) > (prev_low - low), np.maximum(high - prev_high, 0.0), 0.0)
    neg_movement = np.where((prev_low - low) > (high - prev_high), np.maximum(prev_low - low, 0.0), 0.0)

    tr_smooth = _pine_recursive_smooth(pd.Series(tr, index=close.index), length)
    plus_smooth = _pine_recursive_smooth(pd.Series(directional_movement_plus, index=close.index), length)
    neg_smooth = _pine_recursive_smooth(pd.Series(neg_movement, index=close.index), length)

    di_positive = plus_smooth / tr_smooth.replace(0, np.nan) * 100.0
    di_negative = neg_smooth / tr_smooth.replace(0, np.nan) * 100.0
    dx = (di_positive - di_negative).abs() / (di_positive + di_negative).replace(0, np.nan) * 100.0
    adx = dx.ewm(alpha=1 / length, adjust=False).mean()
    return _rescale_bounded(adx, 0.0, 100.0, 0.0, 1.0)


def _approximate_ann_prediction(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    bar: int,
    settings: LorentzianSettings,
) -> float:
    distances: list[float] = []
    predictions: list[int] = []
    last_distance = -1.0
    size_loop = min(settings.max_bars_back - 1, bar)
    candidate_indices = np.arange(size_loop + 1, dtype=int)
    candidate_indices = candidate_indices[candidate_indices % 4 != 0]
    if candidate_indices.size == 0:
        return 0.0

    current_values = feature_matrix[bar]
    if not np.isfinite(current_values).all():
        return 0.0

    historical_values = feature_matrix[candidate_indices]
    valid_mask = np.isfinite(historical_values).all(axis=1)
    if not valid_mask.any():
        return 0.0

    candidate_indices = candidate_indices[valid_mask]
    historical_values = historical_values[valid_mask]
    distances_array = np.log1p(np.abs(historical_values - current_values)).sum(axis=1)

    for index, distance in zip(candidate_indices, distances_array):
        if distance >= last_distance:
            last_distance = float(distance)
            distances.append(last_distance)
            predictions.append(int(labels[index]))
            if len(predictions) > settings.neighbors_count:
                quartile_index = int(round(settings.neighbors_count * 3 / 4))
                quartile_index = min(quartile_index, len(distances) - 1)
                last_distance = float(distances[quartile_index])
                distances.pop(0)
                predictions.pop(0)

    return float(sum(predictions))


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    return _true_range(high, low, close).rolling(length, min_periods=length).mean()


def _filter_volatility(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    settings: LorentzianFilterSettings,
) -> pd.Series:
    if not settings.use_volatility_filter:
        return pd.Series(True, index=close.index, dtype="bool")
    short_atr = _atr(high, low, close, 1)
    long_atr = _atr(high, low, close, 10)
    return (short_atr > long_atr).fillna(False)


def _filter_regime(
    src: pd.Series,
    high: pd.Series,
    low: pd.Series,
    settings: LorentzianFilterSettings,
) -> pd.Series:
    if not settings.use_regime_filter:
        return pd.Series(True, index=src.index, dtype="bool")

    value1 = pd.Series(0.0, index=src.index, dtype="float64")
    value2 = pd.Series(0.0, index=src.index, dtype="float64")
    klmf = pd.Series(0.0, index=src.index, dtype="float64")

    for i in range(len(src)):
        src_prev = float(src.iloc[i - 1]) if i > 0 and pd.notna(src.iloc[i - 1]) else float(src.iloc[i])
        high_i = float(high.iloc[i])
        low_i = float(low.iloc[i])
        value1_prev = float(value1.iloc[i - 1]) if i > 0 else 0.0
        value2_prev = float(value2.iloc[i - 1]) if i > 0 else 0.0
        klmf_prev = float(klmf.iloc[i - 1]) if i > 0 else 0.0

        value1.iloc[i] = 0.2 * (float(src.iloc[i]) - src_prev) + 0.8 * value1_prev
        value2.iloc[i] = 0.1 * (high_i - low_i) + 0.8 * value2_prev
        denominator = value2.iloc[i]
        omega = abs(value1.iloc[i] / denominator) if denominator != 0 else 0.0
        alpha = (-omega**2 + np.sqrt(omega**4 + 16 * omega**2)) / 8 if omega != 0 else 0.0
        klmf.iloc[i] = alpha * float(src.iloc[i]) + (1 - alpha) * klmf_prev

    abs_curve_slope = (klmf - klmf.shift(1)).abs()
    ema_abs_curve_slope = abs_curve_slope.ewm(span=200, adjust=False).mean()
    normalized_slope_decline = (abs_curve_slope - ema_abs_curve_slope) / ema_abs_curve_slope.replace(0, np.nan)
    return (normalized_slope_decline >= settings.regime_threshold).fillna(False)


def _filter_adx(src: pd.Series, high: pd.Series, low: pd.Series, settings: LorentzianFilterSettings) -> pd.Series:
    if not settings.use_adx_filter:
        return pd.Series(True, index=src.index, dtype="bool")

    length = 14
    prev_src = src.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_src).abs(),
            (low - prev_src).abs(),
        ],
        axis=1,
    ).max(axis=1)
    directional_movement_plus = np.where((high - prev_high) > (prev_low - low), np.maximum(high - prev_high, 0.0), 0.0)
    neg_movement = np.where((prev_low - low) > (high - prev_high), np.maximum(prev_low - low, 0.0), 0.0)

    tr_smooth = _pine_recursive_smooth(pd.Series(tr, index=src.index), length)
    plus_smooth = _pine_recursive_smooth(pd.Series(directional_movement_plus, index=src.index), length)
    neg_smooth = _pine_recursive_smooth(pd.Series(neg_movement, index=src.index), length)
    di_positive = plus_smooth / tr_smooth.replace(0, np.nan) * 100.0
    di_negative = neg_smooth / tr_smooth.replace(0, np.nan) * 100.0
    dx = (di_positive - di_negative).abs() / (di_positive + di_negative).replace(0, np.nan) * 100.0
    adx = dx.ewm(alpha=1 / length, adjust=False).mean()
    return (adx > settings.adx_threshold).fillna(False)


def _rational_quadratic_kernel(src: pd.Series, lookback: int, relative_weight: float, start_at_bar: int) -> pd.Series:
    return _kernel_regression(src, lookback, start_at_bar, kernel="rq", relative_weight=relative_weight)


def _gaussian_kernel(src: pd.Series, lookback: int, start_at_bar: int) -> pd.Series:
    return _kernel_regression(src, lookback, start_at_bar, kernel="gaussian", relative_weight=1.0)


def _kernel_regression(
    src: pd.Series,
    lookback: int,
    start_at_bar: int,
    *,
    kernel: str,
    relative_weight: float,
) -> pd.Series:
    values = src.to_numpy(dtype=float)
    output = np.full(len(src), np.nan, dtype=float)
    lookback = max(int(lookback), 1)
    alpha = max(float(relative_weight), 1e-6)

    max_offset = max(int(start_at_bar) + 1, 0)
    for idx in range(len(values)):
        if idx < start_at_bar:
            continue
        current_weight = 0.0
        cumulative_weight = 0.0
        for offset in range(0, max_offset + 1):
            hist_idx = idx - offset
            if hist_idx < 0:
                continue
            y = values[hist_idx]
            if np.isnan(y):
                continue
            if kernel == "gaussian":
                w = np.exp(-(offset**2) / (2.0 * (lookback**2)))
            else:
                w = (1.0 + ((offset**2) / (((lookback**2) * 2.0 * alpha)))) ** (-alpha)
            current_weight += y * w
            cumulative_weight += w
        output[idx] = current_weight / cumulative_weight if cumulative_weight else np.nan

    return pd.Series(output, index=src.index, dtype="float64")


def _pine_recursive_smooth(series: pd.Series, length: int) -> pd.Series:
    output = pd.Series(0.0, index=series.index, dtype="float64")
    for i in range(len(series)):
        prev = float(output.iloc[i - 1]) if i > 0 else 0.0
        current = float(series.iloc[i]) if pd.notna(series.iloc[i]) else 0.0
        output.iloc[i] = prev - prev / length + current
    return output


def _crossover(left: pd.Series, right: pd.Series) -> pd.Series:
    return ((left > right) & (left.shift(1) <= right.shift(1))).fillna(False)


def _crossunder(left: pd.Series, right: pd.Series) -> pd.Series:
    return ((left < right) & (left.shift(1) >= right.shift(1))).fillna(False)


def _bars_held(signal: pd.Series) -> pd.Series:
    held: list[int] = []
    current = 0
    prev = NEUTRAL_LABEL
    for value in signal.fillna(NEUTRAL_LABEL).astype(int):
        if value != prev:
            current = 0
        else:
            current += 1
        held.append(current)
        prev = value
    return pd.Series(held, index=signal.index, dtype="int64")


def _bars_since(condition: pd.Series) -> pd.Series:
    bars: list[float] = []
    last_true: int | None = None
    for idx, flag in enumerate(condition.fillna(False).astype(bool)):
        if flag:
            last_true = idx
            bars.append(0.0)
        elif last_true is None:
            bars.append(np.inf)
        else:
            bars.append(float(idx - last_true))
    return pd.Series(bars, index=condition.index, dtype="float64")


def _shift_bool(series: pd.Series, periods: int) -> pd.Series:
    shifted = series.astype("boolean").shift(periods)
    return shifted.fillna(False).astype(bool)


def _empty_lorentzian_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    defaults = {
        "prediction": pd.Series(dtype="float64"),
        "signal": pd.Series(dtype="int64"),
        "start_long_trade": pd.Series(dtype="bool"),
        "start_short_trade": pd.Series(dtype="bool"),
        "end_long_trade": pd.Series(dtype="bool"),
        "end_short_trade": pd.Series(dtype="bool"),
        "is_early_signal_flip": pd.Series(dtype="bool"),
    }
    for column, value in defaults.items():
        if column not in output.columns:
            output[column] = value
    return output
