from __future__ import annotations

from typing import Any

import pandas as pd


STRONG_BOUND = 0.5
WEAK_BOUND = 0.1
MA_PERIODS = (10, 20, 30, 50, 100, 200)
OSCILLATOR_SIGNAL_COLUMNS = (
    "rating_rsi_14",
    "rating_stoch_14_3_3",
    "rating_cci_20",
    "rating_adx_14_14",
    "rating_ao",
    "rating_mom_10",
    "rating_macd_12_26_9",
    "rating_stoch_rsi_14_14_3_3",
    "rating_williams_r_14",
    "rating_bull_bear_power_50",
    "rating_uo_7_14_28",
)
MA_SIGNAL_COLUMNS = tuple(
    [f"rating_sma_{period}" for period in MA_PERIODS]
    + [f"rating_ema_{period}" for period in MA_PERIODS]
    + ["rating_hma_9", "rating_vwma_20", "rating_ichimoku_9_26_52"]
)


def rating_status(
    value: float | int | pd.NA | None,
    strong_bound: float = STRONG_BOUND,
    weak_bound: float = WEAK_BOUND,
) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if value > strong_bound:
        return "Strong Buy"
    if value > weak_bound:
        return "Buy"
    if value < -strong_bound:
        return "Strong Sell"
    if value < -weak_bound:
        return "Sell"
    return "Neutral"


def compute_technical_ratings(candles: pd.DataFrame) -> pd.DataFrame:
    """Compute TradingView-style technical ratings from OHLCV candles.

    This follows TradingView's documented 26-indicator basket:
    15 MA-based signals and 11 oscillator-based signals.

    TradingView's Help Center documents the indicator conditions and the
    TechnicalRating library documents the current ensemble composition.
    For the "uptrend/downtrend" checks used by Stochastic RSI and Bull Bear
    Power, TradingView does not document a precise standalone rule in the
    Help Center. This implementation uses the close relative to EMA(50),
    which aligns with the library's Bull Bear Power length of 50 and gives a
    stable trend filter for both indicators.
    """

    if candles.empty:
        return candles.copy()

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(candles.columns)
    if missing:
        raise ValueError(f"Missing required candle columns: {sorted(missing)}")

    frame = candles.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values("date" if "date" in frame.columns else frame.index.name or frame.columns[0]).reset_index(
        drop=True
    )

    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")

    prev_close = close.shift(1)

    for period in MA_PERIODS:
        frame[f"sma_{period}"] = close.rolling(period, min_periods=period).mean()
        frame[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

    frame["hma_9"] = _hma(close, 9)
    frame["vwma_20"] = _vwma(close, volume, 20)

    conversion_line = (_rolling_high(high, 9) + _rolling_low(low, 9)) / 2
    base_line = (_rolling_high(high, 26) + _rolling_low(low, 26)) / 2
    leading_span_a = (conversion_line + base_line) / 2
    leading_span_b = (_rolling_high(high, 52) + _rolling_low(low, 52)) / 2

    frame["ichimoku_conversion_9"] = conversion_line
    frame["ichimoku_base_26"] = base_line
    frame["ichimoku_span_a"] = leading_span_a
    frame["ichimoku_span_b"] = leading_span_b

    frame["rsi_14"] = _rsi(close, 14)
    frame["stoch_k_14_3_3"], frame["stoch_d_14_3_3"] = _stochastic(high, low, close, 14, 3, 3)
    frame["cci_20"] = _cci(high, low, close, 20)
    frame["plus_di_14"], frame["minus_di_14"], frame["adx_14_14"] = _adx(high, low, close, 14, 14)
    frame["ao"] = _ao(high, low)
    frame["mom_10"] = close - close.shift(10)
    frame["macd_12_26_9"], frame["macd_signal_12_26_9"] = _macd(close, 12, 26, 9)
    frame["stoch_rsi_k_14_14_3_3"], frame["stoch_rsi_d_14_14_3_3"] = _stoch_rsi(close, 14, 14, 3, 3)
    frame["williams_r_14"] = _williams_r(high, low, close, 14)

    bull_bear_ema = frame["ema_50"]
    frame["bull_power_50"] = high - bull_bear_ema
    frame["bear_power_50"] = low - bull_bear_ema
    frame["uo_7_14_28"] = _ultimate_oscillator(high, low, close, 7, 14, 28)

    trend_up = close > frame["ema_50"]
    trend_down = close < frame["ema_50"]

    for period in MA_PERIODS:
        frame[f"rating_sma_{period}"] = _ma_rating(close, frame[f"sma_{period}"])
        frame[f"rating_ema_{period}"] = _ma_rating(close, frame[f"ema_{period}"])

    frame["rating_hma_9"] = _ma_rating(close, frame["hma_9"])
    frame["rating_vwma_20"] = _ma_rating(close, frame["vwma_20"])
    frame["rating_ichimoku_9_26_52"] = _ichimoku_rating(close, conversion_line, base_line, leading_span_a, leading_span_b)

    frame["rating_rsi_14"] = _three_way_rating(
        (frame["rsi_14"] < 30) & (frame["rsi_14"] > frame["rsi_14"].shift(1)),
        (frame["rsi_14"] > 70) & (frame["rsi_14"] < frame["rsi_14"].shift(1)),
        frame["rsi_14"].isna() | frame["rsi_14"].shift(1).isna(),
    )
    frame["rating_stoch_14_3_3"] = _three_way_rating(
        (frame["stoch_k_14_3_3"] < 20) & (frame["stoch_d_14_3_3"] < 20) & (frame["stoch_k_14_3_3"] > frame["stoch_d_14_3_3"]),
        (frame["stoch_k_14_3_3"] > 80) & (frame["stoch_d_14_3_3"] > 80) & (frame["stoch_k_14_3_3"] < frame["stoch_d_14_3_3"]),
        frame["stoch_k_14_3_3"].isna() | frame["stoch_d_14_3_3"].isna(),
    )
    frame["rating_cci_20"] = _three_way_rating(
        (frame["cci_20"] < -100) & (frame["cci_20"] > frame["cci_20"].shift(1)),
        (frame["cci_20"] > 100) & (frame["cci_20"] < frame["cci_20"].shift(1)),
        frame["cci_20"].isna() | frame["cci_20"].shift(1).isna(),
    )
    frame["rating_adx_14_14"] = _three_way_rating(
        (frame["plus_di_14"] > frame["minus_di_14"]) & (frame["adx_14_14"] > 20) & (frame["adx_14_14"] > frame["adx_14_14"].shift(1)),
        (frame["plus_di_14"] < frame["minus_di_14"]) & (frame["adx_14_14"] > 20) & (frame["adx_14_14"] < frame["adx_14_14"].shift(1)),
        frame["plus_di_14"].isna() | frame["minus_di_14"].isna() | frame["adx_14_14"].isna() | frame["adx_14_14"].shift(1).isna(),
    )

    ao_cross_up = (frame["ao"] > 0) & (frame["ao"].shift(1) <= 0)
    ao_cross_down = (frame["ao"] < 0) & (frame["ao"].shift(1) >= 0)
    ao_turn_up = (
        (frame["ao"] > 0)
        & (frame["ao"].shift(1) > 0)
        & (frame["ao"] > frame["ao"].shift(1))
        & (frame["ao"].shift(1) < frame["ao"].shift(2))
    )
    ao_turn_down = (
        (frame["ao"] < 0)
        & (frame["ao"].shift(1) < 0)
        & (frame["ao"] < frame["ao"].shift(1))
        & (frame["ao"].shift(1) > frame["ao"].shift(2))
    )
    frame["rating_ao"] = _three_way_rating(
        ao_cross_up | ao_turn_up,
        ao_cross_down | ao_turn_down,
        frame["ao"].isna() | frame["ao"].shift(1).isna() | frame["ao"].shift(2).isna(),
    )

    frame["rating_mom_10"] = _three_way_rating(
        frame["mom_10"] > frame["mom_10"].shift(1),
        frame["mom_10"] < frame["mom_10"].shift(1),
        frame["mom_10"].isna() | frame["mom_10"].shift(1).isna(),
    )
    frame["rating_macd_12_26_9"] = _three_way_rating(
        frame["macd_12_26_9"] > frame["macd_signal_12_26_9"],
        frame["macd_12_26_9"] < frame["macd_signal_12_26_9"],
        frame["macd_12_26_9"].isna() | frame["macd_signal_12_26_9"].isna(),
    )
    frame["rating_stoch_rsi_14_14_3_3"] = _three_way_rating(
        trend_down
        & (frame["stoch_rsi_k_14_14_3_3"] < 20)
        & (frame["stoch_rsi_d_14_14_3_3"] < 20)
        & (frame["stoch_rsi_k_14_14_3_3"] > frame["stoch_rsi_d_14_14_3_3"]),
        trend_up
        & (frame["stoch_rsi_k_14_14_3_3"] > 80)
        & (frame["stoch_rsi_d_14_14_3_3"] > 80)
        & (frame["stoch_rsi_k_14_14_3_3"] < frame["stoch_rsi_d_14_14_3_3"]),
        frame["stoch_rsi_k_14_14_3_3"].isna()
        | frame["stoch_rsi_d_14_14_3_3"].isna()
        | frame["ema_50"].isna(),
    )
    frame["rating_williams_r_14"] = _three_way_rating(
        (frame["williams_r_14"] < -80) & (frame["williams_r_14"] > frame["williams_r_14"].shift(1)),
        (frame["williams_r_14"] > -20) & (frame["williams_r_14"] < frame["williams_r_14"].shift(1)),
        frame["williams_r_14"].isna() | frame["williams_r_14"].shift(1).isna(),
    )
    frame["rating_bull_bear_power_50"] = _three_way_rating(
        trend_up & (frame["bear_power_50"] < 0) & (frame["bear_power_50"] > frame["bear_power_50"].shift(1)),
        trend_down & (frame["bull_power_50"] > 0) & (frame["bull_power_50"] < frame["bull_power_50"].shift(1)),
        frame["bear_power_50"].isna() | frame["bear_power_50"].shift(1).isna() | frame["bull_power_50"].isna() | frame["bull_power_50"].shift(1).isna() | frame["ema_50"].isna(),
    )
    frame["rating_uo_7_14_28"] = _three_way_rating(
        frame["uo_7_14_28"] > 70,
        frame["uo_7_14_28"] < 30,
        frame["uo_7_14_28"].isna(),
    )

    frame["ma_rating"] = frame.loc[:, MA_SIGNAL_COLUMNS].mean(axis=1, skipna=True)
    frame["oscillator_rating"] = frame.loc[:, OSCILLATOR_SIGNAL_COLUMNS].mean(axis=1, skipna=True)
    frame["rating"] = frame[["ma_rating", "oscillator_rating"]].mean(axis=1, skipna=True)
    frame["ma_rating_status"] = frame["ma_rating"].apply(rating_status)
    frame["oscillator_rating_status"] = frame["oscillator_rating"].apply(rating_status)
    frame["rating_status"] = frame["rating"].apply(rating_status)
    frame["ma_indicator_count"] = frame.loc[:, MA_SIGNAL_COLUMNS].notna().sum(axis=1)
    frame["oscillator_indicator_count"] = frame.loc[:, OSCILLATOR_SIGNAL_COLUMNS].notna().sum(axis=1)

    return frame


def latest_technical_rating(candles: pd.DataFrame) -> dict[str, Any]:
    ratings = compute_technical_ratings(candles)
    if ratings.empty:
        return {}
    latest = ratings.iloc[-1]
    return {
        "date": latest.get("date"),
        "rating": latest.get("rating"),
        "rating_status": latest.get("rating_status"),
        "ma_rating": latest.get("ma_rating"),
        "ma_rating_status": latest.get("ma_rating_status"),
        "oscillator_rating": latest.get("oscillator_rating"),
        "oscillator_rating_status": latest.get("oscillator_rating_status"),
    }


def _three_way_rating(
    buy_condition: pd.Series,
    sell_condition: pd.Series,
    invalid_mask: pd.Series | None = None,
) -> pd.Series:
    ratings = pd.Series(0, index=buy_condition.index, dtype="float64")
    ratings.loc[buy_condition.fillna(False)] = 1.0
    ratings.loc[sell_condition.fillna(False)] = -1.0
    null_mask = buy_condition.isna() | sell_condition.isna()
    if invalid_mask is not None:
        null_mask = null_mask | invalid_mask.fillna(True)
    ratings.loc[null_mask] = pd.NA
    both = buy_condition.fillna(False) & sell_condition.fillna(False)
    ratings.loc[both] = 0.0
    return ratings


def _ma_rating(price: pd.Series, moving_average: pd.Series) -> pd.Series:
    return _three_way_rating(price > moving_average, price < moving_average, price.isna() | moving_average.isna())


def _ichimoku_rating(
    price: pd.Series,
    conversion_line: pd.Series,
    base_line: pd.Series,
    leading_span_a: pd.Series,
    leading_span_b: pd.Series,
) -> pd.Series:
    invalid_mask = price.isna() | conversion_line.isna() | base_line.isna() | leading_span_a.isna() | leading_span_b.isna()
    buy_condition = (
        (leading_span_a > leading_span_b)
        & (base_line > leading_span_a)
        & (conversion_line > base_line)
        & (price > conversion_line)
    )
    sell_condition = (
        (leading_span_a < leading_span_b)
        & (base_line < leading_span_a)
        & (conversion_line < base_line)
        & (price < conversion_line)
    )
    return _three_way_rating(buy_condition, sell_condition, invalid_mask)


def _rolling_high(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).max()


def _rolling_low(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).min()


def _vwma(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    weighted = (close * volume).rolling(length, min_periods=length).sum()
    vol_sum = volume.rolling(length, min_periods=length).sum()
    return weighted / vol_sum


def _wma(series: pd.Series, length: int) -> pd.Series:
    weights = pd.Series(range(1, length + 1), dtype="float64")
    weight_sum = float(weights.sum())
    return series.rolling(length, min_periods=length).apply(
        lambda values: float((pd.Series(values) * weights).sum()) / weight_sum,
        raw=False,
    )


def _hma(series: pd.Series, length: int) -> pd.Series:
    half_length = max(length // 2, 1)
    sqrt_length = max(int(length**0.5), 1)
    raw = (2 * _wma(series, half_length)) - _wma(series, length)
    return _wma(raw, sqrt_length)


def _rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = _rma(gains, length)
    avg_loss = _rma(losses, length)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask(avg_loss == 0, 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return rsi


def _stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_length: int,
    smooth_k: int,
    d_length: int,
) -> tuple[pd.Series, pd.Series]:
    lowest_low = _rolling_low(low, k_length)
    highest_high = _rolling_high(high, k_length)
    range_ = highest_high - lowest_low
    raw_k = 100 * (close - lowest_low) / range_
    raw_k = raw_k.where(range_ != 0, 0.0)
    k = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(d_length, min_periods=d_length).mean()
    return k, d


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(length, min_periods=length).mean()
    mad = typical_price.rolling(length, min_periods=length).apply(
        lambda values: float(pd.Series(values).sub(pd.Series(values).mean()).abs().mean()),
        raw=False,
    )
    denominator = 0.015 * mad
    cci = (typical_price - sma) / denominator
    return cci.where(denominator != 0)


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    di_length: int,
    adx_smoothing: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = _rma(true_range, di_length)
    plus_di = 100 * _rma(plus_dm, di_length) / atr
    minus_di = 100 * _rma(minus_dm, di_length) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = _rma(dx, adx_smoothing)
    return plus_di, minus_di, adx


def _ao(high: pd.Series, low: pd.Series) -> pd.Series:
    median_price = (high + low) / 2
    return median_price.rolling(5, min_periods=5).mean() - median_price.rolling(34, min_periods=34).mean()


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series]:
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _stoch_rsi(
    close: pd.Series,
    rsi_length: int,
    stoch_length: int,
    smooth_k: int,
    smooth_d: int,
) -> tuple[pd.Series, pd.Series]:
    rsi = _rsi(close, rsi_length)
    lowest_rsi = _rolling_low(rsi, stoch_length)
    highest_rsi = _rolling_high(rsi, stoch_length)
    range_ = highest_rsi - lowest_rsi
    raw = 100 * (rsi - lowest_rsi) / range_
    raw = raw.where(range_ != 0, 0.0)
    k = raw.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(smooth_d, min_periods=smooth_d).mean()
    return k, d


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    highest_high = _rolling_high(high, length)
    lowest_low = _rolling_low(low, length)
    range_ = highest_high - lowest_low
    percent_r = -100 * (highest_high - close) / range_
    return percent_r.where(range_ != 0)


def _ultimate_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fast: int,
    middle: int,
    slow: int,
) -> pd.Series:
    prev_close = close.shift(1)
    buying_pressure = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    true_range = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat([low, prev_close], axis=1).min(axis=1)

    avg_fast = buying_pressure.rolling(fast, min_periods=fast).sum() / true_range.rolling(fast, min_periods=fast).sum()
    avg_middle = buying_pressure.rolling(middle, min_periods=middle).sum() / true_range.rolling(
        middle, min_periods=middle
    ).sum()
    avg_slow = buying_pressure.rolling(slow, min_periods=slow).sum() / true_range.rolling(slow, min_periods=slow).sum()
    return 100 * ((4 * avg_fast) + (2 * avg_middle) + avg_slow) / 7
