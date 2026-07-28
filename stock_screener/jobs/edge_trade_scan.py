"""Daily edge-trade scan orchestrator.

Designed to be called from the existing daily scan once approved.

Public entry point: ``run_edge_trade_scan(config, storage, instruments)``.

This module **does not** call the Kite API. It reuses whatever daily candles
the main ``daily_scan`` pipeline has already persisted to ``Storage``. That
keeps the API budget identical to today and adds zero extra HTTP latency.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import pandas as pd

from stock_screener.data.storage import Storage
from stock_screener.strategy.edge_trades import (
    EdgeTradeConfig,
    EdgeSignal,
    SETUP_BREAKOUT,
    SETUP_MEAN_REVERSION,
    evaluate_symbol,
    signals_to_dataframe,
)


@dataclass
class EdgeScanResult:
    signals: pd.DataFrame                # one row per fresh candidate
    trades: pd.DataFrame                 # historical backtest trades (audit)
    summary: dict[str, Any]


def edge_trade_config_from_settings(config: dict[str, Any]) -> EdgeTradeConfig:
    section = (config.get("edge_trades") or {})
    base = EdgeTradeConfig()
    if not section:
        return base
    # Only override known fields; ignore unknown keys for forward compatibility.
    overrides = {k: v for k, v in section.items() if k in base.__dataclass_fields__}
    return EdgeTradeConfig(**{**base.__dict__, **overrides})


def run_edge_trade_scan(
    config: dict[str, Any],
    storage: Storage,
    instruments: pd.DataFrame,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> EdgeScanResult:
    cfg = edge_trade_config_from_settings(config)

    universe = _select_universe(instruments, config)
    if universe.empty:
        return EdgeScanResult(
            signals=pd.DataFrame(),
            trades=pd.DataFrame(),
            summary={"symbols_scanned": 0, "signals": 0},
        )

    signal_rows: list[EdgeSignal] = []
    trade_frames: list[pd.DataFrame] = []
    skipped_no_history = 0
    skipped_no_liquidity = 0

    for idx, instrument in enumerate(universe.to_dict(orient="records"), start=1):
        exchange = str(instrument["exchange"])
        symbol = str(instrument["tradingsymbol"])

        if progress_callback:
            progress_callback({
                "phase": "Edge-trade scoring",
                "completed": idx - 1,
                "total": len(universe),
                "current_symbol": symbol,
                "current_exchange": exchange,
            })

        candles = storage.load_candles(exchange, symbol, "1D")
        if candles.empty or len(candles) < cfg.min_history_days:
            skipped_no_history += 1
            continue

        signals, trades = evaluate_symbol(candles, symbol=symbol, exchange=exchange, cfg=cfg)
        if not signals and trades.empty:
            # Liquidity gate failed or no historical trades fired.
            if (candles["close"] * candles["volume"]).rolling(20).mean().iloc[-1] < cfg.min_avg_traded_value_inr:
                skipped_no_liquidity += 1
            continue

        signal_rows.extend(signals)
        if not trades.empty:
            trade_frames.append(trades)

    signals_df = signals_to_dataframe(signal_rows)
    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

    summary = {
        "symbols_scanned": len(universe),
        "skipped_no_history": skipped_no_history,
        "skipped_no_liquidity": skipped_no_liquidity,
        "signals": int(len(signals_df)),
        "breakout_signals": int((signals_df["setup"] == SETUP_BREAKOUT).sum()) if not signals_df.empty else 0,
        "mean_reversion_signals": int((signals_df["setup"] == SETUP_MEAN_REVERSION).sum()) if not signals_df.empty else 0,
        "historical_trades": int(len(trades_df)),
    }

    storage.save_signals("latest_edge_trade_signals.csv", signals_df)
    storage.save_signals("latest_edge_trade_trades.csv", trades_df)

    if progress_callback:
        progress_callback({
            "phase": "Edge-trade complete",
            "completed": len(universe),
            "total": len(universe),
            "current_symbol": "",
            "summary": summary,
        })

    return EdgeScanResult(signals=signals_df, trades=trades_df, summary=summary)


def _select_universe(instruments: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Use whatever universe the project already builds.

    Falls back to ``stock_screener.universe.build_universe`` so the edge scan
    operates on the exact same NSE list as the existing weekly scan.
    """
    from stock_screener.universe import build_universe

    universe = build_universe(instruments, config)
    if universe.empty:
        return universe
    # Edge trades target liquid NSE equities only; drop BSE-only listings here
    # because the liquidity gate is computed inside ``evaluate_symbol`` anyway.
    return universe[universe["exchange"].astype(str).str.upper() == "NSE"].copy()


# Helper for ad-hoc CLI use (`python -m stock_screener.jobs.edge_trade_scan`)
if __name__ == "__main__":  # pragma: no cover
    from stock_screener.config import get_data_root, load_config

    loaded_config = load_config()
    data_root = get_data_root(loaded_config)
    store = Storage(data_root)
    instruments_df = store.load_instruments()
    result = run_edge_trade_scan(loaded_config, store, instruments_df)
    print(result.summary)
