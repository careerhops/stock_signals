from __future__ import annotations

from typing import Any

import pandas as pd


def explain_strategy_row(row: pd.Series) -> str:
    signal = str(row.get("signal", "NONE"))
    close = _fmt(row.get("close"))
    upper = _fmt(row.get("upper_level"))
    lower = _fmt(row.get("lower_level"))
    bull_fvg = _fmt(row.get("fvg_bull_recent"), decimals=0)
    bear_fvg = _fmt(row.get("fvg_bear_recent"), decimals=0)

    if bool(row.get("final_buy", False)):
        return f"BUY: close {close} crossed above structural ceiling {upper}; bullish FVG count in lookback is {bull_fvg}."
    if bool(row.get("final_sell", False)):
        return f"SELL: close {close} crossed below structural floor {lower}; bearish FVG count in lookback is {bear_fvg}."
    if bool(row.get("buy_signal", False)):
        return "Raw BUY condition was true, but repeated same-direction signal was suppressed."
    if bool(row.get("sell_signal", False)):
        return "Raw SELL condition was true, but repeated same-direction signal was suppressed."
    if bool(row.get("bull_break", False)):
        return f"Bullish structure break happened, but bullish FVG count is {bull_fvg}."
    if bool(row.get("bear_break", False)):
        return f"Bearish structure break happened, but bearish FVG count is {bear_fvg}."
    if bool(row.get("fvg_bull", False)):
        return "Bullish FVG exists on this candle, but there is no bullish structure break."
    if bool(row.get("fvg_bear", False)):
        return "Bearish FVG exists on this candle, but there is no bearish structure break."
    return "No signal: structure break and FVG confluence are not both present."


def strategy_rows_for_display(strategy_output: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    if strategy_output.empty:
        return strategy_output

    frame = strategy_output.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values("date").tail(limit).copy()
    frame["explanation"] = frame.apply(explain_strategy_row, axis=1)
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame


def build_signal_quality_report(
    raw_signals: pd.DataFrame,
    filtered_signals: pd.DataFrame,
    scan_details: pd.DataFrame,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    filtered_non_buy = _non_buy_filtered_rows(filtered_signals)
    checks.append(
        _check_row(
            "Filtered list contains only BUY rows",
            len(filtered_non_buy) == 0,
            len(filtered_non_buy),
            f"{len(filtered_signals)} filtered rows checked",
        )
    )
    issues.extend(filtered_non_buy)

    latest_mismatches = _latest_raw_signal_mismatches(raw_signals, filtered_signals)
    checks.append(
        _check_row(
            "Filtered BUY agrees with latest raw signal per stock",
            len(latest_mismatches) == 0,
            len(latest_mismatches),
            "Prevents stale BUY rows when latest raw signal is SELL or older",
        )
    )
    issues.extend(latest_mismatches)

    audit_mismatches = _scan_audit_mismatches(raw_signals, scan_details)
    checks.append(
        _check_row(
            "Scan audit latest signal agrees with raw strategy output",
            len(audit_mismatches) == 0,
            len(audit_mismatches),
            f"{len(scan_details)} scan-audit rows checked",
        )
    )
    issues.extend(audit_mismatches)

    missing_candles = _scan_rows_without_candles(scan_details)
    checks.append(
        _check_row(
            "Scanned stocks have local OHLC rows",
            len(missing_candles) == 0,
            len(missing_candles),
            "Stocks with no OHLC rows cannot be validated",
        )
    )
    issues.extend(missing_candles)

    return {
        "checks": checks,
        "issues": issues,
        "summary": {
            "raw_signal_count": len(raw_signals),
            "filtered_signal_count": len(filtered_signals),
            "scan_detail_count": len(scan_details),
            "issue_count": len(issues),
        },
    }


def _non_buy_filtered_rows(filtered_signals: pd.DataFrame) -> list[dict[str, Any]]:
    if filtered_signals.empty or "signal" not in filtered_signals.columns:
        return []
    frame = filtered_signals[filtered_signals["signal"].astype(str).str.upper() != "BUY"].copy()
    return [_issue(row, "Filtered row is not BUY") for _, row in frame.iterrows()]


def _latest_raw_signal_mismatches(
    raw_signals: pd.DataFrame,
    filtered_signals: pd.DataFrame,
) -> list[dict[str, Any]]:
    if raw_signals.empty or filtered_signals.empty:
        return []
    if not {"exchange", "symbol", "date", "signal"}.issubset(raw_signals.columns):
        return []
    if not {"exchange", "symbol", "date", "signal"}.issubset(filtered_signals.columns):
        return []

    raw = raw_signals.copy()
    raw["date_sort"] = pd.to_datetime(raw["date"], errors="coerce")
    latest_raw = raw.sort_values("date_sort").groupby(["exchange", "symbol"], dropna=False).tail(1)
    latest_raw = latest_raw[["exchange", "symbol", "date", "signal"]].rename(
        columns={"date": "latest_raw_date", "signal": "latest_raw_signal"}
    )

    filtered = filtered_signals[["exchange", "symbol", "date", "signal"]].copy()
    merged = filtered.merge(latest_raw, on=["exchange", "symbol"], how="left")
    mismatch = merged[
        (merged["latest_raw_signal"].astype(str).str.upper() != "BUY")
        | (pd.to_datetime(merged["date"], errors="coerce") != pd.to_datetime(merged["latest_raw_date"], errors="coerce"))
    ]

    issues = []
    for _, row in mismatch.iterrows():
        issues.append(
            {
                "exchange": row.get("exchange", ""),
                "symbol": row.get("symbol", ""),
                "date": row.get("date", ""),
                "signal": row.get("signal", ""),
                "problem": (
                    f"Filtered BUY does not match latest raw signal "
                    f"{row.get('latest_raw_signal', '')} on {row.get('latest_raw_date', '')}"
                ),
            }
        )
    return issues


def _scan_audit_mismatches(raw_signals: pd.DataFrame, scan_details: pd.DataFrame) -> list[dict[str, Any]]:
    if raw_signals.empty or scan_details.empty:
        return []
    if not {"exchange", "symbol", "date", "signal"}.issubset(raw_signals.columns):
        return []
    if not {"exchange", "symbol", "latest_signal", "latest_signal_date"}.issubset(scan_details.columns):
        return []

    raw = raw_signals.copy()
    raw["date_sort"] = pd.to_datetime(raw["date"], errors="coerce")
    latest_raw = raw.sort_values("date_sort").groupby(["exchange", "symbol"], dropna=False).tail(1)
    latest_raw = latest_raw[["exchange", "symbol", "date", "signal"]].rename(
        columns={"date": "raw_latest_date", "signal": "raw_latest_signal"}
    )

    audit = scan_details.merge(latest_raw, on=["exchange", "symbol"], how="inner")
    audit_date = pd.to_datetime(audit["latest_signal_date"], errors="coerce")
    raw_date = pd.to_datetime(audit["raw_latest_date"], errors="coerce")
    mismatch = audit[
        (audit["latest_signal"].astype(str).str.upper() != audit["raw_latest_signal"].astype(str).str.upper())
        | (audit_date != raw_date)
    ]
    return [_issue(row, "Scan audit latest signal differs from raw strategy output") for _, row in mismatch.iterrows()]


def _scan_rows_without_candles(scan_details: pd.DataFrame) -> list[dict[str, Any]]:
    if scan_details.empty or "daily_rows" not in scan_details.columns:
        return []
    rows = scan_details[pd.to_numeric(scan_details["daily_rows"], errors="coerce").fillna(0) <= 0]
    return [_issue(row, "No local daily OHLC rows") for _, row in rows.iterrows()]


def _check_row(name: str, passed: bool, failures: int, details: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "failures": failures,
        "details": details,
    }


def _issue(row: pd.Series, problem: str) -> dict[str, Any]:
    return {
        "exchange": row.get("exchange", ""),
        "symbol": row.get("symbol", ""),
        "date": row.get("date", row.get("latest_signal_date", "")),
        "signal": row.get("signal", row.get("latest_signal", "")),
        "problem": problem,
    }


def _fmt(value: Any, decimals: int = 2) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.{decimals}f}"
