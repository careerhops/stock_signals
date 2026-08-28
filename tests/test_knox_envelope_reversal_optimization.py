from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from stock_screener.knox_envelope_reversal_optimization import (
    ConfirmationParameters,
    _confirmation_events,
    _confirmed_signal,
)


class KnoxEnvelopeReversalOptimizationTests(unittest.TestCase):
    def test_confirmation_can_only_fire_on_or_after_recent_setup(self) -> None:
        setup = np.array([False, True, False, False, False])
        event = np.array([True, False, False, True, True])

        actual = _confirmed_signal(setup, event, window=2)

        self.assertEqual(actual.tolist(), [False, False, False, True, False])

    def test_close_above_prior_high_uses_only_prior_bars(self) -> None:
        frame = _daily_frame(
            close=[9.5, 10.5, 12.1, 11.0],
            high=[10.0, 11.0, 12.0, 12.0],
        )
        confirmation = ConfirmationParameters("close_above_prior_high", 0, 2, 0.0)

        event = _confirmation_events(frame, (confirmation,))[confirmation.name]

        self.assertEqual(event.tolist(), [False, False, True, False])

    def test_rsi_confirmation_is_a_cross_not_a_persistent_state(self) -> None:
        close = [100.0] * 8 + [95.0, 90.0, 85.0, 80.0, 82.0, 84.0, 86.0, 88.0]
        frame = _daily_frame(close=close)
        confirmation = ConfirmationParameters("rsi_cross", 2, 3, 30.0)

        event = _confirmation_events(frame, (confirmation,))[confirmation.name]

        indexes = np.flatnonzero(event)
        self.assertLessEqual(len(indexes), 1)


def _daily_frame(
    *,
    close: list[float],
    high: list[float] | None = None,
) -> pd.DataFrame:
    close_series = pd.Series(close, dtype=float)
    high_series = pd.Series(high, dtype=float) if high is not None else close_series + 1.0
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(close), freq="B"),
            "open": close_series - 0.5,
            "high": high_series,
            "low": close_series - 1.0,
            "close": close_series,
            "volume": 1000.0,
        }
    )


if __name__ == "__main__":
    unittest.main()
