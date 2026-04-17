# NSE/BSE Investment Signal Screener

End-of-day investment screener for NSE/BSE stocks. The app translates the existing TradingView PineScript signal logic into Python, runs it across a configurable stock universe after market close, applies investment filters, and sends a Telegram alert with the final shortlist.

This is not an intraday trading system and does not place orders.

## Recommended Architecture

- Backend and dashboard: FastAPI + Jinja templates
- Data processing: Pandas
- Storage: local files on Railway Volume, with CSV/Parquet outputs
- Broker data: Zerodha Kite Connect
- Scheduler: Railway Cron service
- Alerts: Telegram Bot API first; WhatsApp later if required

## Why FastAPI + Jinja Instead Of Streamlit

Streamlit is excellent for local prototypes, but this project is going to Railway and needs a clean web service plus a separate cron job. FastAPI with server-rendered templates gives us:

- one Python codebase
- simple Railway deployment
- low memory footprint
- easy health checks
- clearer separation between daily scan job and dashboard
- no React/Node build complexity

## Project Services On Railway

Create two Railway services from the same GitHub repo:

1. `screener-web`
   - Persistent web service
   - Start command:
     ```bash
     uvicorn stock_screener.web.main:app --host 0.0.0.0 --port $PORT
     ```

2. `screener-daily-scan`
   - Cron service
   - Start command:
     ```bash
     python -m stock_screener.jobs.daily_scan
     ```
   - Cron schedule for 6:00 PM IST, Monday-Friday:
     ```cron
     30 12 * * 1-5
     ```

Railway cron schedules use UTC, so 6:00 PM IST equals 12:30 PM UTC.

## Local Setup Commands

Run these from VSCode terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cp config/settings.example.yaml config/settings.yaml
```

Then edit:

```text
.env
config/settings.yaml
```

Run the daily scan manually:

```bash
python -m stock_screener.jobs.daily_scan
```

Run the dashboard:

```bash
uvicorn stock_screener.web.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

## Universe Modes

Scan all NSE equity instruments from Kite:

```yaml
universe:
  mode: "nse_all"
  max_symbols: null
  allow_symbols: []
```

Scan a handpicked list:

```yaml
universe:
  mode: "configured"
  allow_symbols: ["AEROFLEX", "PNGJL", "JIOFIN"]
```

Optional industry and market cap filtering is controlled through:

```text
config/symbol_metadata.csv
```

## NSE Large Deals And Supabase

Run this SQL in Supabase first:

```text
supabase/schema.sql
```

Then set backend environment variables:

```env
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
SUPABASE_LARGE_DEALS_TABLE=nse_large_deals
```

Fetch and save NSE Large Deals from terminal:

```bash
python scripts/fetch_nse_large_deals.py
```

Or use the UI:

```text
Big Bull Deals -> Fetch NSE Large Deals
```

## Data Flow

```text
Kite Connect
   |
   v
Instrument Master + Daily Candles
   |
   v
Local Data Folder / Railway Volume
   |
   v
Weekly Resampling
   |
   v
Python Strategy Engine
   |
   v
Filter Engine
   |
   v
Signal CSV/Parquet
   |
   v
FastAPI Dashboard + Telegram Alert
```

## Important Notes

- Keep secrets in `.env` locally and Railway Variables in production.
- Do not commit `.env`.
- Kite access tokens are daily/session tokens. Use the dashboard's `Login with Kite` button, or run `scripts/generate_kite_access_token.py`, to save a fresh token before the evening scan. You do not need to edit `.env` every day.
- The first historical backfill can take time because Kite historical candles are fetched per instrument.
- Daily runs should be incremental after the first backfill.
- If Kite access token is invalid, the cron job will fail fast and send an alert if Telegram is configured.
