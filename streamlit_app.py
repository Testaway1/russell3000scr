import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from streamlit_gsheets import GSheetsConnection
from datetime import datetime, date, timedelta

# ─────────────────────────────────────────────
#  CONFIG  (mirrors standalone script exactly)
# ─────────────────────────────────────────────
ATR_PERIOD        = 10
FACTOR            = 3.0
MA_FAST           = 10
MA_SLOW           = 20
MA_CROSS_LOOKBACK = 10
HISTORY_PERIOD    = "1y"
INTERVAL          = "1d"
BATCH_SIZE        = 50


# ─────────────────────────────────────────────
#  STEP 1 — DATA DOWNLOAD  (batched, matches standalone)
# ─────────────────────────────────────────────
def download_batch(tickers: list[str],
                   as_of_date: date = None) -> dict[str, pd.DataFrame]:
    """
    Batch download via yfinance — identical logic to standalone script.
    Handles both old (flat) and new (MultiIndex) yfinance column layouts.
    Returns { ticker: OHLCV DataFrame }.

    as_of_date: if supplied, data is fetched up to and including that date,
                with 1 year of history ending on that day.
                Defaults to today (original behaviour).
    """
    if not tickers:
        return {}

    # Determine date range
    end_date   = as_of_date if as_of_date else date.today()
    start_date = end_date - timedelta(days=365)

    # yfinance end is exclusive, so add 1 day to include the as_of date itself
    end_fetch  = end_date + timedelta(days=1)

    raw = yf.download(
        tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_fetch.strftime("%Y-%m-%d"),
        interval=INTERVAL,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    result = {}
    if raw.empty:
        return result

    # ── Single ticker — flat columns ──────────────────────────────────────────
    if len(tickers) == 1:
        tkr = tickers[0]
        try:
            df = raw.dropna()
            if len(df) >= MA_SLOW + ATR_PERIOD + 5:
                result[tkr] = df
        except Exception:
            pass
        return result

    # ── Multi-ticker — MultiIndex columns ─────────────────────────────────────
    cols = raw.columns
    if not isinstance(cols, pd.MultiIndex):
        try:
            tkr = tickers[0]
            df  = raw.dropna()
            if len(df) >= MA_SLOW + ATR_PERIOD + 5:
                result[tkr] = df
        except Exception:
            pass
        return result

    level0  = set(cols.get_level_values(0).unique())
    fields  = {"Close", "High", "Low", "Open", "Volume"}
    field_level, ticker_level = (0, 1) if fields & level0 else (1, 0)

    for tkr in tickers:
        try:
            tkr_cols = cols[cols.get_level_values(ticker_level) == tkr]
            if tkr_cols.empty:
                continue
            sub = raw[tkr_cols].copy()
            sub.columns = sub.columns.get_level_values(field_level)
            sub = sub.dropna()
            if len(sub) >= MA_SLOW + ATR_PERIOD + 5:
                result[tkr] = sub
        except Exception:
            continue

    return result


# ─────────────────────────────────────────────
#  STEP 2 — SUPERTREND  (matches Pine Script ta.supertrend())
# ─────────────────────────────────────────────
def calc_supertrend(df: pd.DataFrame,
                    atr_period: int = ATR_PERIOD,
                    factor: float   = FACTOR) -> pd.Series | None:
    """
    Matches Pine Script ta.supertrend() — identical to standalone script.

    Convention (same as Pine Script):
        direction = -1  →  Uptrend   (price above Supertrend line)
        direction = +1  →  Downtrend (price below Supertrend line)

    Changes from previous Streamlit version:
      1. ATR via pandas ewm(alpha=1/period, adjust=False) — matches Pine Script
      2. st_line tracker used for flip logic — matches Pine Script exactly
      3. Direction convention corrected to -1/+1 (was 1/-1)
    """
    try:
        high  = df["High"]
        low   = df["Low"]
        close = df["Close"]

        # ATR — pandas EMA, matches Pine Script / standalone
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()

        hl2         = (high + low) / 2.0
        basic_upper = hl2 + factor * atr
        basic_lower = hl2 - factor * atr

        n           = len(df)
        final_upper = np.zeros(n)
        final_lower = np.zeros(n)
        direction   = np.ones(n)      # +1 = Downtrend (Pine Script convention)
        st_line     = np.zeros(n)     # tracks actual Supertrend line level

        for i in range(1, n):
            # Upper band
            if basic_upper.iloc[i] < final_upper[i-1] or close.iloc[i-1] > final_upper[i-1]:
                final_upper[i] = basic_upper.iloc[i]
            else:
                final_upper[i] = final_upper[i-1]

            # Lower band
            if basic_lower.iloc[i] > final_lower[i-1] or close.iloc[i-1] < final_lower[i-1]:
                final_lower[i] = basic_lower.iloc[i]
            else:
                final_lower[i] = final_lower[i-1]

            # Direction via st_line — matches Pine Script logic exactly
            if st_line[i-1] == final_upper[i-1]:
                direction[i] = 1 if close.iloc[i] <= final_upper[i] else -1
            else:
                direction[i] = -1 if close.iloc[i] >= final_lower[i] else 1

            st_line[i] = final_lower[i] if direction[i] == -1 else final_upper[i]

        return pd.Series(direction, index=df.index, name="direction")

    except Exception as e:
        st.error(f"Supertrend calculation error: {e}")
        return None


# ─────────────────────────────────────────────
#  STEP 3 — INDICATORS
# ─────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma_fast"]   = df["Close"].rolling(MA_FAST).mean()
    df["ma_slow"]   = df["Close"].rolling(MA_SLOW).mean()
    result = calc_supertrend(df, ATR_PERIOD, FACTOR)
    if result is not None:
        df["direction"] = result.values
    else:
        df["direction"] = np.nan
    return df


# ─────────────────────────────────────────────
#  STEP 4 — MA CROSSOVER CHECK  (matches standalone)
# ─────────────────────────────────────────────
def ma_cross_within(df: pd.DataFrame, bars: int = MA_CROSS_LOOKBACK) -> tuple[bool, bool, int]:
    """
    Identical logic to standalone script's ma_cross_within().
    Scans backwards up to `bars` candles for the most recent MA crossover.
    Returns (bullish_cross, bearish_cross, days_ago).
    days_ago = bars back from last bar where cross occurred (-1 if none found).
    """
    fast     = df["ma_fast"].values
    slow     = df["ma_slow"].values
    last_bar = len(df) - 1

    for i in range(last_bar, 0, -1):
        if (last_bar - i) > bars:
            break
        if pd.isna(fast[i]) or pd.isna(slow[i]) or pd.isna(fast[i-1]) or pd.isna(slow[i-1]):
            continue
        if fast[i-1] <= slow[i-1] and fast[i] > slow[i]:   # bullish cross
            return True, False, last_bar - i
        if fast[i-1] >= slow[i-1] and fast[i] < slow[i]:   # bearish cross
            return False, True, last_bar - i

    return False, False, -1


# ─────────────────────────────────────────────
#  STEP 5 — SIGNAL DETECTION  (matches standalone exactly)
# ─────────────────────────────────────────────
def check_signals(df: pd.DataFrame,
                  cross_lookback: int = MA_CROSS_LOOKBACK) -> tuple[str | None, int]:
    """
    Returns (signal, days_ago) where signal is 'LONG', 'SHORT', or None.
    Direction convention: -1 = Uptrend, +1 = Downtrend (Pine Script).
    """
    if len(df) < MA_SLOW + cross_lookback + 2:
        return None, -1

    row = df.iloc[-1]
    if pd.isna(row["ma_fast"]) or pd.isna(row["ma_slow"]) or pd.isna(row["direction"]):
        return None, -1

    direction = row["direction"]
    ma_fast   = row["ma_fast"]
    ma_slow   = row["ma_slow"]
    open_     = row["Open"]
    close     = row["Close"]
    high      = row["High"]
    low       = row["Low"]

    bullish_cross, bearish_cross, days_ago = ma_cross_within(df, cross_lookback)

    long_signal = (
        direction == -1        # Uptrend  (Pine Script convention)
        and ma_fast > ma_slow
        and bullish_cross
        and low   <= ma_slow
        and close >  ma_slow
        and close >  open_
    )

    short_signal = (
        direction == 1         # Downtrend (Pine Script convention)
        and ma_fast < ma_slow
        and bearish_cross
        and high  >= ma_slow
        and close <  ma_slow
        and close <  open_
    )

    if long_signal:  return "LONG",  days_ago
    if short_signal: return "SHORT", days_ago
    return None, -1


# ─────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Russell 3000 Screener", layout="wide")
st.title("📈 Russell 3000 Cloud Screener")
st.subheader("Supertrend + MA Confluence (Daily)")

# --- Google Sheets connection ---
try:
    conn       = st.connection("gsheets", type=GSheetsConnection)
    df_tickers = conn.read(ttl=600)

    if df_tickers is None or df_tickers.empty:
        st.error("No data found in Google Sheets.")
        st.stop()

    if "Ticker" not in df_tickers.columns:
        st.error(f"Sheet is missing a 'Ticker' column. Found: {list(df_tickers.columns)}")
        st.stop()

    tickers = df_tickers["Ticker"].dropna().unique().tolist()
    st.sidebar.success(f"✅ {len(tickers)} tickers loaded from Google Sheets.")

except Exception as e:
    st.error(f"❌ Google Sheets connection failed: {e}")
    st.info(
        "Add a `.streamlit/secrets.toml` with:\n\n"
        "```toml\n[connections.gsheets]\nspreadsheet = 'YOUR_SPREADSHEET_ID'\n```"
    )
    st.stop()

# --- Sidebar controls ---
st.sidebar.header("Screener Controls")

use_samples = st.sidebar.checkbox("Use sample tickers for testing")
if use_samples:
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
    st.sidebar.success(f"Using {len(tickers)} sample tickers")

max_tickers = st.sidebar.number_input(
    "Max tickers to scan (0 = all)", min_value=0, max_value=5000, value=0
)
if max_tickers > 0:
    tickers = tickers[:max_tickers]
    st.sidebar.info(f"Limited to {max_tickers} tickers")

cross_window = st.sidebar.number_input(
    "MA crossover window (days)", min_value=1, max_value=30, value=MA_CROSS_LOOKBACK,
    help="Only flag signals where MA10/MA20 crossed within this many bars"
)

batch_size = st.sidebar.number_input(
    "Batch size", min_value=1, max_value=200, value=BATCH_SIZE,
    help="Tickers downloaded per yfinance batch call (50 recommended)"
)

st.sidebar.markdown("---")
as_of_date = st.sidebar.date_input(
    "Run screener as of",
    value=date.today(),
    max_value=date.today(),
    help="Evaluate signals as at close of this date. Cannot be a future date."
)
# Warn if a weekend or future date is selected
if as_of_date > date.today():
    st.sidebar.warning("Date cannot be in the future — defaulting to today.")
    as_of_date = date.today()
if as_of_date.weekday() >= 5:
    st.sidebar.warning(
        f"{as_of_date.strftime('%A %d %b %Y')} is a weekend. "
        "yfinance will use the last available trading day on or before this date."
    )

run_button = st.sidebar.button("🚀 Run Screener")

# --- Main scan loop ---
if run_button:
    long_hits, short_hits, errors = [], [], []

    batches       = [tickers[i : i + int(batch_size)] for i in range(0, len(tickers), int(batch_size))]
    total         = len(tickers)
    total_batches = len(batches)
    tickers_done  = 0

    progress_bar = st.progress(0)
    status_text  = st.empty()
    results_text = st.empty()

    for b_idx, batch in enumerate(batches):
        status_text.text(
            f"📦 Downloading batch {b_idx + 1}/{total_batches} ({len(batch)} tickers)…"
        )

        try:
            price_data = download_batch(batch, as_of_date=as_of_date)
        except Exception as e:
            for tkr in batch:
                errors.append(f"{tkr}: batch download failed — {str(e)[:80]}")
            tickers_done += len(batch)
            progress_bar.progress(tickers_done / total)
            continue

        for ticker in batch:
            tickers_done += 1
            progress_bar.progress(tickers_done / total)
            status_text.text(
                f"📊 Scanning {ticker} ({tickers_done}/{total}, batch {b_idx + 1}/{total_batches})"
            )

            if ticker not in price_data:
                continue

            try:
                raw_df = price_data[ticker]

                # Slice to as_of_date so signals are evaluated at that day's close
                as_of_ts = pd.Timestamp(as_of_date)
                raw_df   = raw_df[raw_df.index <= as_of_ts]
                if len(raw_df) < MA_SLOW + MA_CROSS_LOOKBACK + 2:
                    continue

                df            = calc_indicators(raw_df)
                sig, days_ago = check_signals(df, cross_lookback=int(cross_window))
            except Exception as e:
                errors.append(f"{ticker}: {str(e)[:120]}")
                continue

            if sig is None:
                continue

            last = df.iloc[-1]
            prev = df.iloc[-2]
            chg  = (last["Close"] - prev["Close"]) / prev["Close"] * 100

            row = {
                "Ticker":           ticker,
                "Signal":           sig,
                "Price":            f"${float(last['Close']):.2f}",
                "MA10":             f"${float(last['ma_fast']):.2f}",
                "MA20":             f"${float(last['ma_slow']):.2f}",
                "Change %":         f"{chg:.2f}%",
                "Cross (days ago)": days_ago,
                "Date":             df.index[-1].strftime("%Y-%m-%d"),
            }

            if sig == "LONG":
                long_hits.append(row)
            else:
                short_hits.append(row)

        results_text.text(f"✅ Found: {len(long_hits)} long  |  {len(short_hits)} short")

    status_text.text("✅ Scan complete!")

    if errors:
        with st.expander(f"⚠️ {len(errors)} ticker errors"):
            st.write("\n".join(errors[:30]))

    st.header(f"📊 Results — as of {as_of_date.strftime('%d %b %Y')}")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟢 Long Signals")
        if long_hits:
            df_long = pd.DataFrame(long_hits)
            st.dataframe(df_long, use_container_width=True)
            st.download_button(
                "📥 Download CSV", df_long.to_csv(index=False),
                f"long_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv"
            )
        else:
            st.info("No long signals found.")

    with col2:
        st.subheader("🔴 Short Signals")
        if short_hits:
            df_short = pd.DataFrame(short_hits)
            st.dataframe(df_short, use_container_width=True)
            st.download_button(
                "📥 Download CSV", df_short.to_csv(index=False),
                f"short_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv"
            )
        else:
            st.info("No short signals found.")

    st.subheader("📈 Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers Scanned", total)
    c2.metric("Long Signals",    len(long_hits))
    c3.metric("Short Signals",   len(short_hits))
    success_rate = (total - len(errors)) / total * 100 if total else 0
    c4.metric("Success Rate",    f"{success_rate:.1f}%")

else:
    st.info("👈 Click 'Run Screener' in the sidebar to start.")
    if tickers:
        st.write(f"Loaded {len(tickers)} tickers.")
        with st.expander("First 20 tickers"):
            st.write(tickers[:20])

st.markdown("---")
st.caption(f"Last updated: {datetime.now():%Y-%m-%d %H:%M:%S}")
