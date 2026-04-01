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
#  STEP 6 — SPY REGIME CHECK
# ─────────────────────────────────────────────
def get_spy_regime(as_of_date: date) -> bool | None:
    """
    Returns True  if SPY MA10 > MA20 on as_of_date (regime bullish — longs allowed).
    Returns False if SPY MA10 <= MA20 (regime bearish — longs suppressed).
    Returns None  if SPY data is unavailable (filter disabled for this run).
    """
    try:
        spy_data = download_batch(["SPY"], as_of_date=as_of_date)
        if "SPY" not in spy_data:
            return None
        df       = spy_data["SPY"]
        as_of_ts = pd.Timestamp(as_of_date)
        df       = df[df.index <= as_of_ts]
        if len(df) < MA_SLOW + 2:
            return None
        df["ma_fast"] = df["Close"].rolling(MA_FAST).mean()
        df["ma_slow"] = df["Close"].rolling(MA_SLOW).mean()
        last = df.iloc[-1]
        if pd.isna(last["ma_fast"]) or pd.isna(last["ma_slow"]):
            return None
        return bool(last["ma_fast"] > last["ma_slow"])
    except Exception:
        return None


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

st.sidebar.markdown("---")

# --- SPY Regime Filter ---
st.sidebar.subheader("📡 SPY Regime Filter")
use_spy_regime = st.sidebar.checkbox(
    "SPY Regime Filter (MA10 > MA20)",
    value=True,
    help="Only show LONG signals when SPY 10-day MA is above 20-day MA on the signal date."
)

st.sidebar.markdown("---")

# --- Capital Allocation ---
st.sidebar.subheader("💰 Capital Allocation")
capital = st.sidebar.number_input(
    "Total capital ($)",
    min_value=1_000,
    max_value=100_000_000,
    value=100_000,
    step=1_000,
    help="Total portfolio capital used to size positions."
)
max_risk_pct = st.sidebar.number_input(
    "Max risk per trade (% of capital)",
    min_value=0.1, max_value=5.0, value=1.0, step=0.1,
    help="Maximum $ loss if stop is hit, as a % of capital. Default 1%."
)
max_position_pct = st.sidebar.number_input(
    "Max position size (% of capital)",
    min_value=1.0, max_value=100.0, value=20.0, step=1.0,
    help="Maximum $ allocated to a single position, as a % of capital. Default 20%."
)

max_risk_dollars     = capital * (max_risk_pct     / 100)
max_position_dollars = capital * (max_position_pct / 100)

run_button = st.sidebar.button("🚀 Run Screener")

# --- Main scan loop ---
if run_button:
    long_hits, short_hits, errors = [], [], []
    batches       = [tickers[i : i + int(batch_size)] for i in range(0, len(tickers), int(batch_size))]
    total         = len(tickers)
    total_batches = len(batches)
    tickers_done  = 0

    # ── SPY regime check ──────────────────────────────────────────────────────
    spy_regime_ok = None
    if use_spy_regime:
        with st.spinner("Checking SPY regime (MA10 vs MA20)…"):
            spy_regime_ok = get_spy_regime(as_of_date)
        if spy_regime_ok is None:
            st.warning("⚠️ Could not fetch SPY data — regime filter disabled for this run.")
        elif spy_regime_ok:
            st.success("✅ SPY regime: BULLISH (MA10 > MA20) — long signals enabled.")
        else:
            st.warning("🚫 SPY regime: BEARISH (MA10 ≤ MA20) — long signals suppressed.")

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

            last  = df.iloc[-1]
            prev  = df.iloc[-2]
            chg   = (last["Close"] - prev["Close"]) / prev["Close"] * 100
            price = float(last["Close"])
            sl    = float(last["Low"])    # stop loss = low of signal bar

            # ── Position sizing ───────────────────────────────────────────────
            risk_per_share = price - sl
            if risk_per_share > 0:
                shares_by_risk     = max_risk_dollars / risk_per_share
                shares_by_position = max_position_dollars / price
                shares             = int(min(shares_by_risk, shares_by_position))
            else:
                shares = 0

            position_value = shares * price
            dollar_risk    = shares * risk_per_share if shares > 0 else 0.0

            row = {
                "Ticker":           ticker,
                "Signal":           sig,
                "Date":             df.index[-1].strftime("%Y-%m-%d"),
                "Price ($)":        round(price, 2),
                "Stop Loss ($)":    round(sl, 2),
                "Shares":           shares,
                "Position ($)":     round(position_value, 2),
                "Risk ($)":         round(dollar_risk, 2),
                "MA10 ($)":         round(float(last["ma_fast"]), 2),
                "MA20 ($)":         round(float(last["ma_slow"]), 2),
                "Change %":         round(chg, 2),
                "Cross (days ago)": days_ago,
            }

            if sig == "LONG":
                # Gate on SPY regime — skip if filter is on and regime is bearish
                if use_spy_regime and spy_regime_ok is False:
                    pass   # regime bearish — suppress long signal
                else:
                    long_hits.append(row)
            else:
                short_hits.append(row)

        results_text.text(f"✅ Found: {len(long_hits)} long  |  {len(short_hits)} short")

    status_text.text("✅ Scan complete!")

    if errors:
        with st.expander(f"⚠️ {len(errors)} ticker errors"):
            st.write("\n".join(errors[:30]))

    st.header(f"📊 Results — as of {as_of_date.strftime('%d %b %Y')}")

    # ── SPY regime status banner ──────────────────────────────────────────────
    if use_spy_regime:
        if spy_regime_ok is True:
            st.info("📡 SPY Regime: **BULLISH** (MA10 > MA20) — long signals shown.")
        elif spy_regime_ok is False:
            st.warning("📡 SPY Regime: **BEARISH** (MA10 ≤ MA20) — long signals suppressed.")
        else:
            st.warning("📡 SPY Regime: **UNAVAILABLE** — regime filter was bypassed.")
    else:
        st.info("📡 SPY Regime Filter: **OFF** — all long signals shown regardless of market.")

    # ── Summary metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers Scanned", total)
    c2.metric("Long Signals",    len(long_hits))
    c3.metric("Short Signals",   len(short_hits))
    success_rate = (total - len(errors)) / total * 100 if total else 0
    c4.metric("Success Rate",    f"{success_rate:.1f}%")

    st.markdown("---")

    # ── Capital allocation summary ────────────────────────────────────────────
    st.caption(
        f"Capital: **${capital:,.0f}**  |  "
        f"Max risk/trade: **${max_risk_dollars:,.0f}** ({max_risk_pct}%)  |  "
        f"Max position: **${max_position_dollars:,.0f}** ({max_position_pct}%)"
    )

    # ── Long signals table ────────────────────────────────────────────────────
    st.subheader("🟢 Long Signals")
    if long_hits:
        df_long = pd.DataFrame(long_hits)
        display_cols = [
            "Ticker", "Date", "Price ($)", "Stop Loss ($)",
            "Shares", "Position ($)", "Risk ($)",
            "MA10 ($)", "MA20 ($)", "Change %", "Cross (days ago)",
        ]
        df_display = df_long[display_cols].copy()
        fmt_dollar = ["Price ($)", "Stop Loss ($)", "MA10 ($)", "MA20 ($)"]
        fmt_money  = ["Position ($)", "Risk ($)"]
        st.dataframe(
            df_display.style
                .format({c: "${:,.2f}" for c in fmt_dollar})
                .format({c: "${:,.0f}" for c in fmt_money})
                .format({"Change %": "{:+.2f}%", "Shares": "{:,}"}),
            use_container_width=True,
            hide_index=True,
        )

        # Aggregate risk exposure
        total_position = df_long["Position ($)"].sum()
        total_risk     = df_long["Risk ($)"].sum()
        pct_deployed   = total_position / capital * 100 if capital else 0
        pct_at_risk    = total_risk     / capital * 100 if capital else 0

        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Signals found",    len(long_hits))
        ec2.metric("Total deployed",   f"${total_position:,.0f}  ({pct_deployed:.1f}%)")
        ec3.metric("Total $ at risk",  f"${total_risk:,.0f}  ({pct_at_risk:.1f}%)")
        ec4.metric("Avg risk/trade",   f"${total_risk / len(long_hits):,.0f}" if long_hits else "—")

        st.download_button(
            "📥 Download Long Signals CSV",
            df_long.to_csv(index=False),
            f"long_{datetime.now():%Y%m%d_%H%M%S}.csv",
            "text/csv",
        )
    else:
        if use_spy_regime and spy_regime_ok is False:
            st.info("No long signals shown — SPY regime is bearish.")
        else:
            st.info("No long signals found.")

else:
    st.info("👈 Click 'Run Screener' in the sidebar to start.")
    if tickers:
        st.write(f"Loaded {len(tickers)} tickers.")
        with st.expander("First 20 tickers"):
            st.write(tickers[:20])

st.markdown("---")
st.caption(f"Last updated: {datetime.now():%Y-%m-%d %H:%M:%S}")
