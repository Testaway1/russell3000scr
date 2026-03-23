import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from streamlit_gsheets import GSheetsConnection
from datetime import datetime

# --- CONFIG ---
ATR_PERIOD = 10
FACTOR = 3.0
MA_FAST = 10
MA_SLOW = 20


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix #1 — yfinance ≥ 0.2.x returns a MultiIndex like (Field, Ticker).
    Flatten to single-level column names so df["Close"] etc. always work.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Level 0 is the field name (Open/High/Low/Close/Volume)
        df.columns = [col[0] for col in df.columns]
    return df


def calc_supertrend(df: pd.DataFrame) -> np.ndarray | None:
    """
    Fix #2 — use numpy arrays throughout to avoid pandas SettingWithCopyWarning
              and silent no-op assignments that broke band tracking.
    Fix #3 — corrected direction logic:
              close > final_upper  → trend turns bearish  (-1)
              close < final_lower  → trend turns bullish   (1)
              otherwise persist previous direction
    Returns an array where  1 = bullish,  -1 = bearish.
    """
    try:
        high  = df["High"].to_numpy(dtype=float)
        low   = df["Low"].to_numpy(dtype=float)
        close = df["Close"].to_numpy(dtype=float)
        n = len(close)

        # --- True Range ---
        tr = np.empty(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i]  - close[i - 1]),
            )

        # --- ATR (EMA) ---
        alpha = 1.0 / ATR_PERIOD
        atr = np.empty(n)
        atr[0] = tr[0]
        for i in range(1, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

        # --- Basic bands ---
        hl2         = (high + low) / 2.0
        basic_upper = hl2 + FACTOR * atr
        basic_lower = hl2 - FACTOR * atr

        # --- Final bands (numpy arrays — no pandas copy issues) ---
        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        direction   = np.ones(n, dtype=int)   # start bullish

        for i in range(1, n):
            # Upper band: tighten unless previous close was above it
            if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
                final_upper[i] = basic_upper[i]
            else:
                final_upper[i] = final_upper[i - 1]

            # Lower band: tighten unless previous close was below it
            if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
                final_lower[i] = basic_lower[i]
            else:
                final_lower[i] = final_lower[i - 1]

            # Direction — Fix #3: correct flip conditions
            prev_dir = direction[i - 1]
            if prev_dir == 1:          # currently bullish
                if close[i] < final_lower[i]:
                    direction[i] = -1  # flip bearish
                else:
                    direction[i] = 1   # stay bullish
            else:                      # currently bearish
                if close[i] > final_upper[i]:
                    direction[i] = 1   # flip bullish
                else:
                    direction[i] = -1  # stay bearish

        return direction

    except Exception as e:
        st.error(f"Supertrend calculation error: {e}")
        return None


# --- STREAMLIT UI ---
st.set_page_config(page_title="Russell 3000 Screener", layout="wide")
st.title("📈 Russell 3000 Cloud Screener")
st.subheader("Supertrend + MA Confluence (Daily)")

# --- Google Sheets connection ---
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    df_tickers = conn.read(ttl=600)   # Fix: ttl must be int (seconds), not "10m"

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

max_tickers = st.sidebar.number_input("Max tickers to scan (0 = all)", min_value=0, max_value=5000, value=0)
if max_tickers > 0:
    tickers = tickers[:max_tickers]
    st.sidebar.info(f"Limited to {max_tickers} tickers")

run_button = st.sidebar.button("🚀 Run Screener")

# --- Main scan loop ---
if run_button:
    long_hits, short_hits, errors = [], [], []

    progress_bar = st.progress(0)
    status_text  = st.empty()
    results_text = st.empty()
    total        = len(tickers)

    for idx, ticker in enumerate(tickers):
        status_text.text(f"📊 Scanning {ticker} ({idx + 1}/{total})")
        progress_bar.progress((idx + 1) / total)

        try:
            raw = yf.download(ticker, period="1y", interval="1d",
                              progress=False, auto_adjust=True)

            if raw is None or len(raw) < 30:
                continue

            # Fix #1 — flatten MultiIndex columns
            df = flatten_columns(raw.copy())

            required = ["Open", "High", "Low", "Close"]
            if not all(c in df.columns for c in required):
                errors.append(f"{ticker}: missing columns after flatten ({list(df.columns)})")
                continue

            df["MA10"] = df["Close"].rolling(MA_FAST).mean()
            df["MA20"] = df["Close"].rolling(MA_SLOW).mean()

            direction = calc_supertrend(df)
            if direction is None:
                continue
            df["dir"] = direction

            curr = df.iloc[-1]
            prev = df.iloc[-2]

            if pd.isna(curr[["MA10", "MA20", "dir"]]).any():
                continue

            chg = (curr["Close"] - prev["Close"]) / prev["Close"] * 100

            # Long: Supertrend bullish (1), MA10 > MA20, candle touched MA20 then closed above it bullish
            is_long = (
                curr["dir"] == 1
                and curr["MA10"] > curr["MA20"]
                and curr["Low"]   <= curr["MA20"]
                and curr["Close"] >  curr["MA20"]
                and curr["Close"] >  curr["Open"]
            )

            # Short: Supertrend bearish (-1), MA10 < MA20, candle touched MA20 then closed below it bearish
            is_short = (
                curr["dir"] == -1
                and curr["MA10"] < curr["MA20"]
                and curr["High"]  >= curr["MA20"]
                and curr["Close"] <  curr["MA20"]
                and curr["Close"] <  curr["Open"]
            )

            row = {
                "Ticker":   ticker,
                "Price":    f"${curr['Close']:.2f}",
                "MA10":     f"${curr['MA10']:.2f}",
                "MA20":     f"${curr['MA20']:.2f}",
                "Change %": f"{chg:.2f}%",
            }

            if is_long:
                long_hits.append({**row, "Direction": "Bullish"})
            if is_short:
                short_hits.append({**row, "Direction": "Bearish"})

        except Exception as e:
            errors.append(f"{ticker}: {str(e)[:120]}")

        results_text.text(f"✅ Found: {len(long_hits)} long  |  {len(short_hits)} short")

    status_text.text("✅ Scan complete!")

    if errors:
        with st.expander(f"⚠️ {len(errors)} ticker errors"):
            st.write("\n".join(errors[:30]))

    st.header("📊 Results")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟢 Long Signals")
        if long_hits:
            df_long = pd.DataFrame(long_hits)
            st.dataframe(df_long, use_container_width=True)
            st.download_button("📥 Download CSV", df_long.to_csv(index=False),
                               f"long_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
        else:
            st.info("No long signals found.")

    with col2:
        st.subheader("🔴 Short Signals")
        if short_hits:
            df_short = pd.DataFrame(short_hits)
            st.dataframe(df_short, use_container_width=True)
            st.download_button("📥 Download CSV", df_short.to_csv(index=False),
                               f"short_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
        else:
            st.info("No short signals found.")

    st.subheader("📈 Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers Scanned",  total)
    c2.metric("Long Signals",     len(long_hits))
    c3.metric("Short Signals",    len(short_hits))
    success_rate = (total - len(errors)) / total * 100 if total else 0
    c4.metric("Success Rate",     f"{success_rate:.1f}%")

else:
    st.info("👈 Click 'Run Screener' in the sidebar to start.")
    if tickers:
        st.write(f"Loaded {len(tickers)} tickers.")
        with st.expander("First 20 tickers"):
            st.write(tickers[:20])

st.markdown("---")
st.caption(f"Last updated: {datetime.now():%Y-%m-%d %H:%M:%S}")
