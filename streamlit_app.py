import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from streamlit_gsheets import GSheetsConnection
from datetime import datetime, date, timedelta

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
ATR_PERIOD        = 10
FACTOR            = 3.0
MA_FAST           = 10
MA_SLOW           = 20
MA_TREND          = 200
MA_CROSS_LOOKBACK = 10
INTERVAL          = "1d"
BATCH_SIZE        = 50
LOOKBACK_52W      = 252

SHORT_DIST_MIN    = 40.0           # short filter: >40% below 52w high
SHORT_CHANGE_MAX  = -50.0          # short filter: down >50% on year


# ─────────────────────────────────────────────
#  DATA DOWNLOAD
# ─────────────────────────────────────────────
def download_batch(tickers: list[str],
                   as_of_date: date = None) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}

    end_date   = as_of_date if as_of_date else date.today()
    start_date = end_date - timedelta(days=730)   # 2y for MA200 + 52w warmup
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

    result   = {}
    min_bars = MA_SLOW + ATR_PERIOD + 5

    if raw.empty:
        return result

    if len(tickers) == 1:
        tkr = tickers[0]
        try:
            df = raw.dropna()
            if len(df) >= min_bars:
                result[tkr] = df
        except Exception:
            pass
        return result

    cols = raw.columns
    if not isinstance(cols, pd.MultiIndex):
        try:
            df = raw.dropna()
            if len(df) >= min_bars:
                result[tickers[0]] = df
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
            if len(sub) >= min_bars:
                result[tkr] = sub
        except Exception:
            continue

    return result


# ─────────────────────────────────────────────
#  SUPERTREND
# ─────────────────────────────────────────────
def calc_supertrend(df: pd.DataFrame,
                    atr_period: int = ATR_PERIOD,
                    factor: float   = FACTOR) -> pd.Series | None:
    try:
        high  = df["High"]
        low   = df["Low"]
        close = df["Close"]

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
        direction   = np.ones(n)
        st_line     = np.zeros(n)

        for i in range(1, n):
            if basic_upper.iloc[i] < final_upper[i-1] or close.iloc[i-1] > final_upper[i-1]:
                final_upper[i] = basic_upper.iloc[i]
            else:
                final_upper[i] = final_upper[i-1]
            if basic_lower.iloc[i] > final_lower[i-1] or close.iloc[i-1] < final_lower[i-1]:
                final_lower[i] = basic_lower.iloc[i]
            else:
                final_lower[i] = final_lower[i-1]
            if st_line[i-1] == final_upper[i-1]:
                direction[i] = 1 if close.iloc[i] <= final_upper[i] else -1
            else:
                direction[i] = -1 if close.iloc[i] >= final_lower[i] else 1
            st_line[i] = final_lower[i] if direction[i] == -1 else final_upper[i]

        return pd.Series(direction, index=df.index, name="direction")

    except Exception as e:
        st.error(f"Supertrend error: {e}")
        return None


# ─────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma_fast"]      = df["Close"].rolling(MA_FAST).mean()
    df["ma_slow"]      = df["Close"].rolling(MA_SLOW).mean()
    df["ma_trend"]     = df["Close"].rolling(MA_TREND).mean()
    df["high_52w"]     = df["Close"].rolling(LOOKBACK_52W).max()
    df["close_252ago"] = df["Close"].shift(LOOKBACK_52W)
    result = calc_supertrend(df, ATR_PERIOD, FACTOR)
    df["direction"] = result.values if result is not None else np.nan
    return df


# ─────────────────────────────────────────────
#  MA CROSSOVER CHECK
# ─────────────────────────────────────────────
def ma_cross_within(df: pd.DataFrame,
                    bars: int = MA_CROSS_LOOKBACK) -> tuple[bool, bool, int]:
    fast     = df["ma_fast"].values
    slow     = df["ma_slow"].values
    last_bar = len(df) - 1

    for i in range(last_bar, 0, -1):
        if (last_bar - i) > bars:
            break
        if any(pd.isna(v) for v in [fast[i], slow[i], fast[i-1], slow[i-1]]):
            continue
        if fast[i-1] <= slow[i-1] and fast[i] > slow[i]:
            return True, False, last_bar - i
        if fast[i-1] >= slow[i-1] and fast[i] < slow[i]:
            return False, True, last_bar - i

    return False, False, -1


# ─────────────────────────────────────────────
#  LONG SIGNAL DETECTION
# ─────────────────────────────────────────────
def check_long_signal(df: pd.DataFrame,
                      cross_lookback: int = MA_CROSS_LOOKBACK) -> tuple[bool, int]:
    if len(df) < MA_SLOW + cross_lookback + 2:
        return False, -1

    row = df.iloc[-1]
    if any(pd.isna(row.get(c, float("nan")))
           for c in ["ma_fast", "ma_slow", "direction"]):
        return False, -1

    bullish_cross, _, days_ago = ma_cross_within(df, cross_lookback)

    signal = (
        row["direction"] == -1
        and row["ma_fast"]  > row["ma_slow"]
        and bullish_cross
        and row["Low"]     <= row["ma_slow"]
        and row["Close"]    > row["ma_slow"]
        and row["Close"]    > row["Open"]
    )
    return signal, days_ago


# ─────────────────────────────────────────────
#  SHORT SIGNAL DETECTION
# ─────────────────────────────────────────────
def check_short_signal(df: pd.DataFrame,
                       cross_lookback: int = MA_CROSS_LOOKBACK) -> tuple[bool, int, float, float]:
    if len(df) < MA_TREND + MA_CROSS_LOOKBACK + 5:
        return False, -1, 0.0, 0.0

    row = df.iloc[-1]
    for col in ["ma_fast", "ma_slow", "direction", "high_52w", "close_252ago"]:
        if pd.isna(row.get(col, float("nan"))):
            return False, -1, 0.0, 0.0

    price        = float(row["Close"])
    high_52w     = float(row["high_52w"])
    close_252ago = float(row["close_252ago"])

    if high_52w <= 0 or close_252ago <= 0:
        return False, -1, 0.0, 0.0

    dist_pct   = (high_52w - price) / high_52w * 100.0
    change_pct = (price - close_252ago) / close_252ago * 100.0

    if dist_pct < SHORT_DIST_MIN or change_pct >= SHORT_CHANGE_MAX:
        return False, -1, round(dist_pct, 1), round(change_pct, 1)

    _, bearish_cross, days_ago = ma_cross_within(df, cross_lookback)

    signal = (
        row["direction"] == 1
        and row["ma_fast"]  < row["ma_slow"]
        and bearish_cross
        and row["High"]    >= row["ma_slow"]
        and row["Close"]    < row["ma_slow"]
        and row["Close"]    < row["Open"]
    )
    return signal, days_ago, round(dist_pct, 1), round(change_pct, 1)


# ─────────────────────────────────────────────
#  SPY REGIME CHECK
# ─────────────────────────────────────────────
def get_spy_regime(as_of_date: date) -> bool | None:
    try:
        end_date   = as_of_date
        start_date = end_date - timedelta(days=90)
        end_fetch  = end_date + timedelta(days=1)

        raw = yf.download(
            "SPY",
            start=start_date.strftime("%Y-%m-%d"),
            end=end_fetch.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if raw is None or raw.empty:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw      = raw.dropna(subset=["Close"])
        raw      = raw[raw.index <= pd.Timestamp(as_of_date)]

        if len(raw) < MA_SLOW + 2:
            return None

        ma_fast = raw["Close"].rolling(MA_FAST).mean().iloc[-1]
        ma_slow = raw["Close"].rolling(MA_SLOW).mean().iloc[-1]

        if pd.isna(ma_fast) or pd.isna(ma_slow):
            return None

        return bool(ma_fast > ma_slow)

    except Exception as e:
        st.warning(f"SPY regime fetch error: {e}")
        return None


# ─────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Russell 3000 Screener", layout="wide")
st.title("📈 Russell 3000 Cloud Screener")
st.subheader("Supertrend + MA Confluence — Long & Short (Daily)")

# ── Google Sheets connection ──────────────────────────────────────────────────
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
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
)

batch_size = st.sidebar.number_input(
    "Batch size", min_value=1, max_value=200, value=BATCH_SIZE,
)

st.sidebar.markdown("---")

as_of_date = st.sidebar.date_input(
    "Run screener as of",
    value=date.today(),
    max_value=date.today(),
)

if as_of_date > date.today():
    st.sidebar.warning("Date cannot be in the future — defaulting to today.")
    as_of_date = date.today()
if as_of_date.weekday() >= 5:
    st.sidebar.warning(
        f"{as_of_date.strftime('%A %d %b %Y')} is a weekend. "
        "yfinance will use the last available trading day on or before this date."
    )

st.sidebar.markdown("---")

st.sidebar.subheader("📡 SPY Regime Filter")
use_spy_regime = st.sidebar.checkbox(
    "SPY Regime Filter (MA10 > MA20)",
    value=True,
    help="Only show LONG signals when SPY MA10 > MA20. Short signals always shown."
)

st.sidebar.markdown("---")

st.sidebar.subheader("🔻 Short Signal Filters")
st.sidebar.caption(
    f"Applied automatically:\n"
    f"• >{SHORT_DIST_MIN:.0f}% below 52-week high\n"
    f"• Down >{abs(SHORT_CHANGE_MAX):.0f}% on the year"
)

st.sidebar.markdown("---")

st.sidebar.subheader("💰 Capital Allocation")
capital = st.sidebar.number_input(
    "Total capital ($)", min_value=1_000, max_value=100_000_000,
    value=100_000, step=1_000,
)
max_risk_pct = st.sidebar.number_input(
    "Max risk per trade (% of capital)",
    min_value=0.1, max_value=5.0, value=1.0, step=0.1,
    help="Applies to both long and short sizing."
)
max_position_pct = st.sidebar.number_input(
    "Max position size (% of capital)",
    min_value=1.0, max_value=100.0, value=20.0, step=1.0,
)

max_risk_dollars     = capital * (max_risk_pct     / 100)
max_position_dollars = capital * (max_position_pct / 100)

run_button = st.sidebar.button("🚀 Run Screener")

# ── Main scan ─────────────────────────────────────────────────────────────────
if run_button:
    long_hits, short_hits, errors = [], [], []
    batches       = [tickers[i : i + int(batch_size)]
                     for i in range(0, len(tickers), int(batch_size))]
    total         = len(tickers)
    total_batches = len(batches)
    tickers_done  = 0

    # SPY regime check
    spy_regime_ok = None
    if use_spy_regime:
        with st.spinner("Checking SPY regime (MA10 vs MA20)…"):
            spy_regime_ok = get_spy_regime(as_of_date)
        if spy_regime_ok is None:
            st.warning("⚠️ Could not fetch SPY data — regime filter disabled.")
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
                f"📊 Scanning {ticker} ({tickers_done}/{total}, "
                f"batch {b_idx + 1}/{total_batches})"
            )

            if ticker not in price_data:
                continue

            try:
                raw_df   = price_data[ticker]
                as_of_ts = pd.Timestamp(as_of_date)
                raw_df   = raw_df[raw_df.index <= as_of_ts]
                if len(raw_df) < MA_SLOW + MA_CROSS_LOOKBACK + 2:
                    continue
                df = calc_indicators(raw_df)
            except Exception as e:
                errors.append(f"{ticker}: {str(e)[:120]}")
                continue

            last  = df.iloc[-1]
            prev  = df.iloc[-2]
            chg   = (last["Close"] - prev["Close"]) / prev["Close"] * 100
            price = float(last["Close"])

            # ── Long ─────────────────────────────────────────────────────────
            try:
                long_sig, long_days = check_long_signal(df, int(cross_window))
            except Exception:
                long_sig = False

            if long_sig and not (use_spy_regime and spy_regime_ok is False):
                sl  = float(last["Low"])
                rps = price - sl
                if rps > 0:
                    shares = int(min(max_risk_dollars / rps,
                                     max_position_dollars / price))
                else:
                    shares = 0
                long_hits.append({
                    "Ticker"           : ticker,
                    "Date"             : df.index[-1].strftime("%Y-%m-%d"),
                    "Price ($)"        : round(price, 2),
                    "Stop Loss ($)"    : round(sl, 2),
                    "Risk/Share ($)"   : round(rps, 2),
                    "Shares"           : shares,
                    "Position ($)"     : round(shares * price, 2),
                    "Risk ($)"         : round(shares * rps, 2),
                    "MA10 ($)"         : round(float(last["ma_fast"]), 2),
                    "MA20 ($)"         : round(float(last["ma_slow"]), 2),
                    "Change %"         : round(chg, 2),
                    "Cross (days ago)" : long_days,
                })

            # ── Short ────────────────────────────────────────────────────────
            try:
                short_sig, short_days, dist_pct, change_pct = check_short_signal(
                    df, int(cross_window)
                )
            except Exception:
                short_sig = False

            if short_sig:
                sl  = float(last["High"])
                rps = sl - price
                if rps > 0:
                    shares = int(min(max_risk_dollars / rps,
                                     max_position_dollars / price))
                else:
                    shares = 0
                tp = price - 2 * rps
                short_hits.append({
                    "Ticker"            : ticker,
                    "Date"              : df.index[-1].strftime("%Y-%m-%d"),
                    "Price ($)"         : round(price, 2),
                    "Stop Loss ($)"     : round(sl, 2),
                    "TP 2R ($)"         : round(tp, 2),
                    "Risk/Share ($)"    : round(rps, 2),
                    "Shares"            : shares,
                    "Position ($)"      : round(shares * price, 2),
                    "Risk ($)"          : round(shares * rps, 2),
                    "MA10 ($)"          : round(float(last["ma_fast"]), 2),
                    "MA20 ($)"          : round(float(last["ma_slow"]), 2),
                    "Change %"          : round(chg, 2),
                    "Dist 52W High (%)" : dist_pct,
                    "1Y Change (%)"     : change_pct,
                    "Cross (days ago)"  : short_days,
                })

        results_text.text(
            f"✅ Found: {len(long_hits)} long  |  {len(short_hits)} short"
        )

    status_text.text("✅ Scan complete!")

    if errors:
        with st.expander(f"⚠️ {len(errors)} ticker errors"):
            st.write("\n".join(errors[:30]))

    st.header(f"📊 Results — as of {as_of_date.strftime('%d %b %Y')}")

    # SPY banner
    if use_spy_regime:
        if spy_regime_ok is True:
            st.info("📡 SPY Regime: **BULLISH** (MA10 > MA20) — long signals shown.")
        elif spy_regime_ok is False:
            st.warning("📡 SPY Regime: **BEARISH** (MA10 ≤ MA20) — long signals suppressed.")
        else:
            st.warning("📡 SPY Regime: **UNAVAILABLE** — regime filter bypassed.")
    else:
        st.info("📡 SPY Regime Filter: **OFF** — all long signals shown.")

    # Summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers Scanned",  total)
    c2.metric("🟢 Long Signals",  len(long_hits))
    c3.metric("🔴 Short Signals", len(short_hits))
    c4.metric("Success Rate",     f"{(total - len(errors)) / total * 100:.1f}%" if total else "—")

    st.markdown("---")
    st.caption(
        f"Capital: **${capital:,.0f}**  |  "
        f"Max risk/trade: **${max_risk_dollars:,.0f}** ({max_risk_pct}%)  |  "
        f"Max position: **${max_position_dollars:,.0f}** ({max_position_pct}%)"
    )

    # ── Long table ────────────────────────────────────────────────────────────
    st.subheader("🟢 Long Signals")
    st.caption("SPY regime filter ON · MA200 filter OFF · Stop = signal day Low")

    if long_hits:
        df_long     = pd.DataFrame(long_hits)
        long_cols   = ["Ticker", "Date", "Price ($)", "Stop Loss ($)",
                        "Shares", "Position ($)", "Risk ($)",
                        "MA10 ($)", "MA20 ($)", "Change %", "Cross (days ago)"]
        fmt_dollar  = ["Price ($)", "Stop Loss ($)", "MA10 ($)", "MA20 ($)"]
        fmt_money   = ["Position ($)", "Risk ($)"]
        st.dataframe(
            df_long[long_cols].style
                .format({c: "${:,.2f}" for c in fmt_dollar})
                .format({c: "${:,.0f}" for c in fmt_money})
                .format({"Change %": "{:+.2f}%", "Shares": "{:,}"}),
            use_container_width=True, hide_index=True,
        )
        tp  = df_long["Position ($)"].sum()
        tr  = df_long["Risk ($)"].sum()
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Signals found",   len(long_hits))
        ec2.metric("Total deployed",  f"${tp:,.0f}  ({tp/capital*100:.1f}%)")
        ec3.metric("Total $ at risk", f"${tr:,.0f}  ({tr/capital*100:.1f}%)")
        ec4.metric("Avg risk/trade",  f"${tr/len(long_hits):,.0f}")
        st.download_button(
            "📥 Download Long Signals CSV",
            df_long.to_csv(index=False),
            f"long_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv",
        )
    else:
        if use_spy_regime and spy_regime_ok is False:
            st.info("No long signals shown — SPY regime is bearish.")
        else:
            st.info("No long signals found.")

    st.markdown("---")

    # ── Short table ───────────────────────────────────────────────────────────
    st.subheader("🔴 Short Signals")
    st.caption(
        f"No regime filter · Stop = signal day High · TP = 2R  · "
        f"Filters: 52W dist >{SHORT_DIST_MIN:.0f}% | 1Y change <{SHORT_CHANGE_MAX:.0f}%"
    )

    if short_hits:
        df_short    = pd.DataFrame(short_hits)
        short_cols  = ["Ticker", "Date", "Price ($)", "Stop Loss ($)", "TP 2R ($)",
                        "Shares", "Position ($)", "Risk ($)",
                        "MA10 ($)", "MA20 ($)", "Change %",
                        "Dist 52W High (%)", "1Y Change (%)", "Cross (days ago)"]
        fmt_dollar_s = ["Price ($)", "Stop Loss ($)", "TP 2R ($)", "MA10 ($)", "MA20 ($)"]
        fmt_money_s  = ["Position ($)", "Risk ($)"]
        st.dataframe(
            df_short[short_cols].style
                .format({c: "${:,.2f}" for c in fmt_dollar_s})
                .format({c: "${:,.0f}" for c in fmt_money_s})
                .format({
                    "Change %"          : "{:+.2f}%",
                    "Dist 52W High (%)" : "{:.1f}%",
                    "1Y Change (%)"     : "{:.1f}%",
                    "Shares"            : "{:,}",
                }),
            use_container_width=True, hide_index=True,
        )
        tp_s  = df_short["Position ($)"].sum()
        tr_s  = df_short["Risk ($)"].sum()
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Signals found",   len(short_hits))
        sc2.metric("Total exposure",  f"${tp_s:,.0f}  ({tp_s/capital*100:.1f}%)")
        sc3.metric("Total $ at risk", f"${tr_s:,.0f}  ({tr_s/capital*100:.1f}%)")
        sc4.metric("Avg risk/trade",  f"${tr_s/len(short_hits):,.0f}")
        st.download_button(
            "📥 Download Short Signals CSV",
            df_short.to_csv(index=False),
            f"short_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv",
        )
    else:
        st.info("No short signals found.")

else:
    st.info("👈 Click 'Run Screener' in the sidebar to start.")
    if tickers:
        st.write(f"Loaded {len(tickers)} tickers.")
        with st.expander("First 20 tickers"):
            st.write(tickers[:20])

st.markdown("---")
st.caption(f"Last updated: {datetime.now():%Y-%m-%d %H:%M:%S}")
