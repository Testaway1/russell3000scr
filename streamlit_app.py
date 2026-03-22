import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from streamlit_gsheets import GSheetsConnection

# --- 1. CONFIG & ORIGINAL LOGIC ---
ATR_PERIOD = 10
FACTOR = 3.0
MA_FAST = 10
MA_SLOW = 20
MA_CROSS_LOOKBACK = 10

def calc_supertrend(df):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/ATR_PERIOD, adjust=False).mean()
    hl2 = (high + low) / 2.0
    upper, lower = hl2 + FACTOR * atr, hl2 - FACTOR * atr
    n = len(df)
    f_upper, f_lower, direction, st_line = np.zeros(n), np.zeros(n), np.ones(n), np.zeros(n)
    for i in range(1, n):
        f_upper[i] = upper.iloc[i] if upper.iloc[i] < f_upper[i-1] or close.iloc[i-1] > f_upper[i-1] else f_upper[i-1]
        f_lower[i] = lower.iloc[i] if lower.iloc[i] > f_lower[i-1] or close.iloc[i-1] < f_lower[i-1] else f_lower[i-1]
        if st_line[i-1] == f_upper[i-1]: direction[i] = 1 if close.iloc[i] <= f_upper[i] else -1
        else: direction[i] = -1 if close.iloc[i] >= f_lower[i] else 1
        st_line[i] = f_lower[i] if direction[i] == -1 else f_upper[i]
    return direction

def ma_cross_within(df, bars=MA_CROSS_LOOKBACK):
    fast, slow = df["ma_fast"].values, df["ma_slow"].values
    last_idx = len(df) - 1
    for i in range(last_idx, last_idx - bars, -1):
        if i <= 0: break
        if fast[i-1] <= slow[i-1] and fast[i] > slow[i]: return True # Bullish Cross
    return False

def get_data(symbol):
    df = yf.download(symbol, period="1y", interval="1d", auto_adjust=True, progress=False)
    if df.empty: return None
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df["ma_fast"] = df["Close"].rolling(MA_FAST).mean()
    df["ma_slow"] = df["Close"].rolling(MA_SLOW).mean()
    df["st_dir"] = calc_supertrend(df)
    return df

# --- 2. STREAMLIT UI ---
st.title("🛡️ Russell 3000 Signal & Logic Inspector")

# Google Sheets Setup
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    df_tickers = conn.read(ttl="1h")
    tickers = df_tickers["Ticker"].dropna().unique().tolist()
except:
    st.error("Google Sheets not connected. Check Secrets.")
    st.stop()

# --- BKNG LOGIC INSPECTOR ---
st.header("🔍 BKNG Logic Breakdown")
if st.button("Inspect BKNG Now"):
    df = get_data("BKNG")
    if df is not None:
        curr = df.iloc[-1]
        bull_cross = ma_cross_within(df)
        
        # 5 Data Items Explanation
        checks = [
            {"Rule": "1. Supertrend Direction", "Val": f"{int(curr['st_dir'])}", "Req": "-1", "Status": curr['st_dir'] == -1, "Info": "Trend must be bullish."},
            {"Rule": "2. MA Alignment", "Val": f"10MA: {curr['ma_fast']:.2f} > 20MA: {curr['ma_slow']:.2f}", "Req": "10MA > 20MA", "Status": curr['ma_fast'] > curr['ma_slow'], "Info": "Momentum check."},
            {"Rule": "3. Recent Crossover", "Val": "Yes" if bull_cross else "No", "Req": "Within 10 Days", "Status": bull_cross, "Info": "Prevents entering late trades."},
            {"Rule": "4. The 'Touch' Rule", "Val": f"Low: {curr['Low']:.2f} vs 20MA: {curr['ma_slow']:.2f}", "Req": "Low <= MA20", "Status": curr['Low'] <= curr['ma_slow'], "Info": "Requires a dip to support."},
            {"Rule": "5. Candle Color", "Val": f"Open: {curr['Open']:.2f} vs Close: {curr['Close']:.2f}", "Req": "Close > Open", "Status": curr['Close'] > curr['Open'], "Info": "Ensures current buying pressure."}
        ]
        
        for c in checks:
            color = "green" if c['Status'] else "red"
            st.markdown(f"#### {'✅' if c['Status'] else '❌'} {c['Rule']}")
            st.write(f"**Current:** {c['Val']} | **Required:** {c['Req']}")
            st.caption(f"_{c['Info']}_")
            st.divider()
    else:
        st.error("Could not fetch BKNG data.")

# --- FULL SCAN SECTION ---
if st.sidebar.button("Run Full Russell 3000 Scan"):
    # (Loop through tickers and display table as before)
    pass
