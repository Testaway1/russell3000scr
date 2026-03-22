import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from streamlit_gsheets import GSheetsConnection
from datetime import datetime

# --- 1. CONFIG (Mirroring your original script) ---
ATR_PERIOD        = 10
FACTOR            = 3.0
MA_FAST           = 10
MA_SLOW           = 20
MA_CROSS_LOOKBACK = 10

# --- 2. EXACT LOGIC FROM YOUR FILE ---
def calc_supertrend(df):
    """Exact logic from supertrend_trial_russell3000.py"""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / ATR_PERIOD, adjust=False).mean()
    hl2 = (high + low) / 2.0
    basic_upper, basic_lower = hl2 + FACTOR * atr, hl2 - FACTOR * atr

    n = len(df)
    final_upper, final_lower = np.zeros(n), np.zeros(n)
    direction, st_line = np.ones(n), np.zeros(n)

    for i in range(1, n):
        final_upper[i] = basic_upper.iloc[i] if basic_upper.iloc[i] < final_upper[i-1] or close.iloc[i-1] > final_upper[i-1] else final_upper[i-1]
        final_lower[i] = basic_lower.iloc[i] if basic_lower.iloc[i] > final_lower[i-1] or close.iloc[i-1] < final_lower[i-1] else final_lower[i-1]
        
        if st_line[i-1] == final_upper[i-1]:
            direction[i] = 1 if close.iloc[i] <= final_upper[i] else -1
        else:
            direction[i] = -1 if close.iloc[i] >= final_lower[i] else 1
        st_line[i] = final_lower[i] if direction[i] == -1 else final_upper[i]
    return pd.Series(direction, index=df.index)

def ma_cross_within(df, bars=MA_CROSS_LOOKBACK):
    """Exact logic from supertrend_trial_russell3000.py"""
    fast, slow = df["ma_fast"].values, df["ma_slow"].values
    last_bar = len(df) - 1
    for i in range(last_bar, 0, -1):
        if (last_bar - i) > bars: break
        if pd.isna(fast[i]) or pd.isna(slow[i]) or pd.isna(fast[i-1]) or pd.isna(slow[i-1]): continue
        if fast[i-1] <= slow[i-1] and fast[i] > slow[i]: return True, False
        if fast[i-1] >= slow[i-1] and fast[i] < slow[i]: return False, True
    return False, False

def process_ticker(symbol):
    """Downloads data and evaluates all 5 criteria."""
    try:
        df = yf.download(symbol, period="1y", interval="1d", auto_adjust=True, progress=False)
        if df.empty or len(df) < 40: return None
        
        # Robust column flattening
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
            
        df["ma_fast"] = df["Close"].rolling(MA_FAST).mean()
        df["ma_slow"] = df["Close"].rolling(MA_SLOW).mean()
        df["direction"] = calc_supertrend(df).values
        
        bull_x, bear_x = ma_cross_within(df)
        curr = df.iloc[-1]
        
        # The 5 LONG Criteria
        c1 = (curr["direction"] == -1)
        c2 = (curr["ma_fast"] > curr["ma_slow"])
        c3 = bull_x
        c4 = (curr["Low"] <= curr["ma_slow"] and curr["Close"] > curr["ma_slow"])
        c5 = (curr["Close"] > curr["Open"])
        
        is_long = c1 and c2 and c3 and c4 and c5
        
        # The 5 SHORT Criteria
        s1 = (curr["direction"] == 1)
        s2 = (curr["ma_fast"] < curr["ma_slow"])
        s3 = bear_x
        s4 = (curr["High"] >= curr["ma_slow"] and curr["Close"] < curr["ma_slow"])
        s5 = (curr["Close"] < curr["Open"])
        
        is_short = s1 and s2 and s3 and s4 and s5

        res = {"Ticker": symbol, "Price": round(float(curr["Close"]), 2), "Date": df.index[-1].strftime('%Y-%m-%d')}
        if is_long: return {**res, "Signal": "LONG"}, [c1, c2, c3, c4, c5]
        if is_short: return {**res, "Signal": "SHORT"}, [s1, s2, s3, s4, s5]
        return None, [c1, c2, c3, c4, c5] # Return False criteria for debugging
    except:
        return None, []

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Russell 3000 Screener", layout="wide")
st.title("📊 Supertrend + MA Screener")

# Google Sheets Connection
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    df_tickers = conn.read(ttl="1h")
    tickers = df_tickers["Ticker"].dropna().unique().tolist()
    st.sidebar.success(f"Loaded {len(tickers)} tickers")
except:
    st.sidebar.error("Could not connect to Google Sheets.")
    st.stop()

# --- BKNG DEBUG SECTION ---
st.header("🔍 Debug: Why is BKNG missing?")
if st.button("Analyze BKNG Logic"):
    data, checks = process_ticker("BKNG")
    cols = st.columns(5)
    labels = ["ST Up", "10MA > 20MA", "X-Over 10d", "Touch 20MA", "Bullish Candle"]
    for i, (label, val) in enumerate(zip(labels, checks)):
        cols[i].metric(label, "Pass" if val else "Fail")
    if data:
        st.success("BKNG found a signal!")
        st.table([data])
    else:
        st.warning("BKNG did not meet all 5 criteria today.")

# --- FULL SCAN ---
if st.sidebar.button("🚀 Run Full Scan"):
    results = []
    prog = st.progress(0)
    status = st.empty()
    
    for i, t in enumerate(tickers):
        status.text(f"Scanning {t}...")
        prog.progress((i+1)/len(tickers))
        sig_data, _ = process_ticker(t)
        if sig_data:
            results.append(sig_data)
            
    st.write("### 📈 Screening Results")
    if results:
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.info("No signals found.")
