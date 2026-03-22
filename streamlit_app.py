import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from streamlit_gsheets import GSheetsConnection
from datetime import datetime

# --- CONFIG & INDICATOR LOGIC ---
ATR_PERIOD = 10
FACTOR = 3.0
MA_FAST = 10
MA_SLOW = 20

def calc_supertrend(df):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / ATR_PERIOD, adjust=False).mean()
    hl2 = (high + low) / 2.0
    basic_upper, basic_lower = hl2 + FACTOR * atr, hl2 - FACTOR * atr
    
    n = len(df)
    upper, lower, direction = np.zeros(n), np.zeros(n), np.ones(n)
    for i in range(1, n):
        upper[i] = basic_upper.iloc[i] if basic_upper.iloc[i] < upper[i-1] or close.iloc[i-1] > upper[i-1] else upper[i-1]
        lower[i] = basic_lower.iloc[i] if basic_lower.iloc[i] > lower[i-1] or close.iloc[i-1] < lower[i-1] else lower[i-1]
        if direction[i-1] == 1: direction[i] = 1 if close.iloc[i] <= upper[i] else -1
        else: direction[i] = -1 if close.iloc[i] >= lower[i] else 1
    return direction

# --- STREAMLIT UI ---
st.set_page_config(page_title="Russell 3000 Screener", layout="wide")
st.title("📈 Russell 3000 Cloud Screener")
st.subheader("Supertrend + MA Confluence (Daily)")

# Initialize Google Sheets Connection
# It will look for the URL in your secrets.toml or Streamlit Cloud Secrets
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    df_tickers = conn.read(ttl="10m") # Refresh cache every 10 mins
    tickers = df_tickers["Ticker"].dropna().unique().tolist()
    st.sidebar.success(f"Connected to Cloud DB: {len(tickers)} tickers loaded.")
except Exception as e:
    st.error("Could not connect to Google Sheets. Check your Secrets configuration.")
    st.stop()

# User Controls
run_button = st.sidebar.button("🚀 Run Screener")

if run_button:
    long_hits, short_hits = [], []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, ticker in enumerate(tickers):
        status_text.text(f"Scanning {ticker} ({idx+1}/{len(tickers)})")
        progress_bar.progress((idx + 1) / len(tickers))
        
        try:
            # Download 1 year of data to calculate 20 MA and Supertrend accurately
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if len(df) < 30: continue
            
            # Calculations
            df["MA10"] = df["Close"].rolling(MA_FAST).mean()
            df["MA20"] = df["Close"].rolling(MA_SLOW).mean()
            df["dir"] = calc_supertrend(df)
            
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Logic for Long Signal
            # 1. Supertrend Uptrend (dir == -1)
            # 2. MA10 > MA20
            # 3. Touched MA20 (Low <= MA20) and closed above it
            # 4. Bullish Candle
            is_long = (
                curr["dir"] == -1 and 
                curr["MA10"] > curr["MA20"] and 
                curr["Low"] <= curr["MA20"] and 
                curr["Close"] > curr["MA20"] and
                curr["Close"] > curr["Open"]
            )
            
            if is_long:
                long_hits.append({
                    "Ticker": ticker, 
                    "Price": f"${curr['Close']:.2f}",
                    "MA10": f"${curr['MA10']:.2f}",
                    "MA20": f"${curr['MA20']:.2f}"
                })
        except:
            continue

    status_text.text("Scanning Complete!")
    
    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.header("🟢 Long Signals")
        if long_hits:
            st.table(pd.DataFrame(long_hits))
        else:
            st.write("No long signals found.")
            
    with col2:
        st.header("🔴 Short Signals")
        st.write("Short logic omitted for brevity (add same logic inverted).")

else:
    st.info("Click 'Run Screener' in the sidebar to start processing the Russell 3000 list.")
