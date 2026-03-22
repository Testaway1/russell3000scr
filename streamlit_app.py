import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from streamlit_gsheets import GSheetsConnection
from datetime import datetime

# --- CONFIG (Matches your original file) ---
ATR_PERIOD = 10
FACTOR = 3.0
MA_FAST = 10
MA_SLOW = 20
LOOKBACK_XOVER = 10

def calc_supertrend(df):
    df = df.copy()
    # Ensure column names are clean (yfinance sometimes adds levels)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
        
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
        if direction[i-1] == 1:
            direction[i] = 1 if close.iloc[i] <= upper[i] else -1
        else:
            direction[i] = -1 if close.iloc[i] >= lower[i] else 1
    return direction

def run_logic(symbol):
    """Refined logic to match your PC script exactly."""
    try:
        # Get 1 year of data to ensure MAs and ATR are fully 'warmed up'
        data = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
        if data.empty or len(data) < 40: return None
        
        # Clean Multi-Index if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(-1)

        data["MA10"] = data["Close"].rolling(MA_FAST).mean()
        data["MA20"] = data["Close"].rolling(MA_SLOW).mean()
        data["ST_DIR"] = calc_supertrend(data)
        
        # Crossover Logic
        data["XOVER_LONG"] = (data["MA10"] > data["MA20"]) & (data["MA10"].shift(1) <= data["MA20"].shift(1))
        recent_long_xover = data["XOVER_LONG"].tail(LOOKBACK_XOVER).any()

        curr = data.iloc[-1]
        
        # Your specific 5-point criteria
        is_long = (
            curr["ST_DIR"] == -1 and             # 1. Supertrend Uptrend
            curr["MA10"] > curr["MA20"] and      # 2. MA10 > MA20
            recent_long_xover and                # 3. Recent Crossover
            curr["Low"] <= curr["MA20"] and      # 4. Touched 20MA
            curr["Close"] > curr["MA20"] and     # 4. Closed above 20MA
            curr["Close"] > curr["Open"]         # 5. Bullish Candle
        )
        
        if is_long:
            return {"Ticker": symbol, "Price": round(float(curr["Close"]), 2), "MA20": round(float(curr["MA20"]), 2)}
    except:
        return None
    return None

# --- UI ---
st.set_page_config(page_title="Screener", layout="wide")
st.title("🛡️ Supertrend + MA Screener")

col_a, col_b = st.sidebar.columns(2)
test_bkng = col_a.button("Test BKNG")
run_full = col_b.button("Run Full Scan")

# Logic for Test Button
if test_bkng:
    st.write("### 🧪 Testing BKNG...")
    result = run_logic("BKNG")
    if result:
        st.success("✅ BKNG triggered a signal!")
        st.json(result)
    else:
        st.warning("❌ BKNG did not trigger a signal. It may not currently meet all 5 criteria.")

# Logic for Full Scan
if run_full:
    # (Insert the Google Sheets loading logic here as established in previous turns)
    st.info("Starting Russell 3000 Scan...")
    # ... loop through ticker_list and call run_logic(symbol) ...
