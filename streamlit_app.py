import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from streamlit_gsheets import GSheetsConnection
from datetime import datetime

# --- CONFIG (Matches your original file supertrend_trial_russell3000.py) ---
ATR_PERIOD        = 10
FACTOR            = 3.0
MA_FAST           = 10
MA_SLOW           = 20
MA_CROSS_LOOKBACK = 10

# --- EXACT INDICATOR LOGIC FROM YOUR SCRIPT ---
def calc_supertrend(df, atr_period=ATR_PERIOD, factor=FACTOR):
    """Matches your PC script exactly using the st_line state tracker."""
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

def robust_download(symbol):
    """Downloads and flattens columns to prevent KeyError: 'Close'."""
    # Use 1y to ensure ATR and MAs are fully 'warmed up'
    data = yf.download(symbol, period="1y", interval="1d", auto_adjust=True, progress=False)
    if data.empty:
        return None
    
    # Flatten MultiIndex columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        # Find which level contains 'Close', 'High', etc.
        if 'Close' in data.columns.get_level_values(0):
            data.columns = data.columns.get_level_values(0)
        else:
            data.columns = data.columns.get_level_values(-1)
            
    return data.dropna()

def check_signal(df):
    """Implements the 5-point confluence logic from your script."""
    if len(df) < MA_SLOW + MA_CROSS_LOOKBACK + 2:
        return None

    df = df.copy()
    df["ma_fast"] = df["Close"].rolling(MA_FAST).mean()
    df["ma_slow"] = df["Close"].rolling(MA_SLOW).mean()
    df["direction"] = calc_supertrend(df).values
    
    row = df.iloc[-1]
    
    # 10-day Crossover Check
    fast, slow = df["ma_fast"].values, df["ma_slow"].values
    bullish_cross = False
    for i in range(len(df)-1, len(df)-1-MA_CROSS_LOOKBACK, -1):
        if i <= 0: break
        if fast[i-1] <= slow[i-1] and fast[i] > slow[i]:
            bullish_cross = True; break

    # LONG Criteria
    is_long = (
        row["direction"] == -1 and
        row["ma_fast"] > row["ma_slow"] and
        bullish_cross and
        row["Low"] <= row["ma_slow"] and
        row["Close"] > row["ma_slow"] and
        row["Close"] > row["Open"]
    )
    
    if is_long: return "LONG"
    return None

# --- STREAMLIT UI ---
st.title("📈 Russell 3000 Signal Screener")

if st.button("Analyze BKNG (Debug Mode)"):
    st.write("Fetching data for BKNG...")
    df = robust_download("BKNG")
    if df is not None:
        sig = check_signal(df)
        st.write(f"**Current Signal:** {sig if sig else 'None'}")
        
        # Display the data to see the 'Touch'
        last_row = df.iloc[-1]
        st.write(f"Price: {last_row['Close']:.2f} | Low: {last_row['Low']:.2f} | MA20: {df['Close'].rolling(20).mean().iloc[-1]:.2f}")
        st.dataframe(df.tail(10))
    else:
        st.error("Failed to download data for BKNG.")
