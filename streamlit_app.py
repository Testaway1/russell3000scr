import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from streamlit_gsheets import GSheetsConnection

# --- REPLICATING YOUR EXACT PC LOGIC ---
def calc_supertrend(df):
    df = df.copy()
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/10, adjust=False).mean()
    hl2 = (high + low) / 2.0
    upper, lower = hl2 + 3.0 * atr, hl2 - 3.0 * atr
    
    n = len(df)
    f_upper, f_lower, direction = np.zeros(n), np.zeros(n), np.ones(n)
    st_line = np.zeros(n)

    for i in range(1, n):
        f_upper[i] = upper.iloc[i] if upper.iloc[i] < f_upper[i-1] or close.iloc[i-1] > f_upper[i-1] else f_upper[i-1]
        f_lower[i] = lower.iloc[i] if lower.iloc[i] > f_lower[i-1] or close.iloc[i-1] < f_lower[i-1] else f_lower[i-1]
        
        if st_line[i-1] == f_upper[i-1]:
            direction[i] = 1 if close.iloc[i] <= f_upper[i] else -1
        else:
            direction[i] = -1 if close.iloc[i] >= f_lower[i] else 1
        st_line[i] = f_lower[i] if direction[i] == -1 else f_upper[i]
    return direction

def debug_ticker(symbol):
    data = yf.download(symbol, period="60d", interval="1d", auto_adjust=True, progress=False)
    if data.empty: return None
    
    # Fix potential MultiIndex issues from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(-1)

    df = data.copy()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["ST_DIR"] = calc_supertrend(df)
    
    # Crossover check
    df["XOVER"] = (df["MA10"] > df["MA20"]) & (df["MA10"].shift(1) <= df["MA20"].shift(1))
    recent_xover = df["XOVER"].tail(10).any()
    
    curr = df.iloc[-1]
    
    # Rule Checks
    checks = {
        "1. ST Uptrend (-1)": curr["ST_DIR"] == -1,
        "2. MA10 > MA20": curr["MA10"] > curr["MA20"],
        "3. Xover in last 10d": recent_xover,
        "4. Low <= MA20 < Close": (curr["Low"] <= curr["MA20"]) and (curr["Close"] > curr["MA20"]),
        "5. Bullish Candle": curr["Close"] > curr["Open"]
    }
    
    return curr, checks, df.tail(5)

# --- UI ---
st.title("BKNG Debugger")

if st.button("Analyze BKNG Now"):
    curr, checks, history = debug_ticker("BKNG")
    
    st.subheader("Final Rule Tally")
    for rule, passed in checks.items():
        st.write(f"{'✅' if passed else '❌'} {rule}")
    
    st.subheader("Current Values")
    st.write(f"**Price:** {curr['Close']:.2f} | **Low:** {curr['Low']:.2f} | **MA20:** {curr['MA20']:.2f}")
    
    st.subheader("Recent Data (Last 5 Days)")
    st.dataframe(history[['Open', 'High', 'Low', 'Close', 'MA10', 'MA20', 'ST_DIR']])
