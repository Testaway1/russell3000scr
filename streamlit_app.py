import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- 1. THE LOGIC (RE-SYNCED TO YOUR PC) ---
def calc_supertrend(df):
    # Ensure we are working with clean columns
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 10, adjust=False).mean()
    hl2 = (high + low) / 2.0
    upper, lower = hl2 + 3.0 * atr, hl2 - 3.0 * atr
    n = len(df)
    f_upper, f_lower, direction, st_line = np.zeros(n), np.zeros(n), np.ones(n), np.zeros(n)
    for i in range(1, n):
        f_upper[i] = upper.iloc[i] if upper.iloc[i] < f_upper[i-1] or close.iloc[i-1] > f_upper[i-1] else f_upper[i-1]
        f_lower[i] = lower.iloc[i] if lower.iloc[i] > f_lower[i-1] or close.iloc[i-1] < f_lower[i-1] else f_lower[i-1]
        if st_line[i-1] == f_upper[i-1]:
            direction[i] = 1 if close.iloc[i] <= f_upper[i] else -1
        else:
            direction[i] = -1 if close.iloc[i] >= f_lower[i] else 1
        st_line[i] = f_lower[i] if direction[i] == -1 else f_upper[i]
    return direction

def get_clean_data(symbol):
    df = yf.download(symbol, period="1y", interval="1d", auto_adjust=True, progress=False)
    if df.empty: return None
    
    # NUCLEAR OPTION: Force columns to be simple strings
    # This fixes the 'Close' KeyError regardless of yfinance version
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.columns = df.columns.str.capitalize() # Ensures 'Close' not 'close'
    
    return df

# --- 2. THE DEBUGGER ---
st.title("BKNG Signal Deep-Dive")

if st.button("Deep Dive: BKNG Current Values"):
    df = get_clean_data("BKNG")
    
    if df is not None:
        # Calculations
        df["MA10"] = df["Close"].rolling(10).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["ST_DIR"] = calc_supertrend(df)
        
        # Crossover check (last 10 days)
        x_up = (df["MA10"] > df["MA20"]) & (df["MA10"].shift(1) <= df["MA20"].shift(1))
        recent_xover = x_up.tail(10).any()
        
        curr = df.iloc[-1]
        
        # Define the 5 Data Items
        results = [
            {"Rule": "1. Supertrend Direction", "Value": f"{int(curr['ST_DIR'])}", "Goal": "-1", "Status": curr["ST_DIR"] == -1, 
             "Explain": "Checks if the overall volatility trend is bullish (-1)."},
            {"Rule": "2. MA Alignment", "Value": f"10MA: {curr['MA10']:.2f} > 20MA: {curr['MA20']:.2f}", "Goal": "True", "Status": curr["MA10"] > curr["MA20"],
             "Explain": "Short-term trend must be above long-term trend."},
            {"Rule": "3. Recent Crossover", "Value": "Yes" if recent_xover else "No", "Goal": "Yes", "Status": recent_xover,
             "Explain": "10MA must have crossed 20MA within the last 10 trading days."},
            {"Rule": "4. The 'Touch' Rule", "Value": f"Low: {curr['Low']:.2f} vs 20MA: {curr['MA20']:.2f}", "Goal": "Low <= MA20", "Status": curr["Low"] <= curr["MA20"],
             "Explain": "Price must dip to or below the 20MA to test support."},
            {"Rule": "5. Candle Color", "Value": f"C: {curr['Close']:.2f} vs O: {curr['Open']:.2f}", "Goal": "Close > Open", "Status": curr["Close"] > curr["Open"],
             "Explain": "The daily candle must be green (bullish)."}
        ]

        # Display results
        for r in results:
            icon = "✅" if r["Status"] else "❌"
            with st.expander(f"{icon} {r['Rule']}"):
                st.write(f"**Current Data:** {r['Value']}")
                st.write(f"**Requirement:** {r['Goal']}")
                st.info(f"**Explanation:** {r['Explain']}")
                
        # Final Summary
        if all(r["Status"] for r in results):
            st.success("BKNG IS A VALID SIGNAL")
        else:
            st.error("BKNG is NOT a signal today because one or more rules failed.")
    else:
        st.error("Could not download data for BKNG.")
