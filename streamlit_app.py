import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- 1. CORE LOGIC ---
def get_debug_data(symbol):
    try:
        df = yf.download(symbol, period="1y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        
        # Calculations
        df["ma_fast"] = df["Close"].rolling(10).mean()
        df["ma_slow"] = df["Close"].rolling(20).mean()
        # Using the exact calc_supertrend function from earlier
        df["direction"] = calc_supertrend(df).values 
        bull_x, _ = ma_cross_within(df)
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # Data for the 5 Items
        debug_items = [
            {
                "item": "1. Supertrend Direction",
                "value": f"{int(curr['direction'])}",
                "req": "Must be -1 (Uptrend)",
                "passed": curr["direction"] == -1,
                "explanation": "This is the primary trend filter. If it's +1, the trend is bearish."
            },
            {
                "item": "2. MA Alignment",
                "value": f"10MA: {curr['ma_fast']:.2f} vs 20MA: {curr['ma_slow']:.2f}",
                "req": "10MA > 20MA",
                "passed": curr["ma_fast"] > curr["ma_slow"],
                "explanation": "Confirms short-term momentum is higher than long-term momentum."
            },
            {
                "item": "3. Recent Crossover",
                "value": "Detected" if bull_x else "Not Detected",
                "req": "Cross within last 10 days",
                "passed": bull_x,
                "explanation": "Ensures you aren't entering a trade that has already 'run' too far."
            },
            {
                "item": "4. The 'Touch' Rule",
                "value": f"Low: {curr['Low']:.2f} | 20MA: {curr['ma_slow']:.2f}",
                "req": "Low <= 20MA and Close > 20MA",
                "passed": (curr["Low"] <= curr["ma_slow"]) and (curr["Close"] > curr["ma_slow"]),
                "explanation": "Price must dip to the 20MA (test support) and hold above it."
            },
            {
                "item": "5. Candle Color",
                "value": f"Open: {curr['Open']:.2f} | Close: {curr['Close']:.2f}",
                "req": "Close > Open",
                "passed": curr["Close"] > curr["Open"],
                "explanation": "Ensures the day ended with buying pressure (Green Candle)."
            }
        ]
        return debug_items
    except Exception as e:
        st.error(f"Error fetching BKNG: {e}")
        return None

# --- 2. THE UI SECTION ---
st.header("🔍 BKNG Deep Dive Debugger")
st.write("This tool checks why BKNG is or is not appearing in your results today.")

if st.button("Deep Dive: BKNG Current Values"):
    stats = get_debug_data("BKNG")
    
    if stats:
        for s in stats:
            icon = "✅" if s["passed"] else "❌"
            color = "green" if s["passed"] else "red"
            
            with st.expander(f"{icon} {s['item']}"):
                st.markdown(f"**Status:** :{color}[{'PASSED' if s['passed'] else 'FAILED'}]")
                st.write(f"**Current Value:** {s['value']}")
                st.write(f"**Requirement:** {s['req']}")
                st.divider()
                st.caption(f"**Why this item exists:** {s['explanation']}")
