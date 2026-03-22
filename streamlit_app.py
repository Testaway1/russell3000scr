import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- CONFIG ---
MA_CROSS_LOOKBACK = 11  # Increased to 11 for this exercise

def get_bkng_analysis():
    # 1. Download & Clean
    df = yf.download("BKNG", period="1y", interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    
    # 2. Calculations
    df["ma_fast"] = df["Close"].rolling(10).mean()
    df["ma_slow"] = df["Close"].rolling(20).mean()
    # Assume calc_supertrend is defined as per your PC script
    df["st_dir"] = calc_supertrend(df) 
    
    # 3. Lookback Check (Now 11 Days)
    fast, slow = df["ma_fast"].values, df["ma_slow"].values
    crossover_found = False
    days_ago = -1
    
    for i in range(len(df)-1, len(df)-1-MA_CROSS_LOOKBACK, -1):
        if i <= 0: break
        if fast[i-1] <= slow[i-1] and fast[i] > slow[i]:
            crossover_found = True
            days_ago = len(df) - 1 - i
            break

    curr = df.iloc[-1]
    
    # 4. Values for the 5 Criteria
    analysis = [
        {
            "Label": "1. Supertrend",
            "Value": f"{int(curr['st_dir'])}",
            "Req": "-1 (Uptrend)",
            "Status": curr["st_dir"] == -1,
            "Meaning": "Overall market trend is bullish."
        },
        {
            "Label": "2. MA Alignment",
            "Value": f"10MA: ${curr['ma_fast']:,.2f} | 20MA: ${curr['ma_slow']:,.2f}",
            "Req": "10MA > 20MA",
            "Status": curr["ma_fast"] > curr["ma_slow"],
            "Meaning": "Short-term momentum is above long-term average."
        },
        {
            "Label": "3. Crossover",
            "Value": f"Found {days_ago} days ago" if crossover_found else "Not found in 11d",
            "Req": "Within 11 Days",
            "Status": crossover_found,
            "Meaning": "Ensures we aren't chasing an old, exhausted trend."
        },
        {
            "Label": "4. Pullback Touch",
            "Value": f"Low: ${curr['Low']:,.2f} | 20MA: ${curr['ma_slow']:,.2f}",
            "Req": "Low <= 20MA and Close > 20MA",
            "Status": (curr["Low"] <= curr["ma_slow"]) and (curr["Close"] > curr["ma_slow"]),
            "Meaning": "The stock must dip to support before we buy."
        },
        {
            "Label": "5. Candle",
            "Value": f"Open: ${curr['Open']:,.2f} | Close: ${curr['Close']:,.2f}",
            "Req": "Close > Open",
            "Status": curr["Close"] > curr["Open"],
            "Meaning": "The day must end with buyers in control."
        }
    ]
    return analysis

# --- UI DISPLAY ---
if st.button("Run BKNG 11-Day Test"):
    results = get_bkng_analysis()
    
    for r in results:
        color = "green" if r["Status"] else "red"
        st.markdown(f"### {'✅' if r['Status'] else '❌'} {r['Label']}")
        st.write(f"**Current Value:** {r['Value']}")
        st.write(f"**Requirement:** {r['Req']}")
        st.info(f"**Logic:** {r['Meaning']}")
        st.divider()
