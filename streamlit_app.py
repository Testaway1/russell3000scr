import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- CONFIG ---
ATR_PERIOD        = 10
FACTOR            = 3.0
MA_FAST           = 10
MA_SLOW           = 20
MA_CROSS_LOOKBACK = 11  # Updated to 11 days

def calc_supertrend(df):
    """Exact logic from the original python file."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/ATR_PERIOD, adjust=False).mean()
    hl2 = (high + low) / 2.0
    basic_upper, basic_lower = hl2 + FACTOR * atr, hl2 - FACTOR * atr
    n = len(df)
    final_upper, final_lower, direction, st_line = np.zeros(n), np.zeros(n), np.ones(n), np.zeros(n)
    for i in range(1, n):
        final_upper[i] = basic_upper.iloc[i] if basic_upper.iloc[i] < final_upper[i-1] or close.iloc[i-1] > final_upper[i-1] else final_upper[i-1]
        final_lower[i] = basic_lower.iloc[i] if basic_lower.iloc[i] > final_lower[i-1] or close.iloc[i-1] < final_lower[i-1] else final_lower[i-1]
        if st_line[i-1] == final_upper[i-1]:
            direction[i] = 1 if close.iloc[i] <= final_upper[i] else -1
        else:
            direction[i] = -1 if close.iloc[i] >= final_lower[i] else 1
        st_line[i] = final_lower[i] if direction[i] == -1 else final_upper[i]
    return direction

def get_bkng_analysis():
    # 1. Download Data
    df = yf.download("BKNG", period="1y", interval="1d", auto_adjust=True, progress=False)
    
    # 2. THE FIX: Forcefully flatten headers to single strings
    # This removes the ('BKNG', 'Close') structure and makes it just 'Close'
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        # Ensure column names are standard strings
        df.columns = [str(c) for c in df.columns]
    
    if df.empty or "Close" not in df.columns:
        return None
    
    # 3. Indicators
    df["ma_fast"] = df["Close"].rolling(MA_FAST).mean()
    df["ma_slow"] = df["Close"].rolling(MA_SLOW).mean()
    df["direction"] = calc_supertrend(df)
    
    # 4. Check for Bullish Crossover in last 11 days
    fast, slow = df["ma_fast"].values, df["ma_slow"].values
    bull_cross = False
    cross_day = "None found in 11d"
    
    for i in range(len(df)-1, len(df)-1-MA_CROSS_LOOKBACK, -1):
        if i <= 0: break
        if fast[i-1] <= slow[i-1] and fast[i] > slow[i]:
            bull_cross = True
            cross_day = df.index[i].strftime('%Y-%m-%d')
            break

    curr = df.iloc[-1]
    
    # 5. The 5 Data Items Breakdown
    checks = [
        {
            "Item": "1. Supertrend Direction",
            "Value": f"{int(curr['direction'])}",
            "Req": "-1 (Uptrend)",
            "Status": curr["direction"] == -1,
            "Explanation": "Determines if the volatility trend is bullish. -1 is required for Longs."
        },
        {
            "Item": "2. MA Alignment",
            "Value": f"10MA: ${curr['ma_fast']:,.2f} > 20MA: ${curr['ma_slow']:,.2f}",
            "Req": "True",
            "Status": curr["ma_fast"] > curr["ma_slow"],
            "Explanation": "Confirms short-term momentum is currently above the 20-day average."
        },
        {
            "Item": "3. Recent Crossover",
            "Value": f"Crossed on {cross_day}",
            "Req": "Within 11 Days",
            "Status": bull_cross,
            "Explanation": "The 10MA must have crossed the 20MA recently (within 11 trading bars)."
        },
        {
            "Item": "4. The 'Touch' Rule",
            "Value": f"Low: ${curr['Low']:,.2f} vs 20MA: ${curr['ma_slow']:,.2f}",
            "Req": "Low <= 20MA",
            "Status": (curr["Low"] <= curr["ma_slow"]) and (curr["Close"] > curr["ma_slow"]),
            "Explanation": "Price must dip to touch the 20-day MA (acting as support) and close above it."
        },
        {
            "Item": "5. Bullish Candle",
            "Value": f"Open: ${curr['Open']:,.2f} / Close: ${curr['Close']:,.2f}",
            "Req": "Close > Open",
            "Status": curr["Close"] > curr["Open"],
            "Explanation": "The current day must be a 'Green' candle to show buyers are stepping in."
        }
    ]
    return checks

# --- UI ---
st.title("BKNG 11-Day Logic Inspector")

if st.button("Deep Dive: BKNG Analysis"):
    results = get_bkng_analysis()
    
    if results:
        for r in results:
            icon = "✅" if r["Status"] else "❌"
            with st.expander(f"{icon} {r['Item']}"):
                st.write(f"**Current Value:** {r['Value']}")
                st.write(f"**Requirement:** {r['Req']}")
                st.info(f"**Logic:** {r['Explanation']}")
    else:
        st.error("Data error: Still unable to find 'Close' column. Please check yfinance version.")
