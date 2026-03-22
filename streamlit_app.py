import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- CONFIG ---
MA_CROSS_LOOKBACK = 11  # Updated to 11 days for this test

def calc_supertrend(df):
    """Fixed logic to avoid the 'ValueError' during loop comparisons."""
    high, low, close = df["High"], df["Low"], df["Close"]
    
    # Calculate basic bands
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/10, adjust=False).mean()
    hl2 = (high + low) / 2.0
    upper, lower = hl2 + 3.0 * atr, hl2 - 3.0 * atr
    
    n = len(df)
    f_u, f_l = np.zeros(n), np.zeros(n)
    direction = np.ones(n)
    st_line = np.zeros(n)
    
    for i in range(1, n):
        # We must use .iloc[i] and .iloc[i-1] for EVERY reference to avoid the ValueError
        curr_upper = upper.iloc[i]
        curr_lower = lower.iloc[i]
        prev_close = close.iloc[i-1]
        prev_f_u = f_u[i-1]
        prev_f_l = f_l[i-1]
        
        # Final Upper Band
        if curr_upper < prev_f_u or prev_close > prev_f_u:
            f_u[i] = curr_upper
        else:
            f_u[i] = prev_f_u
            
        # Final Lower Band
        if curr_lower > prev_f_l or prev_close < prev_f_l:
            f_l[i] = curr_lower
        else:
            f_l[i] = prev_f_l
            
        # Direction and Supertrend Line
        if st_line[i-1] == prev_f_u:
            direction[i] = 1 if close.iloc[i] <= f_u[i] else -1
        else:
            direction[i] = -1 if close.iloc[i] >= f_l[i] else 1
            
        st_line[i] = f_l[i] if direction[i] == -1 else f_u[i]
        
    return direction

def get_bkng_analysis():
    df = yf.download("BKNG", period="1y", interval="1d", auto_adjust=True, progress=False)
    if df.empty: return None

    # Flatten headers to prevent KeyError
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    
    # Calculate MAs
    df["ma_fast"] = df["Close"].rolling(10).mean()
    df["ma_slow"] = df["Close"].rolling(20).mean()
    df["direction"] = calc_supertrend(df)
    
    # Signal Logic
    fast, slow = df["ma_fast"].values, df["ma_slow"].values
    cross_found, days_ago = False, 0
    for i in range(len(df)-1, len(df)-1-MA_CROSS_LOOKBACK, -1):
        if i <= 0: break
        if fast[i-1] <= slow[i-1] and fast[i] > slow[i]:
            cross_found, days_ago = True, (len(df)-1-i)
            break

    curr = df.iloc[-1]
    
    # The 5 Data Items: Values & Explanations
    return [
        {
            "Item": "1. Supertrend Direction",
            "Value": f"{int(curr['direction'])}",
            "Req": "-1 (Uptrend)",
            "Status": curr["direction"] == -1,
            "Explanation": "Ensures the overall price volatility trend is positive. -1 means 'Long'."
        },
        {
            "Item": "2. MA Alignment",
            "Value": f"10MA: {curr['ma_fast']:.2f} > 20MA: {curr['ma_slow']:.2f}",
            "Req": "True",
            "Status": curr["ma_fast"] > curr["ma_slow"],
            "Explanation": "Confirms the short-term average price is above the medium-term average."
        },
        {
            "Item": "3. Recent Crossover",
            "Value": f"{days_ago} days ago" if cross_found else "Not in 11d",
            "Req": f"Within {MA_CROSS_LOOKBACK} Days",
            "Status": cross_found,
            "Explanation": "The 10MA must have crossed above the 20MA recently to ensure a fresh move."
        },
        {
            "Item": "4. The 'Touch' Rule",
            "Value": f"Low: {curr['Low']:.2f} | 20MA: {curr['ma_slow']:.2f}",
            "Req": "Low <= 20MA",
            "Status": curr["Low"] <= curr["ma_slow"],
            "Explanation": "Price must dip down to 'touch' the 20-day MA support line."
        },
        {
            "Item": "5. Bullish Candle",
            "Value": f"O: {curr['Open']:.2f} / C: {curr['Close']:.2f}",
            "Req": "Close > Open",
            "Status": curr["Close"] > curr["Open"],
            "Explanation": "Requires a 'Green Day' where the closing price is higher than the opening price."
        }
    ]

# --- STREAMLIT UI ---
st.title("BKNG Logic Inspector (11-Day Test)")

if st.button("Run Deep Dive"):
    data_items = get_bkng_analysis()
    if data_items:
        for item in data_items:
            status_icon = "✅" if item["Status"] else "❌"
            with st.expander(f"{status_icon} {item['Item']}"):
                st.write(f"**Current Data:** {item['Value']}")
                st.write(f"**Required:** {item['Req']}")
                st.info(f"**Why:** {item['Explanation']}")
    else:
        st.error("Could not fetch data.")
