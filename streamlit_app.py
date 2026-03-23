import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- 1. CONFIGURATION ---
MA_CROSS_LOOKBACK = 11  # Set to 11 days for this exercise

# --- 2. THE LOGIC (FIXED FOR MULTI-INDEX) ---
def calc_supertrend(df):
    """Fixed to handle scalar comparisons correctly."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # 10-period ATR
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/10, adjust=False).mean()
    hl2 = (high + low) / 2.0
    
    # Basic Bands
    upper = hl2 + 3.0 * atr
    lower = hl2 - 3.0 * atr
    
    n = len(df)
    f_u, f_l = np.zeros(n), np.zeros(n)
    direction = np.ones(n)
    st_line = np.zeros(n)
    
    for i in range(1, n):
        # We use .item() or float() to ensure we are comparing single numbers
        curr_u, prev_f_u = float(upper.iloc[i]), float(f_u[i-1])
        curr_l, prev_f_l = float(lower.iloc[i]), float(f_l[i-1])
        prev_close = float(close.iloc[i-1])
        
        # Final Bands
        f_u[i] = curr_u if curr_u < prev_f_u or prev_close > prev_f_u else prev_f_u
        f_l[i] = curr_l if curr_l > prev_f_l or prev_close < prev_f_l else prev_f_l
        
        # Direction
        if st_line[i-1] == f_u[i-1]:
            direction[i] = 1 if float(close.iloc[i]) <= f_u[i] else -1
        else:
            direction[i] = -1 if float(close.iloc[i]) >= f_l[i] else 1
        st_line[i] = f_l[i] if direction[i] == -1 else f_u[i]
        
    return direction

def get_bkng_analysis():
    # A. Download with specific settings
    df = yf.download("BKNG", period="1y", interval="1d", auto_adjust=True)
    
    if df.empty: return None

    # B. THE "NUCLEAR" FIX FOR KEYERROR 'CLOSE'
    # This strips away ticker names and keeps only 'Close', 'Open', etc.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    
    # Ensure columns are properly named
    df.columns = [str(c).strip().capitalize() for c in df.columns]

    # C. Indicators
    df["ma_fast"] = df["Close"].rolling(10).mean()
    df["ma_slow"] = df["Close"].rolling(20).mean()
    df["direction"] = calc_supertrend(df)
    
    # D. Bullish Cross Check (last 11 days)
    fast, slow = df["ma_fast"].values, df["ma_slow"].values
    cross_found, days_ago = False, 0
    for i in range(len(df)-1, len(df)-1-MA_CROSS_LOOKBACK, -1):
        if i <= 0: break
        if fast[i-1] <= slow[i-1] and fast[i] > slow[i]:
            cross_found = True
            days_ago = len(df) - 1 - i
            break

    curr = df.iloc[-1]
    
    # E. Data for the 5 Criteria
    results = [
        {
            "Item": "1. Supertrend Direction",
            "Value": f"Direction: {int(curr['direction'])}",
            "Req": "Must be -1 (Uptrend)",
            "Status": curr["direction"] == -1,
            "Explain": "Checks if the price is above the volatility support line. -1 is a 'Go' for Longs."
        },
        {
            "Item": "2. MA Alignment",
            "Value": f"10MA: ${curr['ma_fast']:,.2f} > 20MA: ${curr['ma_slow']:,.2f}",
            "Req": "10MA > 20MA",
            "Status": curr["ma_fast"] > curr["ma_slow"],
            "Explain": "Confirms that the short-term trend is stronger than the long-term trend."
        },
        {
            "Item": "3. Recent Crossover",
            "Value": f"Crossed {days_ago} days ago" if cross_found else "No cross found",
            "Req": f"Within last {MA_CROSS_LOOKBACK} days",
            "Status": cross_found,
            "Explain": "Ensures the 'Golden Cross' (10 over 20) is fresh so you aren't chasing a tired move."
        },
        {
            "Item": "4. The 'Touch' Rule",
            "Value": f"Low: ${curr['Low']:,.2f} / 20MA: ${curr['ma_slow']:,.2f}",
            "Req": "Low <= 20MA AND Close > 20MA",
            "Status": (curr["Low"] <= curr["ma_slow"]) and (curr["Close"] > curr["ma_slow"]),
            "Explain": "A 'Buy the Dip' check. Price must have dipped to touch the 20-day average and held above it."
        },
        {
            "Item": "5. Bullish Candle",
            "Value": f"Open: ${curr['Open']:,.2f} | Close: ${curr['Close']:,.2f}",
            "Req": "Close > Open",
            "Status": curr["Close"] > curr["Open"],
            "Explain": "Ensures the current day is a 'Green Day', confirming buying pressure exists right now."
        }
    ]
    return results

# --- 3. STREAMLIT UI ---
st.title("BKNG Signal Deep-Dive (11-Day Test)")

if st.button("Run BKNG Logic Check"):
    data_items = get_bkng_analysis()
    
    if data_items:
        for item in data_items:
            icon = "✅" if item["Status"] else "❌"
            color = "green" if item["Status"] else "red"
            
            with st.expander(f"{icon} {item['Item']}"):
                st.markdown(f"**Current Value:** :{color}[{item['Value']}]")
                st.write(f"**Requirement:** {item['Req']}")
                st.info(f"**Why this matters:** {item['Explain']}")
                
        if all(i["Status"] for i in data_items):
            st.success("BKNG IS A VALID SIGNAL TODAY!")
        else:
            st.warning("BKNG did not meet all 5 criteria.")
    else:
        st.error("Could not fetch data for BKNG. Check ticker or connection.")
