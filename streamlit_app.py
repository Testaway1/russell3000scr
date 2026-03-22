import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- CONFIG ---
MA_CROSS_LOOKBACK = 11  # Updated to 11 days

def calc_supertrend(df):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/10, adjust=False).mean()
    hl2 = (high + low) / 2.0
    upper, lower = hl2 + 3.0 * atr, hl2 - 3.0 * atr
    n = len(df)
    f_u, f_l, d, st_l = np.zeros(n), np.zeros(n), np.ones(n), np.zeros(n)
    for i in range(1, n):
        f_u[i] = upper.iloc[i] if upper.iloc[i] < f_u[i-1] or close.iloc[i-1] > f_u[i-1] else f_u[i-1]
        f_l[i] = lower.iloc[i] if lower.iloc[i] > f_l[i-1] or close.iloc[i-1] < f_l[i-1] else f_l[i-1]
        if st_l[i-1] == f_u[i-1]: d[i] = 1 if close.iloc[i] <= f_u[i] else -1
        else: d[i] = -1 if close.iloc[i] >= f_l[i] else 1
        st_l[i] = f_l[i] if d[i] == -1 else f_u[i]
    return d

def get_bkng_analysis():
    # 1. Download
    df = yf.download("BKNG", period="1y", interval="1d", auto_adjust=True, progress=False)
    if df.empty: return None

    # 2. UNIVERSAL FIX FOR KEYERROR 'CLOSE'
    # This ignores MultiIndex/SingleIndex and just grabs the data columns by position
    new_cols = {}
    for col in df.columns:
        if 'Close' in col: new_cols[col] = 'Close'
        elif 'Open' in col: new_cols[col] = 'Open'
        elif 'High' in col: new_cols[col] = 'High'
        elif 'Low' in col: new_cols[col] = 'Low'
    df = df.rename(columns=new_cols)
    
    # Keep only the columns we need to be safe
    df = df[['Open', 'High', 'Low', 'Close']]

    # 3. Indicators
    df["ma_fast"] = df["Close"].rolling(10).mean()
    df["ma_slow"] = df["Close"].rolling(20).mean()
    df["direction"] = calc_supertrend(df)
    
    # 4. Signal Check
    fast, slow = df["ma_fast"].values, df["ma_slow"].values
    cross_found, days_ago = False, 0
    for i in range(len(df)-1, len(df)-1-MA_CROSS_LOOKBACK, -1):
        if i <= 0: break
        if fast[i-1] <= slow[i-1] and fast[i] > slow[i]:
            cross_found, days_ago = True, (len(df)-1-i)
            break

    curr = df.iloc[-1]
    
    # 5. Data Breakdown
    return [
        {"Item": "1. Supertrend", "Value": f"{int(curr['direction'])}", "Req": "-1", "Status": curr["direction"] == -1, "Info": "Uptrend bias."},
        {"Item": "2. MA Alignment", "Value": f"10MA > 20MA", "Req": "True", "Status": curr["ma_fast"] > curr["ma_slow"], "Info": "Momentum check."},
        {"Item": "3. Crossover", "Value": f"{days_ago} days ago" if cross_found else "No", "Req": "Within 11d", "Status": cross_found, "Info": "Ensures the trend is fresh."},
        {"Item": "4. MA20 Touch", "Value": f"Low: {curr['Low']:.2f} / MA20: {curr['ma_slow']:.2f}", "Req": "Low <= MA20", "Status": curr["Low"] <= curr["ma_slow"], "Info": "Check for pullback to support."},
        {"Item": "5. Candle", "Value": f"Close > Open", "Req": "True", "Status": curr["Close"] > curr["Open"], "Info": "Green day confirmation."}
    ]

# --- UI ---
st.title("BKNG Signal Logic Breakdown")
if st.button("Analyze BKNG"):
    results = get_bkng_analysis()
    if results:
        for r in results:
            icon = "✅" if r["Status"] else "❌"
            with st.expander(f"{icon} {r['Item']}"):
                st.write(f"**Value:** {r['Value']} | **Required:** {r['Req']}")
                st.caption(r["Info"])
    else:
        st.error("Failed to fetch data.")
