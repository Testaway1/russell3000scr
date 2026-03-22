import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- CONFIG ---
ATR_PERIOD = 10
FACTOR = 3.0
MA_FAST = 10
MA_SLOW = 20
MA_CROSS_LOOKBACK = 11  # Updated to 11 per your request

# --- CORE FUNCTIONS ---
def calc_supertrend(df):
    """Exact Supertrend logic from your original script."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / ATR_PERIOD, adjust=False).mean()
    hl2 = (high + low) / 2.0
    basic_upper, basic_lower = hl2 + FACTOR * atr, hl2 - FACTOR * atr
    n = len(df)
    f_upper, f_lower, direction, st_line = np.zeros(n), np.zeros(n), np.ones(n), np.zeros(n)
    for i in range(1, n):
        f_upper[i] = basic_upper.iloc[i] if basic_upper.iloc[i] < f_upper[i-1] or close.iloc[i-1] > f_upper[i-1] else f_upper[i-1]
        f_lower[i] = basic_lower.iloc[i] if basic_lower.iloc[i] > f_lower[i-1] or close.iloc[i-1] < f_lower[i-1] else f_lower[i-1]
        if st_line[i-1] == f_upper[i-1]: direction[i] = 1 if close.iloc[i] <= f_upper[i] else -1
        else: direction[i] = -1 if close.iloc[i] >= f_lower[i] else 1
        st_line[i] = f_lower[i] if direction[i] == -1 else f_upper[i]
    return direction

def ma_cross_within(df, bars):
    """Checks for a bullish crossover within the specified lookback."""
    fast, slow = df["ma_fast"].values, df["ma_slow"].values
    last_idx = len(df) - 1
    for i in range(last_idx, last_idx - bars, -1):
        if i <= 0: break
        if fast[i-1] <= slow[i-1] and fast[i] > slow[i]:
            return True, (last_idx - i) # Returns True and how many days ago
    return False, None

def get_bkng_analysis():
    # 1. Download with MultiIndex handling
    df = yf.download("BKNG", period="1y", interval="1d", auto_adjust=True, progress=False)
    if df.empty: return None
    
    # FIX: Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    
    # 2. Indicators
    df["ma_fast"] = df["Close"].rolling(MA_FAST).mean()
    df["ma_slow"] = df["Close"].rolling(MA_SLOW).mean()
    df["st_dir"] = calc_supertrend(df)
    
    # 3. Signals
    curr = df.iloc[-1]
    bull_cross, days_ago = ma_cross_within(df, MA_CROSS_LOOKBACK)
    
    # 4. Results for the 5 Criteria
    checks = [
        {"Rule": "1. Supertrend Direction", "Val": f"{int(curr['st_dir'])}", "Req": "-1 (Uptrend)", "Status": curr['st_dir'] == -1},
        {"Rule": "2. MA Alignment", "Val": f"10MA: {curr['ma_fast']:.2f} > 20MA: {curr['ma_slow']:.2f}", "Req": "10MA > 20MA", "Status": curr['ma_fast'] > curr['ma_slow']},
        {"Rule": "3. Recent Crossover", "Val": f"Found {days_ago} days ago" if bull_cross else "No", "Req": f"Within {MA_CROSS_LOOKBACK} Days", "Status": bull_cross},
        {"Rule": "4. Pullback Touch", "Val": f"Low: {curr['Low']:.2f} vs 20MA: {curr['ma_slow']:.2f}", "Req": "Low <= MA20", "Status": curr['Low'] <= curr['ma_slow']},
        {"Rule": "5. Candle", "Val": f"O: {curr['Open']:.2f} | C: {curr['Close']:.2f}", "Req": "Close > Open", "Status": curr['Close'] > curr['Open']}
    ]
    return checks

# --- UI ---
st.title("BKNG Signal Check (11-Day Lookback)")

if st.button("Analyze BKNG"):
    results = get_bkng_analysis()
    if results:
        for r in results:
            icon = "✅" if r["Status"] else "❌"
            with st.expander(f"{icon} {r['Rule']}"):
                st.write(f"**Current:** {r['Val']}")
                st.write(f"**Requirement:** {r['Req']}")
    else:
        st.error("Could not fetch data.")
