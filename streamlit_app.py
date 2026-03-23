import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from streamlit_gsheets import GSheetsConnection
from datetime import datetime
import time

# --- CONFIG & INDICATOR LOGIC ---
ATR_PERIOD = 10
FACTOR = 3.0
MA_FAST = 10
MA_SLOW = 20

def calc_supertrend(df):
    """Calculate Supertrend indicator"""
    try:
        high, low, close = df["High"], df["Low"], df["Close"]
        
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Average True Range
        atr = tr.ewm(alpha=1.0 / ATR_PERIOD, adjust=False).mean()
        
        # Basic Upper and Lower bands
        hl2 = (high + low) / 2.0
        basic_upper = hl2 + FACTOR * atr
        basic_lower = hl2 - FACTOR * atr
        
        # Calculate direction
        n = len(df)
        upper = np.zeros(n)
        lower = np.zeros(n)
        direction = np.ones(n)
        
        for i in range(1, n):
            # Upper band logic
            if basic_upper.iloc[i] < upper[i-1] or close.iloc[i-1] > upper[i-1]:
                upper[i] = basic_upper.iloc[i]
            else:
                upper[i] = upper[i-1]
            
            # Lower band logic
            if basic_lower.iloc[i] > lower[i-1] or close.iloc[i-1] < lower[i-1]:
                lower[i] = basic_lower.iloc[i]
            else:
                lower[i] = lower[i-1]
            
            # Direction logic
            if direction[i-1] == 1:
                direction[i] = 1 if close.iloc[i] <= upper[i] else -1
            else:
                direction[i] = -1 if close.iloc[i] >= lower[i] else 1
        
        return direction
    except Exception as e:
        st.error(f"Error in supertrend calculation: {e}")
        return None

# --- STREAMLIT UI ---
st.set_page_config(page_title="Russell 3000 Screener", layout="wide")
st.title("📈 Russell 3000 Cloud Screener")
st.subheader("Supertrend + MA Confluence (Daily)")

# Debug section
with st.expander("🔧 Connection Debug Info"):
    st.write("Checking connection configuration...")
    
    # Show available secrets (without revealing actual values)
    try:
        st.write("Secrets available:", list(st.secrets.keys()) if hasattr(st, 'secrets') else "No secrets found")
    except:
        st.write("Unable to access secrets")

# Initialize Google Sheets Connection with better error handling
try:
    st.write("Attempting to connect to Google Sheets...")
    conn = st.connection("gsheets", type=GSheetsConnection)
    
    # Test the connection with a simple read
    df_tickers = conn.read(ttl="10m")
    
    if df_tickers is not None and not df_tickers.empty:
        # Check if the dataframe has the expected column
        if "Ticker" in df_tickers.columns:
            tickers = df_tickers["Ticker"].dropna().unique().tolist()
            st.sidebar.success(f"✅ Connected to Cloud DB: {len(tickers)} tickers loaded.")
            st.write(f"Sample tickers: {tickers[:5]}...")
        else:
            st.error("Google Sheets missing 'Ticker' column. Please ensure your sheet has a column named 'Ticker'")
            st.write("Available columns:", list(df_tickers.columns))
            st.stop()
    else:
        st.error("No data found in Google Sheets")
        st.stop()
        
except Exception as e:
    st.error(f"❌ Could not connect to Google Sheets: {str(e)}")
    st.info("""
    To fix this issue:
    1. Create a `.streamlit/secrets.toml` file in your project directory
    2. Add your Google Sheets connection details:
    
    ```toml
    [connections.gsheets]
    spreadsheet = "YOUR_SPREADSHEET_ID"
