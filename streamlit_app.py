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
    st.info("To fix this issue:\n\n"
            "1. Create a `.streamlit/secrets.toml` file in your project directory\n"
            "2. Add your Google Sheets connection details:\n\n"
            "```toml\n"
            "[connections.gsheets]\n"
            "spreadsheet = 'YOUR_SPREADSHEET_ID'\n"
            "```\n\n"
            "3. Make sure your Google Sheet has a column named 'Ticker' with the list of Russell 3000 tickers\n"
            "4. If using Streamlit Cloud, add these secrets in the dashboard")
    st.stop()

# User Controls
st.sidebar.header("Screener Controls")
run_button = st.sidebar.button("🚀 Run Screener")

# Add sample tickers option for testing
if st.sidebar.checkbox("Use sample tickers for testing"):
    sample_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    tickers = sample_tickers
    st.sidebar.info(f"Using sample tickers: {sample_tickers}")

if run_button:
    long_hits = []
    short_hits = []
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_text = st.empty()
    
    total_tickers = len(tickers)
    
    for idx, ticker in enumerate(tickers):
        status_text.text(f"📊 Scanning {ticker} ({idx+1}/{total_tickers})")
        progress_bar.progress((idx + 1) / total_tickers)
        
        try:
            # Download data with error handling
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            
            if len(df) < 30:
                continue
            
            # Calculate indicators
            df["MA10"] = df["Close"].rolling(MA_FAST).mean()
            df["MA20"] = df["Close"].rolling(MA_SLOW).mean()
            
            # Calculate supertrend direction
            direction = calc_supertrend(df)
            if direction is not None:
                df["dir"] = direction
            
            # Get current and previous data
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Long signal logic
            is_long = (
                curr["dir"] == -1 and 
                curr["MA10"] > curr["MA20"] and 
                curr["Low"] <= curr["MA20"] and 
                curr["Close"] > curr["MA20"] and
                curr["Close"] > curr["Open"]
            )
            
            if is_long:
                long_hits.append({
                    "Ticker": ticker, 
                    "Price": f"${curr['Close']:.2f}",
                    "MA10": f"${curr['MA10']:.2f}",
                    "MA20": f"${curr['MA20']:.2f}",
                    "Change %": f"{((curr['Close'] - prev['Close'])/prev['Close']*100):.2f}%"
                })
            
            # Short signal logic
            is_short = (
                curr["dir"] == 1 and 
                curr["MA10"] < curr["MA20"] and 
                curr["High"] >= curr["MA20"] and 
                curr["Close"] < curr["MA20"] and
                curr["Close"] < curr["Open"]
            )
            
            if is_short:
                short_hits.append({
                    "Ticker": ticker, 
                    "Price": f"${curr['Close']:.2f}",
                    "MA10": f"${curr['MA10']:.2f}",
                    "MA20": f"${curr['MA20']:.2f}",
                    "Change %": f"{((curr['Close'] - prev['Close'])/prev['Close']*100):.2f}%"
                })
                
        except Exception as e:
            st.warning(f"Error processing {ticker}: {str(e)}")
            continue
        
        # Update results in real-time
        results_text.text(f"Found: {len(long_hits)} long signals, {len(short_hits)} short signals")
    
    status_text.text("✅ Scanning Complete!")
    
    # Display Results
    st.header("📊 Screening Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🟢 Long Signals")
        if long_hits:
            df_long = pd.DataFrame(long_hits)
            st.dataframe(df_long, use_container_width=True)
            st.success(f"Found {len(long_hits)} long signals")
        else:
            st.info("No long signals found.")
            
    with col2:
        st.subheader("🔴 Short Signals")
        if short_hits:
            df_short = pd.DataFrame(short_hits)
            st.dataframe(df_short, use_container_width=True)
            st.warning(f"Found {len(short_hits)} short signals")
        else:
            st.info("No short signals found.")
    
    # Summary statistics
    st.subheader("📈 Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tickers Scanned", total_tickers)
    with col2:
        st.metric("Long Signals", len(long_hits))
    with col3:
        st.metric("Short Signals", len(short_hits))

else:
    st.info("👈 Click 'Run Screener' in the sidebar to start processing the Russell 3000 list.")
    
    # Show sample of loaded tickers
    if 'tickers' in locals() and tickers:
        st.write(f"Loaded {len(tickers)} tickers from Google Sheets")
        with st.expander("View first 20 tickers"):
            st.write(tickers[:20])

# Add footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
