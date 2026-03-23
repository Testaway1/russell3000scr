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
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        
        # True Range - using vectorized operations
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        # Create DataFrame with all three TR components and take max
        tr_df = pd.DataFrame({
            'tr1': tr1,
            'tr2': tr2,
            'tr3': tr3
        })
        tr = tr_df.max(axis=1)
        
        # Average True Range using EMA
        atr = tr.ewm(alpha=1.0/ATR_PERIOD, adjust=False).mean()
        
        # Basic Upper and Lower bands
        hl2 = (high + low) / 2.0
        basic_upper = hl2 + FACTOR * atr
        basic_lower = hl2 - FACTOR * atr
        
        # Initialize arrays for the loop
        n = len(df)
        upper = np.zeros(n)
        lower = np.zeros(n)
        direction = np.ones(n)
        
        # Use scalar values in the loop (avoid Series comparisons)
        for i in range(1, n):
            # Convert to scalar values for comparison
            basic_upper_i = basic_upper.iloc[i]
            basic_lower_i = basic_lower.iloc[i]
            upper_prev = upper[i-1]
            lower_prev = lower[i-1]
            close_prev = close.iloc[i-1]
            close_i = close.iloc[i]
            
            # Upper band logic - using scalar comparisons
            if basic_upper_i < upper_prev or close_prev > upper_prev:
                upper[i] = basic_upper_i
            else:
                upper[i] = upper_prev
            
            # Lower band logic
            if basic_lower_i > lower_prev or close_prev < lower_prev:
                lower[i] = basic_lower_i
            else:
                lower[i] = lower_prev
            
            # Direction logic
            if direction[i-1] == 1:
                direction[i] = 1 if close_i <= upper[i] else -1
            else:
                direction[i] = -1 if close_i >= lower[i] else 1
        
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
    
    # Show available secrets
    try:
        st.write("Secrets available:", list(st.secrets.keys()) if hasattr(st, 'secrets') else "No secrets found")
    except:
        st.write("Unable to access secrets")

# Initialize Google Sheets Connection
try:
    st.write("Attempting to connect to Google Sheets...")
    conn = st.connection("gsheets", type=GSheetsConnection)
    
    # Test the connection
    df_tickers = conn.read(ttl="10m")
    
    if df_tickers is not None and not df_tickers.empty:
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
use_samples = st.sidebar.checkbox("Use sample tickers for testing")
if use_samples:
    sample_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
    tickers = sample_tickers
    st.sidebar.success(f"Using {len(sample_tickers)} sample tickers for testing")

# Add option to limit number of tickers for faster testing
max_tickers = st.sidebar.number_input("Max tickers to scan (0 = all)", min_value=0, max_value=500, value=0)
if max_tickers > 0 and not use_samples:
    tickers = tickers[:max_tickers]
    st.sidebar.info(f"Limited to first {max_tickers} tickers")

if run_button:
    long_hits = []
    short_hits = []
    errors = []
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_text = st.empty()
    error_text = st.empty()
    
    total_tickers = len(tickers)
    
    for idx, ticker in enumerate(tickers):
        status_text.text(f"📊 Scanning {ticker} ({idx+1}/{total_tickers})")
        progress_bar.progress((idx + 1) / total_tickers)
        
        try:
            # Download data with error handling and retry logic
            max_retries = 2
            df = None
            
            for attempt in range(max_retries):
                try:
                    df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)
            
            if df is None or len(df) < 30:
                continue
            
            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols):
                continue
            
            # Calculate indicators
            df["MA10"] = df["Close"].rolling(MA_FAST).mean()
            df["MA20"] = df["Close"].rolling(MA_SLOW).mean()
            
            # Calculate supertrend direction
            direction = calc_supertrend(df)
            if direction is not None:
                df["dir"] = direction
            else:
                continue
            
            # Check if we have enough data
            if len(df) < 2:
                continue
                
            # Get current and previous data
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Skip if any NaN values
            if pd.isna(curr["MA10"]) or pd.isna(curr["MA20"]) or pd.isna(curr["dir"]):
                continue
            
            # Long signal logic
            try:
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
            except Exception as e:
                st.warning(f"Error evaluating long signal for {ticker}: {str(e)}")
            
            # Short signal logic
            try:
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
                st.warning(f"Error evaluating short signal for {ticker}: {str(e)}")
                
        except Exception as e:
            errors.append(f"{ticker}: {str(e)[:100]}")
            continue
        
        # Update results in real-time
        results_text.text(f"✅ Found: {len(long_hits)} long, {len(short_hits)} short signals")
        
        # Show errors if any
        if errors and idx % 10 == 0:  # Show errors periodically
            with error_text.container():
                st.warning(f"Errors encountered: {len(errors)} tickers failed")
    
    status_text.text("✅ Scanning Complete!")
    
    # Show errors summary
    if errors:
        with st.expander(f"⚠️ Errors ({len(errors)} tickers)"):
            st.write("\n".join(errors[:20]))  # Show first 20 errors
    
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
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tickers Scanned", total_tickers)
    with col2:
        st.metric("Long Signals", len(long_hits))
    with col3:
        st.metric("Short Signals", len(short_hits))
    with col4:
        success_rate = ((total_tickers - len(errors)) / total_tickers * 100) if total_tickers > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")

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
