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
        high = df["High"].values
        low = df["Low"].values
        close = df["Close"].values
        
        n = len(df)
        
        # Calculate True Range
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Calculate ATR using EMA
        atr = np.zeros(n)
        atr[0] = tr[0]
        alpha = 1.0 / ATR_PERIOD
        for i in range(1, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
        
        # Calculate basic upper and lower bands
        hl2 = (high + low) / 2.0
        basic_upper = hl2 + FACTOR * atr
        basic_lower = hl2 - FACTOR * atr
        
        # Initialize arrays
        upper = np.zeros(n)
        lower = np.zeros(n)
        direction = np.ones(n)
        
        # Calculate final upper and lower bands and direction
        for i in range(1, n):
            # Upper band logic
            if basic_upper[i] < upper[i-1] or close[i-1] > upper[i-1]:
                upper[i] = basic_upper[i]
            else:
                upper[i] = upper[i-1]
            
            # Lower band logic
            if basic_lower[i] > lower[i-1] or close[i-1] < lower[i-1]:
                lower[i] = basic_lower[i]
            else:
                lower[i] = lower[i-1]
            
            # Direction logic
            if direction[i-1] == 1:
                direction[i] = 1 if close[i] <= upper[i] else -1
            else:
                direction[i] = -1 if close[i] >= lower[i] else -1
        
        return direction
    except Exception as e:
        st.error(f"Error in supertrend calculation: {e}")
        return None

# Alternative simpler supertrend implementation (if the above still has issues)
def calc_supertrend_simple(df):
    """Simplified Supertrend calculation"""
    try:
        # Calculate ATR
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR with EMA
        atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
        
        # Basic bands
        hl2 = (high + low) / 2
        basic_upper = hl2 + (FACTOR * atr)
        basic_lower = hl2 - (FACTOR * atr)
        
        # Initialize final bands
        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        
        # Calculate final bands and direction
        direction = pd.Series(1, index=df.index)
        
        for i in range(1, len(df)):
            # Upper band
            if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
                final_upper.iloc[i] = basic_upper.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i-1]
            
            # Lower band
            if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i-1]
            
            # Direction
            if direction.iloc[i-1] == 1:
                if close.iloc[i] > final_upper.iloc[i]:
                    direction.iloc[i] = -1
                else:
                    direction.iloc[i] = 1
            else:
                if close.iloc[i] < final_lower.iloc[i]:
                    direction.iloc[i] = 1
                else:
                    direction.iloc[i] = -1
        
        return direction.values
    except Exception as e:
        st.error(f"Error in simplified supertrend calculation: {e}")
        return None

# --- STREAMLIT UI ---
st.set_page_config(page_title="Russell 3000 Screener", layout="wide")
st.title("📈 Russell 3000 Cloud Screener")
st.subheader("Supertrend + MA Confluence (Daily)")

# Choose which supertrend implementation to use
use_simple_supertrend = st.sidebar.checkbox("Use simplified supertrend (more stable)", value=True)

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
            # Download data
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            
            if df is None or len(df) < 30:
                continue
            
            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols):
                continue
            
            # Calculate moving averages
            df["MA10"] = df["Close"].rolling(MA_FAST).mean()
            df["MA20"] = df["Close"].rolling(MA_SLOW).mean()
            
            # Calculate supertrend direction using selected method
            if use_simple_supertrend:
                direction = calc_supertrend_simple(df)
            else:
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
                        "Change %": f"{((curr['Close'] - prev['Close'])/prev['Close']*100):.2f}%",
                        "Direction": "Bullish"
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
                        "Change %": f"{((curr['Close'] - prev['Close'])/prev['Close']*100):.2f}%",
                        "Direction": "Bearish"
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
            
            # Download button for long signals
            csv_long = df_long.to_csv(index=False)
            st.download_button(
                label="📥 Download Long Signals CSV",
                data=csv_long,
                file_name=f"long_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No long signals found.")
            
    with col2:
        st.subheader("🔴 Short Signals")
        if short_hits:
            df_short = pd.DataFrame(short_hits)
            st.dataframe(df_short, use_container_width=True)
            st.warning(f"Found {len(short_hits)} short signals")
            
            # Download button for short signals
            csv_short = df_short.to_csv(index=False)
            st.download_button(
                label="📥 Download Short Signals CSV",
                data=csv_short,
                file_name=f"short_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
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
