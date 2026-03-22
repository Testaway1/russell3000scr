def debug_bkng_values():
    df = yf.download("BKNG", period="1y", interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    
    # Standard calculations from your script
    df["ma_fast"] = df["Close"].rolling(10).mean()
    df["ma_slow"] = df["Close"].rolling(20).mean()
    df["direction"] = calc_supertrend(df).values
    bull_x, _ = ma_cross_within(df)
    
    row = df.iloc[-1]
    
    # Data for the 5 items
    items = {
        "1. Supertrend Direction": {
            "Value": "Uptrend (-1)" if row["direction"] == -1 else "Downtrend (+1)",
            "Requirement": "Must be -1",
            "Result": row["direction"] == -1,
            "Explanation": "Determines the overall trend bias using ATR and price volatility."
        },
        "2. MA Alignment": {
            "Value": f"10MA: {row['ma_fast']:.2f} | 20MA: {row['ma_slow']:.2f}",
            "Requirement": "10MA > 20MA",
            "Result": row["ma_fast"] > row["ma_slow"],
            "Explanation": "Ensures short-term momentum is stronger than long-term momentum."
        },
        "3. Recent Crossover": {
            "Value": "Yes" if bull_x else "No",
            "Requirement": "True (within last 10 days)",
            "Result": bull_x,
            "Explanation": "Ensures the 10MA recently crossed above the 20MA to catch the start of a move."
        },
        "4. The MA20 Touch": {
            "Value": f"Low: {row['Low']:.2f} | 20MA: {row['ma_slow']:.2f}",
            "Requirement": "Low <= 20MA AND Close > 20MA",
            "Result": (row["Low"] <= row["ma_slow"]) and (row["Close"] > row["ma_slow"]),
            "Explanation": "This is a 'mean reversion' check; the price must dip to the average and bounce."
        },
        "5. Candle Color": {
            "Value": f"Open: {row['Open']:.2f} | Close: {row['Close']:.2f}",
            "Requirement": "Close > Open",
            "Result": row["Close"] > row["Open"],
            "Explanation": "Ensures buying pressure is present on the final day (Green candle)."
        }
    }
    return items

# --- STREAMLIT UI DISPLAY ---
if st.button("Deep Dive: BKNG Current Values"):
    results = debug_bkng_values()
    
    for label, data in results.items():
        status = "✅ PASS" if data["Result"] else "❌ FAIL"
        with st.expander(f"{status} - {label}"):
            st.write(f"**Current Value:** {data['Value']}")
            st.write(f"**Requirement:** {data['Requirement']}")
            st.info(f"**Why this matters:** {data['Explanation']}")
