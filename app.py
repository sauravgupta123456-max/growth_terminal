import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Buy Level Generator")
st.title("🎯 Growth Stock Buy Level Generator")

# --- SIDEBAR ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker (e.g., SIVE.ST, IQE, NBIS)", value="AAPL").upper()
portfolio_value = st.sidebar.number_input("Total Portfolio Value ($)", value=100000)
run_btn = st.sidebar.button("Generate Buy Levels", type="primary")

# --- HELPERS ---
def calculate_kelly(win_rate=0.45, reward_risk=2.5):
    kelly = win_rate - ((1 - win_rate) / reward_risk)
    return max(kelly / 2, 0) # Half-Kelly

@st.cache_data(ttl=3600)
def get_data(tick):
    return yf.Ticker(tick).history(period="1y")

# --- MAIN LOGIC ---
if run_btn:
    try:
        df = get_data(ticker)
        if df.empty:
            st.error("Ticker not found.")
            st.stop()
        
        # 1. CALCULATE INDICATORS
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        current_price = df['Close'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        
        # 2. FIND BUY LEVELS (Always calculate them)
        high_1y = df['High'].max()
        low_1y = df['Low'].min()
        
        # Use 6-month data if stock is down massively to avoid zombie levels
        if high_1y > (current_price * 1.5):
            recent_df = df.tail(126)
            high_base = recent_df['High'].max()
            low_base = recent_df['Low'].min()
        else:
            high_base = high_1y
            low_base = low_1y
            
        diff = high_base - low_base
        fib_382 = high_base - (0.382 * diff)
        fib_50 = high_base - (0.5 * diff)
        
        # Capitulation (Last 30 days only)
        df['Avg_Range'] = (df['High'] - df['Low']).rolling(20).mean()
        df['Avg_Vol'] = df['Volume'].rolling(20).mean()
        df['Capitulation'] = ((df['High'] - df['Low']) > (2.5 * df['Avg_Range'])) & (df['Volume'] > (3 * df['Avg_Vol']))
        recent_caps = df.tail(30)[df.tail(30)['Capitulation'] == True]
        
        # Gaps (Last 30 days only)
        recent_30 = df.tail(30)
        gaps = recent_30[recent_30['Open'] > recent_30['Close'].shift(1) * 1.02]
        
        # VWAP
        df['VWAP_20'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).rolling(20).sum() / df['Volume'].rolling(20).sum()
        
        # Compile Levels Dictionary
        buy_levels = {}
        if not recent_caps.empty: buy_levels['Capitulation Spike'] = recent_caps['Low'].min()
        if not gaps.empty: buy_levels['Gap Fill Support'] = gaps['Open'].min()
        if not np.isnan(df['VWAP_20'].iloc[-1]): buy_levels['20D VWAP'] = df['VWAP_20'].iloc[-1]
        buy_levels['38.2% Fibonacci'] = fib_382
        
        # Sort levels from highest to lowest (closest to current price first)
        buy_levels = dict(sorted(buy_levels.items(), key=lambda item: item[1], reverse=True))

        # 3. RISK MANAGEMENT MATH
        base_kelly = calculate_kelly()
        kelly_size = base_kelly * portfolio_value
        
        avg_vol = df['Avg_Vol'].iloc[-1]
        if avg_vol < 100000: kelly_size *= 0.5 # Liquidity cut

        # 4. DRAW CHART
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='50 SMA', line=dict(color='red', width=2)))
        
        # Plot Buy Levels on Chart
        colors = ['lime', 'cyan', 'purple', 'blue']
        for i, (name, level) in enumerate(buy_levels.items()):
            if level < current_price: # Only plot levels below current price
                fig.add_hline(y=level, line_dash="dash", line_color=colors[i%4], annotation_text=f"{name}: ${level:.2f}")
                
        fig.update_layout(xaxis_rangeslider_visible=False, height=600, title=f"{ticker} - Current: ${current_price:.2f}")
        st.plotly_chart(fig, use_container_width=True)

        # 5. PRINT THE BUY LEVELS CLEARLY
        st.subheader("📊 Actionable Buy Levels & Position Sizing")
        
        warnings = []
        if current_price < fib_50: warnings.append("⚠️ Deep Retracement: Stock is down >50% from recent high.")
        if avg_vol < 100000: warnings.append("⚠️ Low Liquidity: Slippage risk high.")
        
        if warnings:
            for w in warnings:
                st.warning(w)
            st.markdown("*Levels shown below for monitoring, but exercise extreme caution.*")

        # Display Table
        for name, level in buy_levels.items():
            # If the level is above current price, skip it (we don't want to buy higher than current price)
            if level >= current_price:
                continue
                
            stop_loss = max(level - (2 * atr), 0.01) # Never let stop be negative
            risk_per_share = level - stop_loss
            
            if risk_per_share <= 0: continue # Skip broken math
            
            shares_to_buy = int(kelly_size / risk_per_share)
            cost = shares_to_buy * level
            risk_dollars = shares_to_buy * risk_per_share
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.markdown(f"**{name}**")
            col2.metric("Entry", f"${level:.2f}")
            col3.metric("Stop Loss", f"${stop_loss:.2f}")
            col4.metric("Shares", shares_to_buy)
            col5.metric("Total Risk", f"${risk_dollars:,.2f}")
            st.divider()

    except Exception as e:
        st.error(f"Error: {e}")
