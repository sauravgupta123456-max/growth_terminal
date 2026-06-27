import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Buy Level Generator")
st.title("🎯 Structural Buy Level Generator")

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
        
        # 1. CALCULATE BASE INDICATORS
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Calculate True Range / Daily Range
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        current_price = df['Close'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        
        # -----------------------------------------------------------------
        # STRUCTURAL BUY LEVEL ENGINE
        # -----------------------------------------------------------------
        buy_levels = {}

        # A. THE INSTITUTIONAL ANCHOR (Anchored VWAP from 60-day high)
        lookback = 60
        recent_data = df.tail(lookback)
        if not recent_data.empty:
            swing_high_idx = recent_data['High'].idxmax()
            # Calculate VWAP from that exact peak to today
            anchor_df = df.loc[swing_high_idx:]
            tp = (anchor_df['High'] + anchor_df['Low'] + anchor_df['Close']) / 3
            anchored_vwap = (tp * anchor_df['Volume']).cumsum() / anchor_df['Volume'].cumsum()
            current_anchored_vwap = anchored_vwap.iloc[-1]
            if not np.isnan(current_anchored_vwap):
                buy_levels['⚓️ Institutional Anchor (Anchored VWAP)'] = current_anchored_vwap

        # B. MARKET MECHANICS (Gap Fills - The Trampoline)
        df['Prev_High'] = df['High'].shift(1)
        # A gap up is when today's low is higher than yesterday's high
        df['Is_Gap_Up'] = df['Low'] > df['Prev_High'] 
        # The "trampoline" is yesterday's high (the bottom of the empty space)
        df['Gap_Support_Level'] = np.where(df['Is_Gap_Up'], df['Prev_High'], np.nan)
        
        # Find the highest recent gap support level that is below current price
        recent_gaps = df.tail(30)[(df.tail(30)['Is_Gap_Up']) & (df.tail(30)['Gap_Support_Level'] < current_price)]
        if not recent_gaps.empty:
            # We want the highest gap level below price for the best bouncy support
            best_gap_level = recent_gaps['Gap_Support_Level'].max()
            buy_levels['🏀 Gap Fill Trampoline'] = best_gap_level

        # C. THE PANIC FLUSH (Capitulation Spike)
        df['Avg_Range'] = high_low.rolling(20).mean() # Fixed: use high_low instead of df['Range']
        df['Avg_Vol'] = df['Volume'].rolling(20).mean()
        # Range > 2.5x normal AND Volume > 3x normal
        df['Is_Capitulation'] = (high_low > (2.5 * df['Avg_Range'])) & (df['Volume'] > (3 * df['Avg_Vol']))
        
        recent_caps = df.tail(30)[df.tail(30)['Is_Capitulation']]
        if not recent_caps.empty:
            # The buy level is the absolute bottom wick of that panic day
            cap_level = recent_caps['Low'].min()
            buy_levels['🩸 Panic Flush (Capitulation Spike)'] = cap_level

        # D. HEALTHY PULLBACK (20 EMA)
        ema_20_val = df['EMA_20'].iloc[-1]
        if not np.isnan(ema_20_val):
            buy_levels['🟢 Healthy Pullback (20 EMA)'] = ema_20_val

        # Sort levels from highest to lowest (closest to current price first)
        buy_levels = dict(sorted(buy_levels.items(), key=lambda item: item[1], reverse=True))

        # -----------------------------------------------------------------
        # RISK MANAGEMENT MATH
        # -----------------------------------------------------------------
        base_kelly = calculate_kelly()
        kelly_size = base_kelly * portfolio_value
        avg_vol = df['Avg_Vol'].iloc[-1]
        
        if avg_vol < 100000: 
            kelly_size *= 0.5 # Liquidity cut

        # -----------------------------------------------------------------
        # DRAW CHART
        # -----------------------------------------------------------------
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
        
        # Plot moving averages
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='20 EMA', line=dict(color='cyan', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='50 SMA (Trend Filter)', line=dict(color='red', width=2, dash='dot')))
        
        # Plot the Structural Buy Levels
        level_colors = {
            '⚓️ Institutional Anchor (Anchored VWAP)': 'blue',
            '🏀 Gap Fill Trampoline': 'purple',
            '🩸 Panic Flush (Capitulation Spike)': 'red',
            '🟢 Healthy Pullback (20 EMA)': 'lime'
        }
        
        for name, level in buy_levels.items():
            if level < current_price: # Only plot levels below current price
                color = level_colors.get(name, 'white')
                fig.add_hline(y=level, line_dash="dash", line_width=2, line_color=color, annotation_text=name)
                
        # Mark Capitulation Spikes with big red triangles
        cap_dates = df[df['Is_Capitulation'] == True].index
        cap_lows = df[df['Is_Capitulation'] == True]['Low']
        fig.add_trace(go.Scatter(x=cap_dates, y=cap_lows, mode='markers', marker=dict(color='red', size=12, symbol='triangle-down'), name='Capitulation Day'))

        fig.update_layout(xaxis_rangeslider_visible=False, height=700, title=f"{ticker} - Current: ${current_price:.2f}")
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------------------------------------
        # PRINT THE BUY LEVELS TABLE
        # -----------------------------------------------------------------
        st.subheader("📊 Structural Buy Levels & Position Sizing")
        
        warnings = []
        if current_price < df['SMA_50'].iloc[-1]: warnings.append("🚨 TREND BROKEN: Stock is below 50 SMA. High risk of falling knife.")
        if avg_vol < 100000: warnings.append("⚠️ Low Liquidity: Slippage risk high.")
        
        if warnings:
            for w in warnings:
                st.warning(w)

        if not any(l < current_price for l in buy_levels.values()):
            st.info("Stock is in a strong uptrend. No structural pullback levels are currently below the price.")

        for name, level in buy_levels.items():
            if level >= current_price:
                continue # Skip levels above current price
                
            stop_loss = max(level - (2 * atr), 0.01) 
            risk_per_share = level - stop_loss
            
            if risk_per_share <= 0: continue 
            
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
