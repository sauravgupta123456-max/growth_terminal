import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Institutional PM Terminal")
st.title("🏦 Institution-Grade PM Terminal")

st.sidebar.header("Trade Setup")
ticker = st.sidebar.text_input("Ticker (e.g., SIVE.ST, IQE, AAPL)", value="AAPL").upper()
portfolio_value = st.sidebar.number_input("Portfolio Value ($)", value=1000000, step=100000)
run_btn = st.sidebar.button("Run PM Matrix", type="primary")

def get_data(tick, period="1y"):
    df = yf.Ticker(tick).history(period=period)
    return df

def check_trend(df, window=50):
    if len(df) < window: return "N/A"
    current = df['Close'].iloc[-1]
    sma = df['Close'].tail(window).mean()
    return "Bullish" if current > sma else "Bearish"

if run_btn:
    try:
        stock_df = get_data(ticker)
        if stock_df.empty: st.stop()
        
        current_price = stock_df['Close'].iloc[-1]
        
        # -----------------------------------------------------------------
        # LAYER 0, 1, 2: MACRO & CROSS-ASSET PROXIES (Using Free ETFs)
        # -----------------------------------------------------------------
        st.subheader("LAYER 0-2: Global Macro & Liquidity Matrix")
        st.caption("Using ETF proxies to gauge systemic risk. Red = Hostile environment for growth.")
        
        # Fetch Proxies
        tnx_df = get_data("^TNX", "3mo") # 10Y Yield (Discount Rate Proxy)
        dxy_df = get_data("DX-Y.NYB", "3mo") # Dollar (Liquidity Vacuum Proxy)
        hyg_df = get_data("HYG", "3mo") # High Yield Credit (Canary in Coal Mine)
        qqq_df = get_data("QQQ", "3mo") # Nasdaq 100 (Growth Breadth Proxy)
        
        tnx_trend = check_trend(tnx_df, 50)
        dxy_trend = check_trend(dxy_df, 50)
        hyg_trend = check_trend(hyg_df, 50)
        qqq_trend = check_trend(qqq_df, 50)
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Color logic: Red if bad for growth, Green if good
        col1.metric("10Y Yield (TNX)", tnx_trend, "🚨 Rising Rates" if tnx_trend=="Bullish" else "✅ Flat/Falling")
        col2.metric("Dollar (DXY)", dxy_trend, "🚨 Strong Dollar" if dxy_trend=="Bullish" else "✅ Weak Dollar")
        col3.metric("Junk Bonds (HYG)", hyg_trend, "🚨 Credit Stress" if hyg_trend=="Bearish" else "✅ Stable Credit")
        col4.metric("Nasdaq (QQQ)", qqq_trend, "🚨 Growth Weakening" if qqq_trend=="Bearish" else "✅ Growth Stable")
        
        # Macro Override Logic
        macro_blocker = False
        if tnx_trend == "Bullish" or hyg_trend == "Bearish":
            macro_blocker = True
            st.error("🛑 MACRO OVERRIDE: Systemic risk is too high. Do not buy illiquid growth assets.")

        # -----------------------------------------------------------------
        # LAYER 3: FUNDAMENTALS & DERIVATIVES PROXIES
        # -----------------------------------------------------------------
        st.subheader("LAYER 3: Micro Fundamentals & Volatility Surface")
        
        tick = yf.Ticker(ticker)
        
        col_f1, col_f2, col_f3 = st.columns(3)
        
        # 1. Insider Buying
        insider_signal = "Neutral"
        try:
            insiders = tick.insider_transactions
            if not insiders.empty:
                recent_insiders = insiders.iloc[0] # Get most recent
                if recent_insiders['Transaction'] == "Purchase":
                    insider_signal = "✅ Bullish (Recent Purchase)"
                else:
                    insider_signal = "🚨 Bearish (Recent Sale)"
            else:
                insider_signal = "N/A (No EU/UK data usually)"
        except:
            insider_signal = "N/A"
        col_f1.metric("Insider Activity", insider_signal)
        
        # 2. Analyst Revision Momentum (Target vs Current Price)
        analyst_signal = "Neutral"
        try:
            targets = tick.analyst_price_targets
            if targets and 'mean' in targets:
                mean_target = targets['mean']
                upside = ((mean_target - current_price) / current_price) * 100
                if upside > 20: analyst_signal = f"✅ Bullish (+{upside:.1f}% upside)"
                elif upside < 0: analyst_signal = f"🚨 Bearish ({upside:.1f}% downside)"
        except:
            analyst_signal = "N/A"
        col_f2.metric("Forward Estimates", analyst_signal)
        
        # 3. Implied Volatility Proxy (Options Surface)
        iv_signal = "Neutral"
        try:
            # Get options chain expiring closest to 30 days out
            expirations = tick.options
            if expirations:
                opt_chain = tick.option_chain(expirations[0])
                # Get at-the-money call IV
                calls = opt_chain.calls
                atm_idx = (calls['strike'] - current_price).abs().idxmin()
                atm_iv = calls.loc[atm_idx, 'impliedVolatility']
                
                if atm_iv > 0.6: iv_signal = f"🚨 Extreme Fear (IV: {atm_iv*100:.1f}%)"
                elif atm_iv > 0.4: iv_signal = f"✅ Elevated (IV: {atm_iv*100:.1f}% - Good for longs)"
                else: iv_signal = f"⚠️ Complacent (IV: {atm_iv*100:.1f}%)"
        except:
            iv_signal = "N/A (No Options Data)"
        col_f3.metric("Volatility Surface (ATM IV)", iv_signal)

        # -----------------------------------------------------------------
        # LAYER 4: MICROSTRUCTURE & STRUCTURAL LEVELS
        # -----------------------------------------------------------------
        tab1, tab2 = st.tabs(["Chart & Structural Levels", "PM Trade Ticket"])
        
        with tab1:
            # Calculate Engine (Anchored VWAP, Gaps, Caps, EMAs)
            stock_df['EMA_20'] = stock_df['Close'].ewm(span=20, adjust=False).mean()
            stock_df['SMA_50'] = stock_df['Close'].rolling(window=50).mean()
            
            high_low = stock_df['High'] - stock_df['Low']
            true_range = pd.concat([high_low, np.abs(stock_df['High'] - stock_df['Close'].shift()), np.abs(stock_df['Low'] - stock_df['Close'].shift())], axis=1).max(axis=1)
            stock_df['ATR'] = true_range.rolling(14).mean()
            atr = stock_df['ATR'].iloc[-1]
            
            buy_levels = {}
            
            # A. Anchored VWAP
            recent_data = stock_df.tail(60)
            swing_high_idx = recent_data['High'].idxmax()
            anchor_df = stock_df.loc[swing_high_idx:]
            tp = (anchor_df['High'] + anchor_df['Low'] + anchor_df['Close']) / 3
            anchored_vwap = (tp * anchor_df['Volume']).cumsum() / anchor_df['Volume'].cumsum()
            buy_levels['⚓️ Anchored VWAP'] = anchored_vwap.iloc[-1]
            
            # B. Gap Fills
            stock_df['Prev_High'] = stock_df['High'].shift(1)
            stock_df['Is_Gap_Up'] = stock_df['Low'] > stock_df['Prev_High']
            gaps = stock_df.tail(30)[(stock_df.tail(30)['Is_Gap_Up']) & (stock_df.tail(30)['Prev_High'] < current_price)]
            if not gaps.empty: buy_levels['🏀 Gap Fill'] = gaps['Prev_High'].max()
            
            # C. Capitulation
            stock_df['Avg_Range'] = high_low.rolling(20).mean()
            stock_df['Avg_Vol'] = stock_df['Volume'].rolling(20).mean()
            stock_df['Is_Cap'] = (high_low > (2.5 * stock_df['Avg_Range'])) & (stock_df['Volume'] > (3 * stock_df['Avg_Vol']))
            caps = stock_df.tail(30)[stock_df.tail(30)['Is_Cap']]
            if not caps.empty: buy_levels['🩸 Capitulation Wick'] = caps['Low'].min()
            
            # D. 20 EMA
            buy_levels['🟢 20 EMA'] = stock_df['EMA_20'].iloc[-1]
            
            buy_levels = dict(sorted(buy_levels.items(), key=lambda item: item[1], reverse=True))

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=stock_df.index, open=stock_df['Open'], high=stock_df['High'], low=stock_df['Low'], close=stock_df['Close'], name="Price"))
            fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['EMA_20'], mode='lines', name='20 EMA', line=dict(color='cyan', width=2)))
            fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['SMA_50'], mode='lines', name='50 SMA', line=dict(color='red', width=2, dash='dot')))
            
            for name, level in buy_levels.items():
                if level < current_price:
                    fig.add_hline(y=level, line_dash="dash", line_width=2, line_color='lime', annotation_text=f"{name}: ${level:.2f}")
            
            fig.update_layout(xaxis_rangeslider_visible=False, height=600)
            st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------------------------------------
        # THE FINAL SYNTHESIS: PM TRADE TICKET
        # -----------------------------------------------------------------
        with tab2:
            st.subheader("Portfolio Manager Risk Sheet")
            
            # Scoring Matrix (Start at 100, deduct for negatives)
            score = 100
            reasons = []
            
            if macro_blocker: score -= 100; reasons.append("MACRO OVERRIDE")
            if insider_signal.startswith("🚨"): score -= 20; reasons.append("Insider Selling")
            if analyst_signal.startswith("🚨"): score -= 20; reasons.append("Estimates Slashing")
            if iv_signal.startswith("⚠️"): score -= 10; reasons.append("Low Vol (No edge)")
            
            # Determine Action
            if score <= 0:
                action = "DO NOT TRADE"
                color = "red"
            elif score < 80:
                action = "REDUCE SIZE (50% Cut)"
                color = "orange"
            else:
                action = "APPROVED FOR FULL SIZING"
                color = "green"
                
            st.markdown(f"### <span style='color:{color}; font-size: 24px;'>{action} (Score: {score}/100)</span>", unsafe_allow_html=True)
            if reasons: st.write("**Deductions:**", ", ".join(reasons))
            
            if score > 0:
                st.divider()
                base_kelly = 0.125 # 12.5% Half-Kelly
                kelly_size = base_kelly * portfolio_value
                if score < 80: kelly_size *= 0.5 # Cut size if not perfect score
                
                for name, level in buy_levels.items():
                    if level >= current_price: continue
                    stop_loss = max(level - (2 * atr), 0.01)
                    risk_per_share = level - stop_loss
                    if risk_per_share <= 0: continue
                    
                    shares = int(kelly_size / risk_per_share)
                    cost = shares * level
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.markdown(f"**{name}**")
                    c2.metric("Exec Price", f"${level:.2f}")
                    c3.metric("Risk Stop", f"${stop_loss:.2f}")
                    c4.metric("Units to Buy", f"{shares:,}")

    except Exception as e:
        st.error(f"Data ingestion error: {e}")
