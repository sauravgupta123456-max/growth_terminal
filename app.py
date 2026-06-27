import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

# Download VADER lexicon silently
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

st.set_page_config(layout="wide", page_title="Growth Terminal")
st.title("📊 Institutional Growth Stock Terminal")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Control Panel")
ticker = st.sidebar.text_input("Ticker (e.g., SIVE.ST, IQE)", value="AAPL").upper()
sector_etf = st.sidebar.text_input("Sector ETF (e.g., SMH, XBI)", value="SPY").upper()
portfolio_value = st.sidebar.number_input("Total Portfolio Value ($)", value=100000)
run_btn = st.sidebar.button("Run Analysis", type="primary")

def calculate_kelly(win_rate=0.45, reward_risk=2.5):
    kelly = win_rate - ((1 - win_rate) / reward_risk)
    return max(kelly / 2, 0) # Half-Kelly, floor at 0

@st.cache_data
def get_data(tick):
    return yf.Ticker(tick).history(period="1y")

if run_btn:
    try:
        df = get_data(ticker)
        if df.empty:
            st.error("Ticker not found or no data.")
            st.stop()
        
        # ---------------------------------------------------------
        # MODULE 1: GROWTH TECHNICAL ENGINE
        # ---------------------------------------------------------
        # Native Pandas calculations (No pandas_ta needed!)
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Manual ATR calculation
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Fibonacci
        high_1y = df['High'].max()
        low_1y = df['Low'].min()
        diff = high_1y - low_1y
        fib_382 = high_1y - (0.382 * diff)
        fib_50 = high_1y - (0.5 * diff)
        
        # Capitulation Spike
        df['Avg_Range'] = (df['High'] - df['Low']).rolling(20).mean()
        df['Avg_Vol'] = df['Volume'].rolling(20).mean()
        df['Capitulation'] = ((df['High'] - df['Low']) > (2.5 * df['Avg_Range'])) & (df['Volume'] > (3 * df['Avg_Vol']))
        
        # VWAP
        df['VWAP_20'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).rolling(20).sum() / df['Volume'].rolling(20).sum()
        
        # Gap Fill (Simplified: find if close > high of prev day by > 1%)
        df['Gap_Up'] = df['Close'].shift(1) * 1.01 < df['Open']
        gap_support = df.loc[df['Gap_Up'] == True, 'Open'].min()
        
        current_price = df['Close'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        avg_vol = df['Avg_Vol'].iloc[-1]
        
        # Determine Buy Levels
        buy_levels = {}
        # Check for capitulation near fibs
        recent_cap = df[df['Capitulation'] == True].tail(3)
        if not recent_cap.empty:
            buy_levels['Capitulation Spike'] = recent_cap['Low'].min()
        
        if not np.isnan(gap_support) and gap_support < current_price:
            buy_levels['Gap Fill Support'] = gap_support
            
        vwap = df['VWAP_20'].iloc[-1]
        if vwap < current_price:
            buy_levels['20D VWAP'] = vwap

        # ---------------------------------------------------------
        # MODULE 2: MACRO MATRIX
        # ---------------------------------------------------------
        vix = yf.Ticker("^VIX").history(period="1mo")['Close'].iloc[-1]
        tnx = yf.Ticker("^TNX").history(period="3mo")
        tnx_sma_50 = tnx['Close'].tail(50).mean()
        tnx_current = tnx['Close'].iloc[-1]
        
        ndx = yf.Ticker("^NDX").history(period="3mo")
        ndx_sma_50 = ndx['Close'].tail(50).mean()
        ndx_current = ndx['Close'].iloc[-1]

        macro_blocker = False
        macro_warning = ""
        if tnx_current > tnx_sma_50 or ndx_current < ndx_sma_50:
            macro_blocker = True
            macro_warning = "🚨 HARD OVERRIDE: Growth Liquidity Crisis (Rates rising or Nasdaq broken)"
        elif vix > 25:
            macro_warning = "⚠️ WARNING: Elevated Fear (VIX > 25). Size reduced 50%."

        # ---------------------------------------------------------
        # MODULE 3 & 4: SENTIMENT ENGINES (UI DRIVEN)
        # ---------------------------------------------------------
        # We handle News and FinTwit in the UI tabs to avoid complex async API calls in this simple script.
        
        # ---------------------------------------------------------
        # TABS LAYOUT
        # ---------------------------------------------------------
        tab1, tab2, tab3, tab4 = st.tabs(["Chart & Techs", "Macro & Fundamentals", "NLP Sentiment Engine", "Final Trade Ticket"])
        
        with tab1:
            st.subheader("Growth Technical Matrix")
            fig = make_subplots(rows=1, cols=1)
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_10'], mode='lines', name='EMA 10', line=dict(color='blue', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='EMA 20', line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='red', width=2)))
            
            # Plot Buy Levels
            colors = ['lime', 'cyan', 'purple']
            for i, (name, level) in enumerate(buy_levels.items()):
                fig.add_hline(y=level, line_dash="dash", line_color=colors[i%3], annotation_text=f"{name}: ${level:.2f}")
                
            # Plot Capitulation Spikes
            cap_dates = df[df['Capitulation'] == True].index
            cap_lows = df[df['Capitulation'] == True]['Low']
            fig.add_trace(go.Scatter(x=cap_dates, y=cap_lows, mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Capitulation'))
            
            fig.update_layout(xaxis_rangeslider_visible=False, height=700)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.metric("VIX", f"{vix:.2f}", "Elevated" if vix > 25 else "Normal")
            st.metric("10Y Yield vs 50SMA", f"{tnx_current:.2f}", "Dangerous" if tnx_current > tnx_sma_50 else "Safe")
            st.metric("Nasdaq vs 50SMA", f"{ndx_current:.2f}", "Broken" if ndx_current < ndx_sma_50 else "Healthy")
            if macro_blocker:
                st.error(macro_warning)
            elif macro_warning:
                st.warning(macro_warning)
            else:
                st.success("✅ Macro environment is clear for growth stocks.")

            st.subheader("Microstructure Checks")
            st.write(f"**Liquidity Filter:** Avg Daily Volume: {avg_vol:,.0f} {'(LOW - Size Halved)' if avg_vol < 100000 else '(OK)'}")
            if current_price < fib_50:
                st.warning("⚠️ Deep Retracement: Price is below 50% Fib. Growth thesis may be compromised.")
            else:
                st.success("✅ Shallow Retracement: Price holding above critical 50% Fib level.")

        with tab3:
            st.subheader("Automated News Sentiment (Free Yahoo Feed)")
            news = yf.Ticker(ticker).news
            if news:
                news_score = 0
                pos_words = ['beat', 'raised', 'surge', 'bullish', 'upgrade', 'partnership']
                neg_words = ['missed', 'crash', 'investigation', 'layoff', 'downgrade', 'warning']
                
                for article in news[:5]:
                    title = article.get('title', '')
                    for w in pos_words: news_score += 1 if w in title.lower() else 0
                    for w in neg_words: news_score -= 1 if w in title.lower() else 0
                    st.write(f"- {title}")
                
                st.metric("News Sentiment Score", news_score, "Positive" if news_score > 0 else "Negative")
                if news_score < -2: st.error("NEWS OVERRIDE: Negative catalyst detected.")
            else:
                st.write("No news found.")

            st.markdown("---")
            st.subheader("FinTwit OSINT Distiller (Paste Alpha Here)")
            tweets = st.text_area("Paste tweets/threads from your smart accounts here:", height=150)
            if tweets:
                sentiment_scores = [sia.polarity_params(t)['compound'] for t in tweets.split('\n') if t.strip()]
                avg_score = np.mean(sentiment_scores) if sentiment_scores else 0
                st.metric("FinTwit VADER Score", f"{avg_score:.2f}", "Bullish" if avg_score > 0.2 else ("Bearish" if avg_score < -0.2 else "Neutral"))
                
                # Extract tickers
                tickers_found = re.findall(r'\$([A-Za-z]{2,4})\b', tweets)
                if tickers_found:
                    st.write("**Mentioned Tickers:**", ", ".join(set(tickers_found)))

        with tab4:
            st.subheader("Final Trade Ticket Synthesis")
            
            base_kelly = calculate_kelly()
            kelly_size = base_kelly * portfolio_value
            
            # Apply Reductions
            reduction_reasons = []
            if vix > 25: kelly_size *= 0.5; reduction_reasons.append("High VIX")
            if avg_vol < 100000: kelly_size *= 0.5; reduction_reasons.append("Low Liquidity")
            
            final_decision = "DO NOT BUY"
            decision_color = "red"
            
            if macro_blocker:
                final_decision = "DO NOT BUY (Macro Override)"
            elif buy_levels and not (current_price < fib_50):
                final_decision = "BUY"
                decision_color = "green"
                if reduction_reasons:
                    final_decision = f"REDUCE SIZE (50% cut: {', '.join(reduction_reasons)})"
                    decision_color = "orange"

            st.markdown(f"### <span style='color:{decision_color}'>{final_decision}</span>", unsafe_allow_html=True)
            
            if final_decision != "DO NOT BUY" and final_decision != "DO NOT BUY (Macro Override)":
                for name, level in buy_levels.items():
                    stop_loss = level - (2 * atr)
                    risk_per_share = level - stop_loss
                    shares_to_buy = int(kelly_size / risk_per_share) if risk_per_share > 0 else 0
                    cost = shares_to_buy * level
                    
                    st.markdown(f"**Level: {name}**")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Entry Price", f"${level:.2f}")
                    col2.metric("Stop Loss (2ATR)", f"${stop_loss:.2f}")
                    col3.metric("Shares to Buy", shares_to_buy)
                    col4.metric("Capital Allocated", f"${cost:,.2f}")
                    st.divider()

    except Exception as e:
        st.error(f"Error fetching data: {e}")
