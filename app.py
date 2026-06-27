import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import asyncio
from playwright.async_api import async_playwright

# --- INITIALIZATION ---
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

st.set_page_config(layout="wide", page_title="Growth Terminal")
st.title("📊 Institutional Growth Stock Terminal")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Control Panel")
ticker = st.sidebar.text_input("Ticker (e.g., SIVE.ST, IQE, NBIS)", value="AAPL").upper()
portfolio_value = st.sidebar.number_input("Total Portfolio Value ($)", value=100000)
run_btn = st.sidebar.button("Run Analysis", type="primary")

# --- HELPER FUNCTIONS ---
def calculate_kelly(win_rate=0.45, reward_risk=2.5):
    kelly = win_rate - ((1 - win_rate) / reward_risk)
    return max(kelly / 2, 0) # Half-Kelly base

@st.cache_data(ttl=3600)
def get_data(tick):
    return yf.Ticker(tick).history(period="1y")

async def fetch_serenity():
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=['--no-sandbox'])
            page = await browser.new_page()
            await page.goto("https://aichainmap.com/serenity/", timeout=20000)
            await page.wait_for_timeout(6000) # Wait 6 seconds for React/JS to render
            text = await page.inner_text('body')
            await browser.close()
            return text
    except Exception as e:
        return f"SCRAPER_ERROR: {str(e)}"

# --- MAIN APP LOGIC ---
if run_btn:
    try:
        df = get_data(ticker)
        if df.empty:
            st.error("Ticker not found or no data.")
            st.stop()
        
        # =========================================================
        # MODULE 1: GROWTH TECHNICAL ENGINE (SANITY FILTERS)
        # =========================================================
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Manual ATR calculation
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        current_price = df['Close'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        
        # Dynamic Fibonacci (Use 6-month data if stock is down >50%)
        high_1y = df['High'].max()
        if high_1y > (current_price * 1.5):
            recent_df = df.tail(126)
            high_base = recent_df['High'].max()
            low_base = recent_df['Low'].min()
        else:
            high_base = high_1y
            low_base = df['Low'].min()
            
        diff = high_base - low_base
        fib_50 = high_base - (0.5 * diff)
        
        # Capitulation & Gaps (Strictly last 30 days)
        df['Avg_Range'] = (df['High'] - df['Low']).rolling(20).mean()
        df['Avg_Vol'] = df['Volume'].rolling(20).mean()
        df['Capitulation'] = ((df['High'] - df['Low']) > (2.5 * df['Avg_Range'])) & (df['Volume'] > (3 * df['Avg_Vol']))
        
        recent_30 = df.tail(30)
        recent_caps = recent_30[recent_30['Capitulation'] == True]
        gaps = recent_30[recent_30['Open'] > recent_30['Close'].shift(1) * 1.02]
        
        # VWAP
        df['VWAP_20'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).rolling(20).sum() / df['Volume'].rolling(20).sum()
        
        # --- THE SANITY FILTERS ---
        max_acceptable_drop = current_price * 0.70 
        avg_vol = df['Avg_Vol'].iloc[-1]
        
        buy_levels = {}
        if not recent_caps.empty:
            cap_level = recent_caps['Low'].min()
            if current_price > cap_level > max_acceptable_drop:
                buy_levels['Capitulation Spike'] = cap_level
                
        if not gaps.empty:
            gap_support = gaps['Open'].min()
            if current_price > gap_support > max_acceptable_drop:
                buy_levels['Gap Fill Support'] = gap_support
            
        vwap = df['VWAP_20'].iloc[-1]
        if not np.isnan(vwap):
            if current_price > vwap > max_acceptable_drop:
                buy_levels['20D VWAP'] = vwap

        # =========================================================
        # MODULE 2: MACRO MATRIX
        # =========================================================
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
            macro_warning = "🚨 HARD OVERRIDE: Growth Liquidity Crisis"
        elif vix > 25:
            macro_warning = "⚠️ WARNING: Elevated Fear (VIX > 25). Size reduced 50%."

        # =========================================================
        # UI LAYOUT
        # =========================================================
        tab1, tab2, tab3, tab4 = st.tabs(["Chart & Techs", "Macro & Fundamentals", "News & Scrapers", "Final Trade Ticket"])
        
        with tab1:
            st.subheader("Growth Technical Matrix")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_10'], mode='lines', name='EMA 10', line=dict(color='blue', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='EMA 20', line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='red', width=2)))
            
            colors = ['lime', 'cyan', 'purple']
            for i, (name, level) in enumerate(buy_levels.items()):
                fig.add_hline(y=level, line_dash="dash", line_color=colors[i%3], annotation_text=f"{name}: ${level:.2f}")
                
            cap_dates = df[df['Capitulation'] == True].index
            cap_lows = df[df['Capitulation'] == True]['Low']
            fig.add_trace(go.Scatter(x=cap_dates, y=cap_lows, mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Capitulation'))
            
            fig.update_layout(xaxis_rangeslider_visible=False, height=700)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.metric("VIX", f"{vix:.2f}", "Elevated" if vix > 25 else "Normal")
            st.metric("10Y Yield vs 50SMA", f"{tnx_current:.2f}", "Dangerous" if tnx_current > tnx_sma_50 else "Safe")
            st.metric("Nasdaq vs 50SMA", f"{ndx_current:.2f}", "Broken" if ndx_current < ndx_sma_50 else "Healthy")
            if macro_blocker: st.error(macro_warning)
            elif macro_warning: st.warning(macro_warning)
            else: st.success("✅ Macro environment clear.")
            
            st.subheader("Microstructure Checks")
            st.write(f"**Liquidity:** {avg_vol:,.0f} shares/day {'(LOW - Size Halved)' if avg_vol < 100000 else '(OK)'}")
            if current_price < fib_50: st.warning("⚠️ Deep Retracement: Below 50% Fib.")
            else: st.success("✅ Shallow Retracement: Holding above 50% Fib.")

        with tab3:
            st.subheader("1. Automated Serenity Ingestion (Playwright)")
            if st.button("Fetch Serenity Analysis", key="serenity_btn"):
                with st.spinner("Launching headless browser to aichainmap.com..."):
                    serenity_text = asyncio.run(fetch_serenity())
                    
                    if "SCRAPER_ERROR" in serenity_text:
                        st.error("Failed to scrape Serenity (likely Cloudflare block or missing Chromium packages.txt).")
                    else:
                        st.success("Successfully ingested Serenity Alpha!")
                        with st.expander("View Raw Serenity Text"):
                            st.text(serenity_text[:1500] + "...")

            st.markdown("---")
            st.subheader("2. Free Yahoo News Sentiment")
            news = yf.Ticker(ticker).news
            news_score = 0
            if news:
                pos_words = ['beat', 'raised', 'surge', 'bullish', 'upgrade', 'partnership']
                neg_words = ['missed', 'crash', 'investigation', 'layoff', 'downgrade', 'warning']
                for article in news[:5]:
                    title = article.get('title', '')
                    for w in pos_words: news_score += 1 if w in title.lower() else 0
                    for w in neg_words: news_score -= 1 if w in title.lower() else 0
                    st.write(f"- {title}")
                st.metric("News Sentiment Score", news_score, "Positive" if news_score > 0 else "Negative")
                if news_score < -2:
                    st.error("News Override: Extreme negative catalyst detected.")
            else:
                st.write("No news found for this ticker.")

        with tab4:
            st.subheader("Final Trade Ticket Synthesis")
            
            base_kelly = calculate_kelly()
            kelly_size = base_kelly * portfolio_value
            
            reduction_reasons = []
            
            # Apply Reductions
            if vix > 25: kelly_size *= 0.5; reduction_reasons.append("High VIX")
            if avg_vol < 100000: kelly_size *= 0.5; reduction_reasons.append("Low Liquidity")
            
            final_decision = "DO NOT BUY"
            decision_color = "red"
            
            if macro_blocker:
                final_decision = "DO NOT BUY (Macro Override)"
            elif not buy_levels:
                final_decision = "DO NOT BUY (No Valid Levels - Stock likely in structural breakdown)"
            elif current_price < fib_50:
                final_decision = "DO NOT BUY (Deep Retracement > 50%)"
            else:
                final_decision = "BUY"
                decision_color = "green"
                if reduction_reasons:
                    final_decision = f"REDUCE SIZE (50% cut: {', '.join(reduction_reasons)})"
                    decision_color = "orange"

            st.markdown(f"### <span style='color:{decision_color}'>{final_decision}</span>", unsafe_allow_html=True)
            
            if "BUY" in final_decision:
                for name, level in buy_levels.items():
                    stop_loss = max(level - (2 * atr), 0.01) # Never negative
                    risk_per_share = level - stop_loss
                    if risk_per_share <= 0: continue # Skip if math breaks
                        
                    shares_to_buy = int(kelly_size / risk_per_share)
                    cost = shares_to_buy * level
                    
                    st.markdown(f"**Level: {name}**")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Entry Price", f"${level:.2f}")
                    col2.metric("Stop Loss (2ATR)", f"${stop_loss:.2f}")
                    col3.metric("Shares to Buy", shares_to_buy)
                    col4.metric("Capital Allocated", f"${cost:,.2f}")
                    st.divider()

    except Exception as e:
        st.error(f"Critical Data Error: {e}")
