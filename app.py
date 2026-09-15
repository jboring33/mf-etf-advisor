import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Multi-Timeframe ETF Engine", layout="wide")

# 1. URL Parameter Parsing
query_params = st.query_params
default_tickers = "SPY, QQQ, SCHD, SGOV, JAAA"

raw_tickers = query_params.get("tickers", default_tickers)
if isinstance(raw_tickers, list):
    raw_tickers = raw_tickers[0]

initial_tickers = str(raw_tickers).strip()

st.title("🛡️ Multi-Timeframe ETF Advisor")

# Sidebar Controls
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Tickers (comma separated):", value=initial_tickers)
st.query_params["tickers"] = ticker_input

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

macro_preset = st.sidebar.selectbox(
    "Macro Backdrop Preset",
    ["Neutral / Transition", "Fed Easing / Bullish Macro", "Tightening / High Volatility / Bearish"]
)

# Macro Weight Adjustments
macro_weights = {
    "Neutral / Transition": {"macro": 0.35, "timing": 0.35, "risk": 0.30},
    "Fed Easing / Bullish Macro": {"macro": 0.50, "timing": 0.30, "risk": 0.20},
    "Tightening / High Volatility / Bearish": {"macro": 0.20, "timing": 0.35, "risk": 0.45}
}
weights = macro_weights[macro_preset]

st.sidebar.markdown(
    f"**Active Model Weights:**\n"
    f"* Macro Trend (200-SMA): `{weights['macro']*100:.0f}%`\n"
    f"* Short-Term Entry Timing (RSI/Stoch): `{weights['timing']*100:.0f}%`\n"
    f"* Volatility & Capital Preservation: `{weights['risk']*100:.0f}%`"
)

# Fetch Historical Data
@st.cache_data(ttl=3600)
def load_etf_data(ticker_list):
    data = {}
    for ticker in ticker_list:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1y")
            info = t.info
            if not hist.empty:
                data[ticker] = {"hist": hist, "info": info}
        except Exception:
            pass
    return data

if tickers:
    etf_data = load_etf_data(tickers)
    results = []

    for ticker in tickers:
        if ticker not in etf_data:
            continue
        
        df = etf_data[ticker]["hist"]
        info = etf_data[ticker]["info"]
        
        close = df['Close'].iloc[-1]
        
        # --- 1. Macro Trend (200-Day SMA) ---
        sma_200 = df['Close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else close
        macro_score = float(np.clip(50 + ((close - sma_200) / sma_200) * 200, 0, 100))
        
        # --- 2. Short-Term Timing (14-Day RSI & Stochastic) ---
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = float((100 - (100 / (1 + rs))).iloc[-1]) if not rs.empty else 50.0
        
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        stoch_k = float(((df['Close'] - low_14) / (high_14 - low_14) * 100).iloc[-1]) if len(df) >= 14 else 50.0
        
        # Timing Score: Rewards oversold conditions within a bull market
        if rsi > 70:
            timing_score = 30.0  # Overbought: Bad time to enter
        elif rsi < 35:
            timing_score = 90.0  # Oversold dip: High-conviction entry
        else:
            timing_score = float(np.clip(100 - abs(rsi - 45) * 2, 40, 80))

        # --- 3. Downside Risk, OBV, & Volatility Preservation ---
        returns = df['Close'].pct_change().dropna()
        ann_vol = float(returns.std() * np.sqrt(252)) if not returns.empty else 0.0
        
        roll_max = df['Close'].cummax()
        drawdown = (df['Close'] - roll_max) / roll_max
        max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
        
        # OBV Trend (20-day direction)
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        obv_trend = "Bullish" if obv.iloc[-1] > obv.rolling(20).mean().iloc[-1] else "Bearish"
        
        vol_score = float(np.clip(100 - (ann_vol * 200), 0, 100))
        dd_score = float(np.clip(100 + (max_dd * 200), 0, 100))
        obv_score = 80.0 if obv_trend == "Bullish" else 40.0
        
        risk_preservation_score = (vol_score * 0.4) + (dd_score * 0.4) + (obv_score * 0.2)

        # Composite Multi-Factor Score
        composite_score = (
            (macro_score * weights['macro']) +
            (timing_score * weights['timing']) +
            (risk_preservation_score * weights['risk'])
        )

        # Actionable Signal Engine
        if composite_score >= 70 and rsi < 65:
            signal = "STRONG BUY (Oversold Dip)"
            allocation = "100% Target Allocation"
        elif composite_score >= 60 and rsi >= 65:
            signal = "HOLD / DCA ONLY"
            allocation = "Pause Lump Sum (Overbought Short-Term)"
        elif composite_score <= 45:
            signal = "SELL / UNDERWEIGHT"
            allocation = "0% - Capital Preservation Mode"
        else:
            signal = "HOLD / NEUTRAL"
            allocation = "50% Target Allocation"

        div_yield = info.get("dividendYield", 0) or 0

        results.append({
            "Ticker": ticker,
            "Signal": signal,
            "Execution Guidance": allocation,
            "Composite Score": round(composite_score, 1),
            "Macro Trend (200d)": round(macro_score, 1),
            "14-Day RSI": round(rsi, 1),
            "14-Day Stoch %K": round(stoch_k, 1),
            "OBV Trend": obv_trend,
            "Capital Preservation": round(risk_preservation_score, 1),
            "1Y Volatility": f"{ann_vol*100:.1f}%",
            "Yield": f"{div_yield*100:.2f}%" if div_yield else "N/A"
        })

    if results:
        res_df = pd.DataFrame(results)

        st.subheader("Multi-Timeframe ETF Evaluation")
        st.dataframe(
            res_df.style.highlight_max(subset=["Composite Score"], color="#d4edda"),
            use_container_width=True
        )

        st.subheader("Factor Score Breakdown")
        st.bar_chart(res_df.set_index("Ticker")[["Macro Trend (200d)", "Capital Preservation"]])
    else:
        st.warning("No valid data retrieved for the specified tickers.")
