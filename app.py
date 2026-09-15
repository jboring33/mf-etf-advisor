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

st.title("🛡️ Institutional-Grade ETF Decision Engine")

# 2. Live Macro Data Retrieval
@st.cache_data(ttl=1800)
def fetch_macro_data():
    try:
        tickers = ["^VIX", "^TNX"]
        macro_df = yf.download(tickers, period="3m", progress=False)["Close"]
        
        vix_current = float(macro_df["^VIX"].iloc[-1])
        vix_50ma = float(macro_df["^VIX"].rolling(50).mean().iloc[-1])
        tnx_current = float(macro_df["^TNX"].iloc[-1])
        
        return {
            "vix": vix_current,
            "vix_50ma": vix_50ma,
            "vix_elevated": vix_current > vix_50ma,
            "tnx": tnx_current
        }
    except Exception:
        return {"vix": 18.0, "vix_50ma": 16.0, "vix_elevated": False, "tnx": 4.0}

macro_data = fetch_macro_data()

# Sidebar Setup & Dynamic Macro Selection
st.sidebar.header("Configuration & Live Macro")

vix_status = "ELEVATED ⚠️" if macro_data["vix_elevated"] else "NORMAL ✅"
st.sidebar.metric("Live VIX", f"{macro_data['vix']:.2f}", delta=vix_status, delta_color="inverse")
st.sidebar.metric("10-Year Yield (^TNX)", f"{macro_data['tnx']:.2f}%")

# Auto-detect default macro regime based on live VIX & Yields
default_macro_idx = 2 if macro_data["vix_elevated"] or macro_data["vix"] > 20 else 0

macro_preset = st.sidebar.selectbox(
    "Macro Backdrop Preset",
    ["Fed Easing / Bullish Macro", "Neutral / Transition", "Tightening / High Volatility / Bearish"],
    index=default_macro_idx
)

# Dynamic Factor Weighting
macro_weights = {
    "Fed Easing / Bullish Macro": {"macro": 0.50, "timing": 0.30, "risk": 0.20},
    "Neutral / Transition": {"macro": 0.35, "timing": 0.35, "risk": 0.30},
    "Tightening / High Volatility / Bearish": {"macro": 0.20, "timing": 0.35, "risk": 0.45}
}
weights = macro_weights[macro_preset]

# Ticker Input
ticker_input = st.sidebar.text_input("Tickers (comma separated):", value=initial_tickers)
st.query_params["tickers"] = ticker_input
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# 3. Robust Data Retrieval & Calculation
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
        
        df = etf_data[ticker]["hist"].copy()
        info = etf_data[ticker]["info"]
        
        close = float(df['Close'].iloc[-1])
        
        # --- 1. Macro Trend (200-Day SMA & Distance) ---
        sma_200 = float(df['Close'].rolling(200).mean().iloc[-1]) if len(df) >= 200 else close
        pct_above_200 = ((close - sma_200) / sma_200) * 100
        macro_score = float(np.clip(50 + (pct_above_200 * 5), 0, 100))
        
        # --- 2. Accurate Short-Term Technical Timing (Wilder's RSI & Stoch) ---
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Exponential Moving Average for Wilder's Smoothing
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = float((100 - (100 / (1 + rs))).iloc[-1])
        
        # 14-Day Stochastic %K
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        stoch_k = float(((df['Close'] - low_14) / (high_14 - low_14) * 100).iloc[-1]) if len(df) >= 14 else 50.0

        # Timing Score Logic
        if rsi >= 70:
            timing_score = 25.0   # Overbought peak: Avoid buying
        elif rsi <= 35:
            timing_score = 90.0   # Oversold dip: Prime entry opportunity
        else:
            timing_score = float(np.clip(100 - abs(rsi - 45) * 2.2, 35, 80))

        # --- 3. Capital Preservation & OBV Trend ---
        returns = df['Close'].pct_change().dropna()
        ann_vol = float(returns.std() * np.sqrt(252)) if not returns.empty else 0.0
        
        roll_max = df['Close'].cummax()
        drawdown = (df['Close'] - roll_max) / roll_max
        max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
        
        # On-Balance Volume (OBV) Calculation
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        obv_ma20 = obv.rolling(20).mean().iloc[-1]
        obv_trend = "Bullish" if obv.iloc[-1] > obv_ma20 else "Bearish"
        
        vol_score = float(np.clip(100 - (ann_vol * 220), 0, 100))
        dd_score = float(np.clip(100 + (max_dd * 220), 0, 100))
        obv_score = 85.0 if obv_trend == "Bullish" else 35.0
        
        risk_preservation_score = (vol_score * 0.4) + (dd_score * 0.4) + (obv_score * 0.2)

        # Composite Score Calculation
        composite_score = (
            (macro_score * weights['macro']) +
            (timing_score * weights['timing']) +
            (risk_preservation_score * weights['risk'])
        )

        # Multi-Factor Signal Decision Engine
        if composite_score >= 68 and rsi < 62:
            signal = "BUY (Oversold Dip)"
            execution = "Full Target Position"
        elif composite_score >= 60 and rsi >= 62:
            signal = "HOLD / DCA ONLY"
            execution = "Pause Lump Sum (Short-Term High)"
        elif composite_score <= 45:
            signal = "SELL / CAPITAL PRESERVE"
            execution = "0% Target Allocation"
        else:
            signal = "HOLD / NEUTRAL"
            execution = "50% Target Position"

        div_yield = info.get("dividendYield", 0) or 0

        results.append({
            "Ticker": ticker,
            "Signal": signal,
            "Execution Guidance": execution,
            "Composite Score": round(composite_score, 1),
            "200D Trend": f"{pct_above_200:+.1f}%",
            "14D RSI": round(rsi, 1),
            "14D Stoch %K": round(stoch_k, 1),
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

        st.subheader("Factor Score Comparison")
        st.bar_chart(res_df.set_index("Ticker")[["Composite Score", "Capital Preservation"]])
    else:
        st.warning("No valid data retrieved for specified tickers.")
