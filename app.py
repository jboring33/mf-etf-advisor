import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Multi-Factor ETF Advisor", layout="wide")

# 1. URL Parameter Parsing (Cleaned)
query_params = st.query_params
default_tickers = "SPY, QQQ, SCHD, SGOV, JAAA"

raw_tickers = query_params.get("tickers", default_tickers)
if isinstance(raw_tickers, list):
    raw_tickers = raw_tickers[0]

initial_tickers = str(raw_tickers).strip()

st.title("🛡️ Multi-Factor ETF Capital Preservation & Return Engine")

# Sidebar Controls
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Tickers (comma separated):", value=initial_tickers)
st.query_params["tickers"] = ticker_input

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

macro_preset = st.sidebar.selectbox(
    "Macro Backdrop Preset",
    ["Neutral / Transition", "Fed Easing / Bullish Macro", "Tightening / High Volatility / Bearish"]
)

# Macro Adjustments
macro_weights = {
    "Neutral / Transition": {"tech": 0.35, "fund": 0.35, "risk": 0.30},
    "Fed Easing / Bullish Macro": {"tech": 0.50, "fund": 0.30, "risk": 0.20},
    "Tightening / High Volatility / Bearish": {"tech": 0.20, "fund": 0.40, "risk": 0.40}
}
weights = macro_weights[macro_preset]

st.sidebar.markdown(
    f"**Active Model Weights:**\n"
    f"* Technical Momentum: `{weights['tech']*100:.0f}%`\n"
    f"* Fundamentals & Yield: `{weights['fund']*100:.0f}%`\n"
    f"* Downside Risk & Volatility: `{weights['risk']*100:.0f}%`"
)

# Fetch Data
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
        
        # Factor 1: Technical Momentum (0 - 100)
        close = df['Close'].iloc[-1]
        sma_200 = df['Close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else close
        trend_score = float(np.clip(50 + ((close - sma_200) / sma_200) * 200, 0, 100))
        
        # Factor 2: Fundamental & Yield Quality (0 - 100)
        div_yield = info.get("dividendYield", 0) or 0
        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        
        yield_score = float(np.clip(div_yield * 1000, 0, 100))
        val_score = 50.0 if pe_ratio is None else float(np.clip(100 - (pe_ratio * 2.5), 10, 90))
        fundamental_score = (yield_score * 0.4) + (val_score * 0.6)

        # Factor 3: Downside Risk & Volatility Preservation (0 - 100)
        returns = df['Close'].pct_change().dropna()
        ann_vol = float(returns.std() * np.sqrt(252)) if not returns.empty else 0.0
        
        roll_max = df['Close'].cummax()
        drawdown = (df['Close'] - roll_max) / roll_max
        max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
        
        vol_score = float(np.clip(100 - (ann_vol * 200), 0, 100))
        dd_score = float(np.clip(100 + (max_dd * 200), 0, 100))
        risk_preservation_score = (vol_score * 0.5) + (dd_score * 0.5)

        # Composite Multi-Factor Score
        composite_score = (
            (trend_score * weights['tech']) +
            (fundamental_score * weights['fund']) +
            (risk_preservation_score * weights['risk'])
        )

        if composite_score >= 68:
            signal = "BUY / OVERWEIGHT"
        elif composite_score <= 45:
            signal = "SELL / UNDERWEIGHT"
        else:
            signal = "HOLD / NEUTRAL"

        results.append({
            "Ticker": ticker,
            "Signal": signal,
            "Composite Score": round(composite_score, 1),
            "Technical Momentum": round(trend_score, 1),
            "Fundamentals & Yield": round(fundamental_score, 1),
            "Capital Preservation": round(risk_preservation_score, 1),
            "1Y Volatility": f"{ann_vol*100:.1f}%",
            "Max Drawdown": f"{max_dd*100:.1f}%",
            "Yield": f"{div_yield*100:.2f}%" if div_yield else "N/A"
        })

    if results:
        res_df = pd.DataFrame(results)

        st.subheader("Composite ETF Evaluation")
        st.dataframe(
            res_df.style.highlight_max(subset=["Composite Score"], color="#d4edda"),
            use_container_width=True
        )

        st.subheader("Multi-Factor Score Comparison")
        st.bar_chart(res_df.set_index("Ticker")[["Technical Momentum", "Fundamentals & Yield", "Capital Preservation"]])
    else:
        st.warning("No valid data retrieved for the specified tickers.")
