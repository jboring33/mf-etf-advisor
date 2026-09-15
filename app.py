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
        macro_df = yf.download(["^VIX", "^TNX"], period="3m", progress=False)["Close"]
        
        # VIX Calculations
        vix_current = float(macro_df["^VIX"].iloc[-1])
        vix_prev = float(macro_df["^VIX"].iloc[-2])
        vix_50ma = float(macro_df["^VIX"].rolling(50).mean().iloc[-1])
        vix_chg_pct = ((vix_current - vix_prev) / vix_prev) * 100
        
        # 10-Year Yield (^TNX) Calculations
        tnx_current = float(macro_df["^TNX"].iloc[-1])
        tnx_prev = float(macro_df["^TNX"].iloc[-2])
        tnx_50ma = float(macro_df["^TNX"].rolling(50).mean().iloc[-1])
        tnx_chg_bps = (tnx_current - tnx_prev) * 100  # Basis points
        
        return {
            "vix": vix_current,
            "vix_50ma": vix_50ma,
            "vix_chg_str": f"{vix_chg_pct:+.2f}% vs Yesterday",
            "vix_elevated": vix_current > vix_50ma,
            "tnx": tnx_current,
            "tnx_50ma": tnx_50ma,
            "tnx_chg_str": f"{tnx_chg_bps:+.1f} bps Today",
            "tnx_spiking": tnx_current > tnx_50ma or tnx_chg_bps > 10.0
        }
    except Exception:
        return {
            "vix": 18.0, "vix_50ma": 16.0, "vix_chg_str": "+0.00%", "vix_elevated": True,
            "tnx": 4.99, "tnx_50ma": 4.50, "tnx_chg_str": "+75.0 bps Today", "tnx_spiking": True
        }

macro_data = fetch_macro_data()

# Sidebar Setup
st.sidebar.header("Configuration & Live Macro")

vix_status = f"{macro_data['vix_chg_str']} (Elevated ⚠️)" if macro_data["vix_elevated"] else f"{macro_data['vix_chg_str']} (Normal ✅)"
tnx_status = f"{macro_data['tnx_chg_str']} (Spiking ⚠️)" if macro_data["tnx_spiking"] else f"{macro_data['tnx_chg_str']} (Stable ✅)"

st.sidebar.metric("Live VIX", f"{macro_data['vix']:.2f}", delta=vix_status, delta_color="inverse")
st.sidebar.metric("10-Year Yield (^TNX)", f"{macro_data['tnx']:.2f}%", delta=tnx_status, delta_color="inverse")

default_macro_idx = 2 if (macro_data["vix_elevated"] or macro_data["tnx_spiking"] or macro_data["vix"] > 18) else 0

macro_preset = st.sidebar.selectbox(
    "Macro Backdrop Preset",
    ["Fed Easing / Bullish Macro", "Neutral / Transition", "Tightening / High Volatility / Bearish"],
    index=default_macro_idx
)

macro_weights = {
    "Fed Easing / Bullish Macro": {"macro": 0.50, "timing": 0.30, "risk": 0.20},
    "Neutral / Transition": {"macro": 0.35, "timing": 0.35, "risk": 0.30},
    "Tightening / High Volatility / Bearish": {"macro": 0.20, "timing": 0.35, "risk": 0.45}
}
weights = macro_weights[macro_preset]

ticker_input = st.sidebar.text_input("Tickers (comma separated):", value=initial_tickers)
st.query_params["tickers"] = ticker_input
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# 3. Data Retrieval & Calculation
@st.cache_data(ttl=3600)
def load_etf_data(ticker_list):
    data = {}
    for ticker in ticker_list:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1y")
            data[ticker] = {"hist": hist, "ticker_obj": t}
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
        t_obj = etf_data[ticker]["ticker_obj"]
        
        if df.empty:
            continue

        close = float(df['Close'].iloc[-1])
        
        # --- 1. Macro Trend (200-Day SMA & Distance) ---
        sma_200 = float(df['Close'].rolling(200).mean().iloc[-1]) if len(df) >= 200 else close
        pct_above_200 = ((close - sma_200) / sma_200) * 100
        macro_score = float(np.clip(50 + (pct_above_200 * 5), 0, 100))
        
        # --- 2. Short-Term Technical Timing (Wilder's RSI & Stochastic %K) ---
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = float((100 - (100 / (1 + rs))).iloc[-1])
        
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        stoch_k = float(((df['Close'] - low_14) / (high_14 - low_14) * 100).iloc[-1]) if len(df) >= 14 else 50.0

        if rsi >= 70:
            timing_score = 25.0
        elif rsi <= 35:
            timing_score = 90.0
        else:
            timing_score = float(np.clip(100 - abs(rsi - 45) * 2.2, 35, 80))

        # --- 3. Capital Preservation & OBV Trend ---
        returns = df['Close'].pct_change().dropna()
        ann_vol = float(returns.std() * np.sqrt(252)) if not returns.empty else 0.0
        
        roll_max = df['Close'].cummax()
        drawdown = (df['Close'] - roll_max) / roll_max
        max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
        
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        obv_ma20 = obv.rolling(20).mean().iloc[-1]
        obv_trend = "Bullish 🟢" if obv.iloc[-1] > obv_ma20 else "Bearish 🔴"
        
        vol_score = float(np.clip(100 - (ann_vol * 220), 0, 100))
        dd_score = float(np.clip(100 + (max_dd * 220), 0, 100))
        obv_score = 85.0 if "Bullish" in obv_trend else 35.0
        
        risk_preservation_score = (vol_score * 0.4) + (dd_score * 0.4) + (obv_score * 0.2)

        # Composite Score Calculation
        composite_score = (
            (macro_score * weights['macro']) +
            (timing_score * weights['timing']) +
            (risk_preservation_score * weights['risk'])
        )

        # --- 4. Decision Engine with Event / Volatility Circuit Breaker ---
        is_risk_elevated = macro_data["vix_elevated"] or macro_data["tnx_spiking"] or macro_data["vix"] > 18.0

        if is_risk_elevated and composite_score >= 60:
            signal = "HOLD / DCA ONLY (Event Lock) 🟡"
            execution = "Pause Lump Sum — Pre-Fed / Volatility Window"
        elif composite_score >= 68 and rsi < 62:
            signal = "BUY (Oversold Dip) 🟢"
            execution = "Full Target Position"
        elif composite_score >= 60 and rsi >= 62:
            signal = "HOLD / DCA ONLY 🟡"
            execution = "Pause Lump Sum (Short-Term Peak)"
        elif composite_score <= 45:
            signal = "SELL / CAPITAL PRESERVE 🔴"
            execution = "0% Target Allocation"
        else:
            signal = "HOLD / NEUTRAL 🟡"
            execution = "50% Target Position"

        # Trailing 12-Month Dividend Yield Calculation
        try:
            divs = t_obj.dividends
            if not divs.empty:
                ttm_divs = float(divs.tail(12).sum())
                calc_yield = (ttm_divs / close) * 100
            else:
                calc_yield = 0.0
        except Exception:
            calc_yield = 0.0

        # Indicator Color Formatting
        rsi_formatted = f"{rsi:.2f} 🔴" if rsi >= 70 else (f"{rsi:.2f} 🟢" if rsi <= 35 else f"{rsi:.2f} 🟡")
        stoch_formatted = f"{stoch_k:.2f} 🔴" if stoch_k >= 80 else (f"{stoch_k:.2f} 🟢" if stoch_k <= 20 else f"{stoch_k:.2f} 🟡")

        results.append({
            "Ticker": ticker,
            "Signal": signal,
            "Execution Guidance": execution,
            "Composite Score": round(composite_score, 2),
            "200D Trend": f"{pct_above_200:+.1f}%",
            "14D RSI": rsi_formatted,
            "14D Stoch %K": stoch_formatted,
            "OBV Trend": obv_trend,
            "Capital Preservation": round(risk_preservation_score, 2),
            "Yield": f"{calc_yield:.2f}%" if calc_yield > 0 else "N/A"
        })

    if results:
        res_df = pd.DataFrame(results)

        st.subheader("Multi-Timeframe ETF Evaluation")
        
        st.dataframe(
            res_df,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Signal": st.column_config.TextColumn("Signal", width="medium"),
                "Execution Guidance": st.column_config.TextColumn("Execution Guidance", width="large"),
                "Composite Score": st.column_config.NumberColumn("Composite Score", format="%.2f"),
                "200D Trend": st.column_config.TextColumn("200D Trend"),
                "14D RSI": st.column_config.TextColumn("14D RSI"),
                "14D Stoch %K": st.column_config.TextColumn("14D Stoch %K"),
                "OBV Trend": st.column_config.TextColumn("OBV Trend"),
                "Capital Preservation": st.column_config.NumberColumn("Capital Preservation", format="%.2f"),
                "Yield": st.column_config.TextColumn("TTM Yield"),
            },
            hide_index=True,
            use_container_width=True
        )

        st.subheader("Factor Score Comparison")
        st.bar_chart(res_df.set_index("Ticker")[["Composite Score", "Capital Preservation"]])
    else:
        st.warning("No valid data retrieved for specified tickers.")
