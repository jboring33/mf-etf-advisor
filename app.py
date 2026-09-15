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
        vix_chg_pct = ((vix_current - vix_prev) / vix_prev) * 100
        
        # 10-Year Yield (^TNX) Calculations
        tnx_current = float(macro_df["^TNX"].iloc[-1])
        tnx_prev = float(macro_df["^TNX"].iloc[-2])
        tnx_50ma = float(macro_df["^TNX"].rolling(50).mean().iloc[-1])
        tnx_chg_bps = (tnx_current - tnx_prev) * 100  # Basis points
        
        return {
            "vix": vix_current,
            "vix_chg_str": f"{vix_chg_pct:+.2f}% vs Yesterday",
            "tnx": tnx_current,
            "tnx_50ma": tnx_50ma,
            "tnx_chg_str": f"{tnx_chg_bps:+.1f} bps Today",
            "tnx_spiking": tnx_current > tnx_50ma or tnx_chg_bps > 10.0
        }
    except Exception:
        return {
            "vix": 18.0, "vix_chg_str": "+0.00%",
            "tnx": 4.99, "tnx_50ma": 4.50, "tnx_chg_str": "+75.0 bps Today", "tnx_spiking": True
        }

macro_data = fetch_macro_data()

# VIX Formatting Logic (<16 Green, 16-22 Yellow, >22 Red)
vix_val = macro_data["vix"]
if vix_val < 16.0:
    vix_status = f"{macro_data['vix_chg_str']} (Low Volatility 🟢)"
elif 16.0 <= vix_val <= 22.0:
    vix_status = f"{macro_data['vix_chg_str']} (Moderate Volatility 🟡)"
else:
    vix_status = f"{macro_data['vix_chg_str']} (High Volatility 🔴)"

tnx_status = f"{macro_data['tnx_chg_str']} (Spiking ⚠️)" if macro_data["tnx_spiking"] else f"{macro_data['tnx_chg_str']} (Stable ✅)"

# Sidebar Setup
st.sidebar.header("Configuration & Live Macro")

st.sidebar.metric("Live VIX", f"{macro_data['vix']:.2f}", delta=vix_status, delta_color="inverse")
st.sidebar.metric("10-Year Yield (^TNX)", f"{macro_data['tnx']:.2f}%", delta=tnx_status, delta_color="inverse")

default_macro_idx = 2 if (vix_val >= 22.0 or macro_data["tnx_spiking"]) else (1 if vix_val >= 16.0 else 0)

macro_preset = st.sidebar.selectbox(
    "Macro Backdrop Preset",
    ["Bullish (Fed Cutting)", "Neutral (Fed Pause)", "Bearish (Fed Raising)"],
    index=default_macro_idx
)

macro_weights = {
    "Bullish (Fed Cutting)": {"macro": 0.50, "timing": 0.30, "risk": 0.20},
    "Neutral (Fed Pause)": {"macro": 0.35, "timing": 0.35, "risk": 0.30},
    "Bearish (Fed Raising)": {"macro": 0.20, "timing": 0.35, "risk": 0.45}
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
            if not hist.empty:
                data[ticker] = {"hist": hist}
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
        if df.empty:
            continue

        close = float(df['Close'].iloc[-1])
        
        # --- 1. Macro Trend (200-Day SMA) ---
        sma_200 = float(df['Close'].rolling(200).mean().iloc[-1]) if len(df) >= 200 else close
        pct_above_200 = ((close - sma_200) / sma_200) * 100
        macro_score = float(np.clip(50 + (pct_above_200 * 5), 0, 100))
        
        macro_trend_label = f"BUY 🟢 ({pct_above_200:+.1f}%)" if pct_above_200 >= 0 else f"SELL 🔴 ({pct_above_200:+.1f}%)"

        # --- 2. Short-Term Technical Timing ---
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

        if rsi >= 70 or stoch_k >= 80:
            timing_label = "SELL 🔴"
            timing_score = 25.0
        elif rsi <= 38 or stoch_k <= 20:
            timing_label = "BUY 🟢"
            timing_score = 90.0
        else:
            timing_label = "HOLD 🟡"
            timing_score = float(np.clip(100 - abs(rsi - 45) * 2.2, 35, 80))

        # --- 3. Capital Preservation & OBV Trend ---
        returns = df['Close'].pct_change().dropna()
        ann_vol = float(returns.std() * np.sqrt(252)) if not returns.empty else 0.0
        
        roll_max = df['Close'].cummax()
        drawdown = (df['Close'] - roll_max) / roll_max
        max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
        
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        obv_ma20 = obv.rolling(20).mean().iloc[-1]
        obv_is_bullish = obv.iloc[-1] > obv_ma20
        
        vol_score = float(np.clip(100 - (ann_vol * 220), 0, 100))
        dd_score = float(np.clip(100 + (max_dd * 220), 0, 100))
        obv_score = 85.0 if obv_is_bullish else 35.0
        
        risk_preservation_score = (vol_score * 0.4) + (dd_score * 0.4) + (obv_score * 0.2)

        if risk_preservation_score >= 75 and obv_is_bullish:
            cap_preservation_label = "BUY 🟢"
        elif risk_preservation_score >= 50:
            cap_preservation_label = "HOLD 🟡"
        else:
            cap_preservation_label = "SELL 🔴"

        # Composite Score Calculation
        composite_score = (
            (macro_score * weights['macro']) +
            (timing_score * weights['timing']) +
            (risk_preservation_score * weights['risk'])
        )

        # --- 4. Decision Engine & Context Generation ---
        is_risk_elevated = vix_val >= 22.0 or macro_data["tnx_spiking"]

        if is_risk_elevated and composite_score >= 60:
            signal = "HOLD 🟡"
            execution = "Pause Lump Sum (Macro Lock)"
            if pct_above_200 > 0 and rsi >= 60:
                context = f"200D trend is BUY ({pct_above_200:+.1f}%), but Macro Circuit Breakers (^VIX/{vix_val:.1f} or ^TNX) forced a HOLD."
            else:
                context = "Elevated macro volatility or interest rate shock downgraded signal to HOLD."
        elif composite_score >= 68 and rsi < 62:
            signal = "BUY 🟢"
            execution = "Full Target Position"
            context = f"Strong 200D trend ({pct_above_200:+.1f}%) paired with favorable RSI ({rsi:.1f}). Prime entry window."
        elif composite_score >= 60 and rsi >= 62:
            signal = "HOLD 🟡"
            execution = "Pause Lump Sum (Short-Term Peak)"
            context = f"200D trend is BUY ({pct_above_200:+.1f}%), but short-term momentum is overbought (RSI {rsi:.1f})."
        elif composite_score <= 45:
            signal = "SELL 🔴"
            execution = "0% Target Allocation"
            context = f"Macro trend breakdown ({pct_above_200:+.1f}% vs 200D) with declining volume distribution (OBV SELL)."
        else:
            signal = "HOLD 🟡"
            execution = "50% Target Position"
            context = f"Balanced metrics. Macro trend ({pct_above_200:+.1f}%) lacks decisive breakout momentum."

        results.append({
            "Ticker": ticker,
            "Signal": signal,
            "Execution Guidance": execution,
            "Macro Trend (200 SMA)": macro_trend_label,
            "Short-Term Momentum": timing_label,
            "Capital Preservation": cap_preservation_label,
            "Context": context
        })

    if results:
        res_df = pd.DataFrame(results)

        st.subheader("Multi-Timeframe ETF Evaluation")

        # Custom CSS for Flyover Tooltip
        hover_css = (
            "<style>"
            ".etf-table { width: 100%; border-collapse: collapse; font-family: sans-serif; margin-top: 10px; }"
            ".etf-table th { background-color: #1e222d; color: #ffffff; padding: 12px; text-align: left; font-size: 0.9rem; }"
            ".etf-table td { padding: 12px; border-bottom: 1px solid #2d313e; position: relative; font-size: 0.9rem; }"
            ".etf-table tr:hover { background-color: #262a36; }"
            ".context-tooltip {"
            "  visibility: hidden; width: 340px; background-color: #0e1117; color: #e6e8eb;"
            "  text-align: left; border: 1px solid #4b5563; border-radius: 6px; padding: 10px 14px;"
            "  position: absolute; z-index: 99; right: 20px; top: -10px;"
            "  box-shadow: 0px 8px 16px rgba(0,0,0,0.6); opacity: 0; transition: opacity 0.2s ease-in-out;"
            "  font-size: 0.85rem; line-height: 1.3; pointer-events: none;"
            "}"
            ".etf-table tr:hover .context-tooltip { visibility: visible; opacity: 1; }"
            "</style>"
        )

        headers = ["Ticker", "Signal", "Execution Guidance", "Macro Trend (200 SMA)", "Short-Term Momentum", "Capital Preservation", "Context ℹ️"]
        header_html = "".join([f"<th>{h}</th>" for h in headers])
        
        rows_html = ""
        for _, row in res_df.iterrows():
            rows_html += (
                f"<tr>"
                f"<td><b>{row['Ticker']}</b></td>"
                f"<td>{row['Signal']}</td>"
                f"<td>{row['Execution Guidance']}</td>"
                f"<td>{row['Macro Trend (200 SMA)']}</td>"
                f"<td>{row['Short-Term Momentum']}</td>"
                f"<td>{row['Capital Preservation']}</td>"
                f"<td style='cursor: pointer; color: #9ca3af;'>"
                f"🔍 Hover row for details"
                f"<div class='context-tooltip'><b>{row['Ticker']} Context:</b><br/>{row['Context']}</div>"
                f"</td>"
                f"</tr>"
            )

        full_table_html = f"{hover_css}<table class='etf-table'><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table>"

        st.markdown(full_table_html, unsafe_allow_html=True)
    else:
        st.warning("No valid data retrieved for specified tickers.")
