# app.py
# Free GEX Webapp (Multi-Expiry • Gamma Flip • Spot Profile • Interactive)
# Uses yfinance (free) + Streamlit + Plotly, with:
# - Load button (prevents rerun spam)
# - Cache (reduces repeat requests)
# - Retry/backoff + friendly handling for Yahoo rate limits (429)

import math
import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Free GEX Dashboard", layout="wide")

# ---------------------------
# Rate-limit resilience
# ---------------------------
def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("too many requests" in msg) or ("rate limit" in msg) or ("429" in msg)

def _sleep_backoff(attempt: int):
    # quick exponential-ish backoff
    waits = [0.0, 0.8, 1.6, 3.2, 6.0]
    time.sleep(waits[min(attempt, len(waits) - 1)])

@st.cache_data(ttl=60 * 60, show_spinner=False)  # 1 hour cache
def get_expirations_cached(ticker: str) -> list[str]:
    return list(yf.Ticker(ticker).options)

@st.cache_data(ttl=30 * 60, show_spinner=False)  # 30 min cache
def load_spot_and_chains_cached(ticker: str, expirations: tuple[str, ...]) -> tuple[float, pd.DataFrame]:
    tk = yf.Ticker(ticker)

    spot = float(tk.fast_info.get("last_price") or tk.info.get("regularMarketPrice") or np.nan)

    rows = []
    for exp in expirations:
        chain = tk.option_chain(exp)
        calls = chain.calls.copy()
        puts = chain.puts.copy()

        calls["type"] = "call"
        puts["type"] = "put"
        calls["expiration"] = exp
        puts["expiration"] = exp

        rows.append(calls)
        rows.append(puts)

    df = pd.concat(rows, ignore_index=True)

    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["openInterest"] = pd.to_numeric(df.get("openInterest"), errors="coerce").fillna(0.0)
    df["impliedVolatility"] = pd.to_numeric(df.get("impliedVolatility"), errors="coerce").clip(lower=0.0)

    return spot, df

def get_expirations_resilient(ticker: str) -> list[str]:
    last_err = None
    for attempt in range(5):
        try:
            _sleep_backoff(attempt)
            exps = get_expirations_cached(ticker)
            if exps:
                return exps
            raise RuntimeError("No expirations returned.")
        except Exception as e:
            last_err = e
            if not _is_rate_limit_error(e):
                break

    if last_err and _is_rate_limit_error(last_err):
        st.warning(
            "Yahoo rate-limited the server IP (common on Streamlit Cloud). "
            "Wait ~1–5 minutes and click **Load options** again. "
            "Once a request succeeds, results are cached to reduce repeats."
        )
        st.stop()

    raise last_err

def load_spot_and_chains_resilient(ticker: str, expirations: list[str]) -> tuple[float, pd.DataFrame]:
    last_err = None
    expt = tuple(expirations)
    for attempt in range(5):
        try:
            _sleep_backoff(attempt)
            return load_spot_and_chains_cached(ticker, expt)
        except Exception as e:
            last_err = e
            if not _is_rate_limit_error(e):
                break

    if last_err and _is_rate_limit_error(last_err):
        st.warning(
            "Yahoo rate-limited the server IP while fetching the option chain. "
            "Wait ~1–5 minutes and click **Load options** again."
        )
        st.stop()

    raise last_err

# ---------------------------
# Black-Scholes gamma + GEX
# ---------------------------
def bs_gamma_matrix(S_grid: np.ndarray, K: np.ndarray, T: float, r: float, q: float, iv: np.ndarray) -> np.ndarray:
    """
    Vectorized Black-Scholes gamma across a spot grid.
    S_grid: (M,) spots
    K: (N,) strikes
    iv: (N,) implied vols (decimal)
    returns: (M, N) gamma
    """
    if T <= 0:
        return np.zeros((S_grid.size, K.size), dtype=float)

    S = S_grid.reshape(-1, 1)     # (M,1)
    K2 = K.reshape(1, -1)         # (1,N)
    iv2 = iv.reshape(1, -1)       # (1,N)

    eps = 1e-12
    iv2 = np.where(iv2 > eps, iv2, np.nan)

    sqrtT = math.sqrt(T)
    d1 = (np.log(S / K2) + (r - q + 0.5 * iv2**2) * T) / (iv2 * sqrtT)

    gamma = norm.pdf(d1) / (S * iv2 * sqrtT)
    gamma = np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
    return gamma

def gex_total_over_spot(gamma: np.ndarray, S_grid: np.ndarray, oi: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """
    Common convention:
      GEX ≈ gamma * OI * 100 * S^2
    gamma: (M,N)
    returns: (M,) net total
    """
    S2 = (S_grid.reshape(-1, 1) ** 2)
    oi2 = oi.reshape(1, -1)
    sign2 = sign.reshape(1, -1)
    gex = gamma * oi2 * 100.0 * S2 * sign2
    return gex.sum(axis=1)

def find_gamma_flip(S_grid: np.ndarray, total_gex: np.ndarray):
    y = total_gex
    s = S_grid
    sign = np.sign(y)
    idx = np.where(sign[:-1] * sign[1:] < 0)[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    x0, y0 = s[i], y[i]
    x1, y1 = s[i + 1], y[i + 1]
    if y1 == y0:
        return float((x0 + x1) / 2.0)
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))

def time_to_expiration_years(exp: str) -> float:
    exp_dt = pd.to_datetime(exp, utc=True)
    now = pd.Timestamp.utcnow()
    return float(max((exp_dt - now).total_seconds(), 0.0) / (365.0 * 24.0 * 3600.0))

# ---------------------------
# UI
# ---------------------------
st.title("Free GEX Webapp (Multi-Expiry • Gamma Flip • Spot Profile • Interactive)")

with st.sidebar:
    st.subheader("Inputs")
    with st.form("controls"):
        ticker = st.text_input("Ticker", value="NBIS").strip().upper()
        r = st.number_input("Risk-free rate (annual, decimal)", value=0.045, step=0.005, format="%.3f")
        q = st.number_input("Dividend yield (annual, decimal)", value=0.000, step=0.005, format="%.3f")

        st.divider()
        st.subheader("Multi-expiration")
        n_exp = st.slider("How many expirations to include", min_value=1, max_value=12, value=4)

        st.divider()
        st.subheader("Spot profile")
        pct_range = st.slider("Spot range (+/- %)", min_value=5, max_value=60, value=20, step=5)
        n_points = st.slider("Spot grid points", min_value=51, max_value=301, value=151, step=50)

        st.divider()
        st.subheader("Filters")
        min_oi = st.number_input("Min open interest (per contract)", value=10, step=10)
        strike_window = st.slider("Strike window around spot (+/- %)", min_value=10, max_value=200, value=50, step=10)

        submitted = st.form_submit_button("Load options")

    st.caption(
        "Tip: Use the **Load options** button after changing inputs. "
        "This prevents Streamlit reruns from repeatedly hitting Yahoo."
    )

if not ticker:
    st.stop()

if not submitted:
    st.info("Set inputs, then click **Load options** in the sidebar.")
    st.stop()

# ---------------------------
# Fetch expirations + chains (resilient)
# ---------------------------
with st.spinner("Fetching expirations..."):
    expirations = get_expirations_resilient(ticker)

exp_list = list(expirations)[: int(n_exp)]

with st.spinner("Fetching option chain..."):
    spot, opt_df = load_spot_and_chains_resilient(ticker, exp_list)

if not np.isfinite(spot):
    st.error("Could not determine spot price from Yahoo.")
    st.stop()

# ---------------------------
# Apply filters
# ---------------------------
lower_strike = spot * (1.0 - strike_window / 100.0)
upper_strike = spot * (1.0 + strike_window / 100.0)

df = opt_df.copy()
df = df[df["openInterest"] >= float(min_oi)]
df = df[(df["strike"] >= lower_strike) & (df["strike"] <= upper_strike)]

if df.empty:
    st.warning("No options left after filters. Lower min OI or widen strike window.")
    st.stop()

# ---------------------------
# Compute spot profile + per-strike (features 1–4)
# ---------------------------
lo = spot * (1.0 - pct_range / 100.0)
hi = spot * (1.0 + pct_range / 100.0)
S_grid = np.linspace(lo, hi, int(n_points))

total_gex_profile = np.zeros_like(S_grid, dtype=float)
per_strike_rows = []

for exp in exp_list:
    T = time_to_expiration_years(exp)
    dfe = df[df["expiration"] == exp].copy()
    if dfe.empty:
        continue

    K = dfe["strike"].to_numpy(dtype=float)
    iv = dfe["impliedVolatility"].to_numpy(dtype=float)
    oi = dfe["openInterest"].to_numpy(dtype=float)
    sign = np.where(dfe["type"].to_numpy() == "call", 1.0, -1.0)

    gamma_mat = bs_gamma_matrix(S_grid, K, T, float(r), float(q), iv)
    total_gex_profile += gex_total_over_spot(gamma_mat, S_grid, oi, sign)

    gamma_spot = bs_gamma_matrix(np.array([spot]), K, T, float(r), float(q), iv).reshape(-1)
    gex_spot = gamma_spot * oi * 100.0 * (spot**2) * sign
    per_strike_rows.append(pd.DataFrame({"strike": K, "net_gex": gex_spot}))

if not per_strike_rows:
    st.warning("No expirations contributed after filtering.")
    st.stop()

per_strike = (
    pd.concat(per_strike_rows, ignore_index=True)
    .groupby("strike", as_index=False)["net_gex"].sum()
    .sort_values("strike")
)

gamma_flip = find_gamma_flip(S_grid, total_gex_profile)
spot_gex = float(np.interp(spot, S_grid, total_gex_profile))

# ---------------------------
# KPIs
# ---------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Spot", f"{spot:,.2f}")
kpi2.metric("Expirations included", f"{len(exp_list)}")
kpi3.metric("Total Net GEX @ Spot (approx)", f"{spot_gex:,.0f}")
kpi4.metric("Gamma Flip (approx)", "—" if gamma_flip is None else f"{gamma_flip:,.2f}")

# ---------------------------
# Charts (interactive)
# ---------------------------
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Total Net GEX Profile Across Spot")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_grid, y=total_gex_profile, mode="lines", name="Total Net GEX"))
    fig.add_vline(x=spot, line_dash="dash")
    if gamma_flip is not None:
        fig.add_vline(x=gamma_flip, line_dash="dot")

    fig.update_layout(
        xaxis_title="Spot price",
        yaxis_title="Total Net GEX (scaled units)",
        hovermode="x unified",
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Net GEX by Strike (at current spot)")
    top_n = 20
    top = per_strike.assign(abs=np.abs(per_strike["net_gex"])).sort_values("abs", ascending=False).head(top_n)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=top["strike"], y=top["net_gex"], name="Net GEX"))
    fig2.add_vline(x=spot, line_dash="dash")
    fig2.update_layout(
        xaxis_title="Strike",
        yaxis_title="Net GEX (scaled units)",
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# Data download / inspect
# ---------------------------
st.divider()
with st.expander("Download / inspect data"):
    st.write("Filtered option rows used (first 200):")
    st.dataframe(df.head(200), use_container_width=True)

    st.write("Per-strike net GEX at spot (aggregated across included expirations):")
    st.dataframe(per_strike, use_container_width=True)

    csv = per_strike.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download per-strike GEX CSV",
        data=csv,
        file_name=f"{ticker}_gex_by_strike.csv",
        mime="text/csv",
    )

st.caption(
    "Note: This is a free-data approximation using Yahoo option chain OI + IV. "
    "Calls treated as +GEX and puts as -GEX by convention (proxy for positioning)."
)
