# app.py — Dash GEX webapp (no Streamlit)
# Browser charts + button-trigger fetch + caching + retry/backoff for Yahoo rate limits

import math
import time
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from cachetools import TTLCache

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update


# ----------------------------
# Caches (in-memory per instance)
# ----------------------------
EXP_CACHE = TTLCache(maxsize=500, ttl=60 * 60)          # 1 hour
CHAIN_CACHE = TTLCache(maxsize=200, ttl=15 * 60)        # 15 min


# ----------------------------
# Helpers: rate limit handling
# ----------------------------
def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("too many requests" in msg) or ("rate limit" in msg) or ("429" in msg)

def _backoff(attempt: int):
    waits = [0.0, 0.8, 1.6, 3.2, 6.0]
    time.sleep(waits[min(attempt, len(waits) - 1)])


# ----------------------------
# Options data
# ----------------------------
def get_expirations(ticker: str) -> list[str]:
    t = ticker.upper().strip()
    if t in EXP_CACHE:
        return EXP_CACHE[t]

    last = None
    for a in range(5):
        try:
            _backoff(a)
            exps = list(yf.Ticker(t).options)
            if not exps:
                raise RuntimeError("No expirations returned.")
            EXP_CACHE[t] = exps
            return exps
        except Exception as e:
            last = e
            if not _is_rate_limit_error(e):
                break
    raise last

def load_spot_and_chain(ticker: str, expirations: list[str]) -> tuple[float, pd.DataFrame]:
    t = ticker.upper().strip()
    key = (t, tuple(expirations))
    if key in CHAIN_CACHE:
        return CHAIN_CACHE[key]

    tk = yf.Ticker(t)

    spot = float(tk.fast_info.get("last_price") or tk.info.get("regularMarketPrice") or np.nan)
    if not np.isfinite(spot):
        raise RuntimeError("Could not determine spot price.")

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

    CHAIN_CACHE[key] = (spot, df)
    return spot, df

def load_spot_and_chain_resilient(ticker: str, expirations: list[str]) -> tuple[float, pd.DataFrame]:
    last = None
    for a in range(5):
        try:
            _backoff(a)
            return load_spot_and_chain(ticker, expirations)
        except Exception as e:
            last = e
            if not _is_rate_limit_error(e):
                break
    raise last


# ----------------------------
# Math: Black-Scholes gamma + GEX
# ----------------------------
def time_to_expiration_years(exp: str) -> float:
    exp_dt = pd.to_datetime(exp, utc=True)
    now = pd.Timestamp.utcnow()
    return float(max((exp_dt - now).total_seconds(), 0.0) / (365.0 * 24.0 * 3600.0))

def bs_gamma_matrix(S_grid: np.ndarray, K: np.ndarray, T: float, r: float, q: float, iv: np.ndarray) -> np.ndarray:
    if T <= 0:
        return np.zeros((S_grid.size, K.size), dtype=float)

    S = S_grid.reshape(-1, 1)
    K2 = K.reshape(1, -1)
    iv2 = iv.reshape(1, -1)

    eps = 1e-12
    iv2 = np.where(iv2 > eps, iv2, np.nan)

    sqrtT = math.sqrt(T)
    d1 = (np.log(S / K2) + (r - q + 0.5 * iv2**2) * T) / (iv2 * sqrtT)

    gamma = norm.pdf(d1) / (S * iv2 * sqrtT)
    return np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)

def gex_total_over_spot(gamma: np.ndarray, S_grid: np.ndarray, oi: np.ndarray, sign: np.ndarray) -> np.ndarray:
    S2 = (S_grid.reshape(-1, 1) ** 2)
    return (gamma * oi.reshape(1, -1) * 100.0 * S2 * sign.reshape(1, -1)).sum(axis=1)

def find_gamma_flip(S_grid: np.ndarray, total_gex: np.ndarray):
    sgn = np.sign(total_gex)
    idx = np.where(sgn[:-1] * sgn[1:] < 0)[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    x0, y0 = float(S_grid[i]), float(total_gex[i])
    x1, y1 = float(S_grid[i + 1]), float(total_gex[i + 1])
    if y1 == y0:
        return (x0 + x1) / 2.0
    return x0 - y0 * (x1 - x0) / (y1 - y0)


# ----------------------------
# Core compute
# ----------------------------
def compute_gex(
    ticker: str,
    n_exp: int,
    r: float,
    q: float,
    pct_range: int,
    n_points: int,
    min_oi: int,
    strike_window: int,
    top_n_strikes: int = 20,
):
    t = ticker.upper().strip()
    exps = get_expirations(t)[: max(1, min(int(n_exp), 12))]

    spot, opt_df = load_spot_and_chain_resilient(t, exps)

    lower_strike = spot * (1.0 - strike_window / 100.0)
    upper_strike = spot * (1.0 + strike_window / 100.0)

    df = opt_df.copy()
    df = df[df["openInterest"] >= float(min_oi)]
    df = df[(df["strike"] >= lower_strike) & (df["strike"] <= upper_strike)]

    if df.empty:
        return {
            "ticker": t,
            "spot": spot,
            "expirations": exps,
            "spot_gex": 0.0,
            "gamma_flip": None,
            "profile": ([], []),
            "top_by_strike": pd.DataFrame(columns=["strike", "net_gex"]),
            "message": "No options left after filters (lower min OI or widen strike window).",
        }

    lo = spot * (1.0 - pct_range / 100.0)
    hi = spot * (1.0 + pct_range / 100.0)
    S_grid = np.linspace(lo, hi, int(n_points))

    total_profile = np.zeros_like(S_grid, dtype=float)
    per_strike_rows = []

    for exp in exps:
        T = time_to_expiration_years(exp)
        dfe = df[df["expiration"] == exp]
        if dfe.empty:
            continue

        K = dfe["strike"].to_numpy(float)
        iv = dfe["impliedVolatility"].to_numpy(float)
        oi = dfe["openInterest"].to_numpy(float)
        sign = np.where(dfe["type"].to_numpy() == "call", 1.0, -1.0)

        gamma_mat = bs_gamma_matrix(S_grid, K, T, float(r), float(q), iv)
        total_profile += gex_total_over_spot(gamma_mat, S_grid, oi, sign)

        gamma_spot = bs_gamma_matrix(np.array([spot]), K, T, float(r), float(q), iv).reshape(-1)
        gex_spot = gamma_spot * oi * 100.0 * (spot**2) * sign
        per_strike_rows.append(pd.DataFrame({"strike": K, "net_gex": gex_spot}))

    per_strike = (
        pd.concat(per_strike_rows, ignore_index=True)
        .groupby("strike", as_index=False)["net_gex"].sum()
        .sort_values("strike")
    )

    top = (
        per_strike.assign(abs=np.abs(per_strike["net_gex"]))
        .sort_values("abs", ascending=False)
        .head(int(top_n_strikes))
        .drop(columns=["abs"])
        .sort_values("strike")
    )

    gamma_flip = find_gamma_flip(S_grid, total_profile)
    spot_gex = float(np.interp(spot, S_grid, total_profile))

    return {
        "ticker": t,
        "spot": spot,
        "expirations": exps,
        "spot_gex": spot_gex,
        "gamma_flip": gamma_flip,
        "profile": (S_grid, total_profile),
        "top_by_strike": top,
        "message": "",
    }


# ----------------------------
# Dash app
# ----------------------------
app = Dash(__name__)
server = app.server  # important for Render

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "18px", "fontFamily": "system-ui"},
    children=[
        html.H2("GEX Dashboard (Dash + yfinance)"),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1.2fr 1fr", "gap": "14px"},
            children=[
                html.Div(
                    style={"border": "1px solid #ddd", "borderRadius": "12px", "padding": "14px"},
                    children=[
                        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "10px"}, children=[
                            html.Div([html.Label("Ticker"), dcc.Input(id="ticker", value="NBIS", type="text", style={"width": "100%"})]),
                            html.Div([html.Label("Expirations (N)"), dcc.Input(id="n_exp", value=4, type="number", min=1, max=12, style={"width": "100%"})]),
                            html.Div([html.Label("Min OI"), dcc.Input(id="min_oi", value=10, type="number", min=0, step=10, style={"width": "100%"})]),
                        ]),
                        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "10px", "marginTop": "10px"}, children=[
                            html.Div([html.Label("r (risk-free)"), dcc.Input(id="r", value=0.045, type="number", step=0.005, style={"width": "100%"})]),
                            html.Div([html.Label("q (div yield)"), dcc.Input(id="q", value=0.0, type="number", step=0.005, style={"width": "100%"})]),
                            html.Div([html.Label("Strike window ±%"), dcc.Input(id="strike_window", value=50, type="number", min=10, max=200, step=10, style={"width": "100%"})]),
                        ]),
                        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "10px", "marginTop": "10px"}, children=[
                            html.Div([html.Label("Spot range ±%"), dcc.Input(id="pct_range", value=20, type="number", min=5, max=60, step=5, style={"width": "100%"})]),
                            html.Div([html.Label("Grid points"), dcc.Input(id="n_points", value=151, type="number", min=51, max=301, step=50, style={"width": "100%"})]),
                            html.Div([html.Label("Top strikes"), dcc.Input(id="top_n", value=20, type="number", min=5, max=50, step=5, style={"width": "100%"})]),
                        ]),
                        html.Button("Load / Refresh", id="load_btn", style={"marginTop": "12px", "padding": "10px 12px", "borderRadius": "10px"}),
                        html.Div(id="status", style={"marginTop": "10px", "color": "#444"}),
                        html.Div(style={"marginTop": "8px", "fontSize": "12px", "color": "#666"},
                                 children="Note: Yahoo can still rate-limit sometimes. This app caches results and only fetches on button click."),
                    ],
                ),

                html.Div(
                    style={"border": "1px solid #ddd", "borderRadius": "12px", "padding": "14px"},
                    children=[
                        html.H4("Key stats"),
                        html.Div(id="kpis")
                    ],
                ),
            ],
        ),

        html.Div(style={"height": "14px"}),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1.3fr 1fr", "gap": "14px"},
            children=[
                html.Div(style={"border": "1px solid #ddd", "borderRadius": "12px", "padding": "10px"},
                         children=[dcc.Graph(id="profile_chart")]),
                html.Div(style={"border": "1px solid #ddd", "borderRadius": "12px", "padding": "10px"},
                         children=[dcc.Graph(id="strike_chart")]),
            ],
        ),
    ],
)


@app.callback(
    Output("profile_chart", "figure"),
    Output("strike_chart", "figure"),
    Output("kpis", "children"),
    Output("status", "children"),
    Input("load_btn", "n_clicks"),
    State("ticker", "value"),
    State("n_exp", "value"),
    State("r", "value"),
    State("q", "value"),
    State("pct_range", "value"),
    State("n_points", "value"),
    State("min_oi", "value"),
    State("strike_window", "value"),
    State("top_n", "value"),
    prevent_initial_call=True,
)
def on_load(_, ticker, n_exp, r, q, pct_range, n_points, min_oi, strike_window, top_n):
    try:
        res = compute_gex(
            ticker=ticker or "",
            n_exp=int(n_exp or 4),
            r=float(r or 0.045),
            q=float(q or 0.0),
            pct_range=int(pct_range or 20),
            n_points=int(n_points or 151),
            min_oi=int(min_oi or 0),
            strike_window=int(strike_window or 50),
            top_n_strikes=int(top_n or 20),
        )

        t = res["ticker"]
        spot = res["spot"]
        spot_gex = res["spot_gex"]
        gamma_flip = res["gamma_flip"]
        exps = res["expirations"]
        msg = res["message"]

        # Profile figure
        S_grid, total_profile = res["profile"]
        fig_profile = go.Figure()
        if len(S_grid) > 0:
            fig_profile.add_trace(go.Scatter(x=S_grid, y=total_profile, mode="lines", name="Total Net GEX"))
            fig_profile.add_vline(x=spot, line_dash="dash")
            if gamma_flip is not None:
                fig_profile.add_vline(x=gamma_flip, line_dash="dot")
        fig_profile.update_layout(
            title="Total Net GEX Profile Across Spot",
            xaxis_title="Spot",
            yaxis_title="Total Net GEX (scaled units)",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=45, b=10),
            height=520,
        )

        # Strike figure
        top_df = res["top_by_strike"]
        fig_strike = go.Figure()
        if not top_df.empty:
            fig_strike.add_trace(go.Bar(x=top_df["strike"], y=top_df["net_gex"], name="Net GEX"))
            fig_strike.add_vline(x=spot, line_dash="dash")
        fig_strike.update_layout(
            title="Top Net GEX by Strike (at current spot)",
            xaxis_title="Strike",
            yaxis_title="Net GEX (scaled units)",
            margin=dict(l=10, r=10, t=45, b=10),
            height=520,
        )

        kpis = html.Ul(
            style={"lineHeight": "1.8"},
            children=[
                html.Li([html.B("Ticker: "), t]),
                html.Li([html.B("Spot: "), f"{spot:,.2f}"]),
                html.Li([html.B("Expirations: "), ", ".join(exps[:4]) + (" ..." if len(exps) > 4 else "")]),
                html.Li([html.B("Total Net GEX @ Spot: "), f"{spot_gex:,.0f}"]),
                html.Li([html.B("Gamma flip: "), "—" if gamma_flip is None else f"{gamma_flip:,.2f}"]),
            ],
        )

        status = "Loaded successfully." if not msg else msg
        return fig_profile, fig_strike, kpis, status

    except Exception as e:
        if _is_rate_limit_error(e):
            status = "Yahoo rate-limited this server IP. Try again in 1–5 minutes."
        else:
            status = f"Error: {e}"

        # Keep previous charts if any; otherwise empty figures
        empty1 = go.Figure().update_layout(title="Total Net GEX Profile Across Spot", height=520)
        empty2 = go.Figure().update_layout(title="Top Net GEX by Strike (at current spot)", height=520)
        return empty1, empty2, no_update, status


if __name__ == "__main__":
    # For local dev
    app.run_server(host="0.0.0.0", port=8050, debug=True)
