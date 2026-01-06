import math
import time
import numpy as np
import pandas as pd
import yfinance as yf
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException
from scipy.stats import norm

app = FastAPI(title="Free GEX API")

# Simple in-memory caches (per service instance)
exp_cache = TTLCache(maxsize=500, ttl=60 * 60)      # 1 hour
chain_cache = TTLCache(maxsize=200, ttl=15 * 60)    # 15 min

def is_rate_limit(e: Exception) -> bool:
    msg = str(e).lower()
    return "too many requests" in msg or "rate limit" in msg or "429" in msg

def backoff(attempt: int):
    waits = [0.0, 0.8, 1.6, 3.2, 6.0]
    time.sleep(waits[min(attempt, len(waits)-1)])

def get_expirations(ticker: str):
    if ticker in exp_cache:
        return exp_cache[ticker]
    last = None
    for a in range(5):
        try:
            backoff(a)
            exps = list(yf.Ticker(ticker).options)
            if not exps:
                raise RuntimeError("No expirations returned.")
            exp_cache[ticker] = exps
            return exps
        except Exception as e:
            last = e
            if not is_rate_limit(e):
                break
    raise last

def load_chain(ticker: str, exps: list[str]):
    key = (ticker, tuple(exps))
    if key in chain_cache:
        return chain_cache[key]

    tk = yf.Ticker(ticker)
    spot = float(tk.fast_info.get("last_price") or tk.info.get("regularMarketPrice") or np.nan)
    if not np.isfinite(spot):
        raise RuntimeError("Could not determine spot.")

    rows = []
    for exp in exps:
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

    chain_cache[key] = (spot, df)
    return spot, df

def time_to_expiration_years(exp: str) -> float:
    exp_dt = pd.to_datetime(exp, utc=True)
    now = pd.Timestamp.utcnow()
    return float(max((exp_dt - now).total_seconds(), 0.0) / (365.0 * 24.0 * 3600.0))

def bs_gamma_matrix(S_grid, K, T, r, q, iv):
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

def gex_total_over_spot(gamma, S_grid, oi, sign):
    S2 = (S_grid.reshape(-1, 1) ** 2)
    return (gamma * oi.reshape(1, -1) * 100.0 * S2 * sign.reshape(1, -1)).sum(axis=1)

def find_gamma_flip(S_grid, total_gex):
    sign = np.sign(total_gex)
    idx = np.where(sign[:-1] * sign[1:] < 0)[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    x0, y0 = S_grid[i], total_gex[i]
    x1, y1 = S_grid[i+1], total_gex[i+1]
    if y1 == y0:
        return float((x0 + x1) / 2.0)
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/expirations/{ticker}")
def expirations(ticker: str):
    try:
        return {"ticker": ticker.upper(), "expirations": get_expirations(ticker.upper())}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/gex/{ticker}")
def gex(
    ticker: str,
    n_exp: int = 4,
    r: float = 0.045,
    q: float = 0.0,
    pct_range: int = 20,
    n_points: int = 151,
    min_oi: int = 10,
    strike_window: int = 50,
):
    t = ticker.upper()
    try:
        exps = get_expirations(t)[:max(1, min(n_exp, 12))]
        spot, opt_df = load_chain(t, exps)

        lower_strike = spot * (1.0 - strike_window / 100.0)
        upper_strike = spot * (1.0 + strike_window / 100.0)
        df = opt_df[(opt_df["openInterest"] >= float(min_oi)) &
                    (opt_df["strike"] >= lower_strike) &
                    (opt_df["strike"] <= upper_strike)].copy()

        if df.empty:
            return {"ticker": t, "spot": spot, "expirations": exps, "message": "No options after filters."}

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

            gamma_mat = bs_gamma_matrix(S_grid, K, T, r, q, iv)
            total_profile += gex_total_over_spot(gamma_mat, S_grid, oi, sign)

            gamma_spot = bs_gamma_matrix(np.array([spot]), K, T, r, q, iv).reshape(-1)
            gex_spot = gamma_spot * oi * 100.0 * (spot**2) * sign
            per_strike_rows.append(pd.DataFrame({"strike": K, "net_gex": gex_spot}))

        per_strike = (pd.concat(per_strike_rows)
                      .groupby("strike", as_index=False)["net_gex"].sum()
                      .sort_values("strike"))

        gamma_flip = find_gamma_flip(S_grid, total_profile)
        spot_gex = float(np.interp(spot, S_grid, total_profile))

        return {
            "ticker": t,
            "spot": spot,
            "expirations": exps,
            "spot_gex": spot_gex,
            "gamma_flip": gamma_flip,
            "profile": {"spot_grid": S_grid.tolist(), "total_net_gex": total_profile.tolist()},
            "gex_by_strike": per_strike.to_dict(orient="records"),
        }
    except Exception as e:
        code = 503 if is_rate_limit(e) else 500
        raise HTTPException(status_code=code, detail=str(e))
