#!/usr/bin/env python3
"""
BTC/USDT LIVE MARKET SIMULATION
===============================
Real-time practice terminal.
Predict UP / DOWN / FLAT on live BTC/USDT candles, wait for
reality to settle on its own, and track your accuracy honestly.

- Data pulled live from Binance (multi-host fallback).
- Predictions anchored to the last CLOSED candle, so the
  reference price does not drift while you wait.
- Settlement uses Binance server time, not your laptop's clock.
- Auto-refresh follows the live market.

This is a TRAINING tool, NOT a trading signal.
Accuracy here does NOT account for fees, spread, slippage, or payout.

Run:
    streamlit run btc_live_sim_streamlit.py
"""

import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="BTC/USDT Live Simulation",
    page_icon="🟢",
    layout="wide",
)

MINUTE_MS = 60_000

BINANCE_HOSTS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
    "https://api.binance.us",
]

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "qav", "num_trades",
    "taker_buy_vol", "taker_quote_vol", "ignore",
]


# =========================================================
# TIME HELPERS
# =========================================================

def utc_str(ms):
    return datetime.fromtimestamp(
        ms / 1000, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")


def utc_short(ms):
    return datetime.fromtimestamp(
        ms / 1000, tz=timezone.utc
    ).strftime("%H:%M:%S")


def interval_to_ms(interval):
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    return int(interval[:-1]) * units[interval[-1]]


# =========================================================
# BINANCE API (with multi-host fallback)
# =========================================================

def fetch_klines(symbol, interval, limit):
    """
    Fetch raw klines, INCLUDING the currently-forming candle.
    Returns: (dataframe, host_used)
    """
    errors = []

    for host in BINANCE_HOSTS:
        try:
            response = requests.get(
                f"{host}/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit,
                },
                timeout=10,
            )
            response.raise_for_status()

            raw = pd.DataFrame(
                response.json(), columns=KLINE_COLUMNS
            )

            for col in ["open_time", "close_time"]:
                raw[col] = pd.to_numeric(
                    raw[col], errors="raise"
                ).astype("int64")

            for col in [
                "open", "high", "low", "close",
                "volume", "taker_buy_vol",
            ]:
                raw[col] = pd.to_numeric(
                    raw[col], errors="coerce"
                ).astype(float)

            if not np.isfinite(
                raw[["open", "high", "low", "close"]].to_numpy()
            ).all():
                raise ValueError("Non-finite price in klines.")

            raw["dt"] = pd.to_datetime(
                raw["open_time"], unit="ms", utc=True
            )

            return raw, host

        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{host} → {exc}")

    raise RuntimeError(
        "All Binance hosts failed:\n" + "\n".join(errors)
    )


def fetch_server_time():
    """Binance server time in ms. Falls back to local clock."""
    for host in BINANCE_HOSTS:
        try:
            response = requests.get(
                f"{host}/api/v3/time", timeout=5
            )
            response.raise_for_status()
            return int(response.json()["serverTime"])
        except (
            requests.RequestException,
            KeyError,
            ValueError,
        ):
            continue
    return int(time.time() * 1000)


# =========================================================
# SESSION STATE
# =========================================================

def init_state():
    defaults = {
        "predictions": [],
        "last_render_ms": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# =========================================================
# PREDICTION LOGIC
# =========================================================

def create_prediction(closed_df, interval_ms, horizon, predicted, threshold):
    """
    Anchor a prediction to the last CLOSED candle.
    Reference price = close of the last closed candle.
    Target = close of the candle at ref_open + horizon * interval.
    """
    last = closed_df.iloc[-1]
    ref_open = int(last["open_time"])
    ref_close_time = ref_open + interval_ms
    ref_price = float(last["close"])

    target_open = ref_open + horizon * interval_ms
    target_close_time = target_open + interval_ms

    return {
        "id": uuid.uuid4().hex[:8],
        "created_at_ms": int(time.time() * 1000),
        "ref_open": ref_open,
        "ref_close_time": ref_close_time,
        "ref_price": ref_price,
        "horizon": horizon,
        "interval_ms": interval_ms,
        "target_open": target_open,
        "target_close_time": target_close_time,
        "predicted": predicted,
        "threshold": threshold,
        "settled": False,
        "target_price": None,
        "actual": None,
        "change_pct": None,
        "correct": None,
        "settled_at_ms": None,
    }


def settle_predictions(predictions, closed_df, server_now_ms):
    """
    Try to settle pending predictions using data we already have.
    Returns: (number settled this cycle, list of just-settled IDs)
    """
    lookup = {
        int(row.open_time): float(row.close)
        for row in closed_df.itertuples()
    }
    settled_count = 0
    just_settled = []

    for p in predictions:
        if p["settled"]:
            continue
        # 2-second buffer to avoid racing a candle that closed
        # microseconds ago
        if server_now_ms < p["target_close_time"] + 2000:
            continue

        price = lookup.get(p["target_open"])
        if price is None:
            # Target candle not in visible window (offline too long);
            # will be picked up when it appears in range.
            continue

        change = price / p["ref_price"] - 1
        if change > p["threshold"]:
            actual = "UP"
        elif change < -p["threshold"]:
            actual = "DOWN"
        else:
            actual = "FLAT"

        p["target_price"] = price
        p["actual"] = actual
        p["change_pct"] = change * 100
        p["correct"] = p["predicted"] == actual
        p["settled"] = True
        p["settled_at_ms"] = server_now_ms
        settled_count += 1
        just_settled.append(p["id"])

    return settled_count, just_settled


# =========================================================
# STATISTICS
# =========================================================

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (
        z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        / denom
    )
    return center - margin, center + margin


# =========================================================
# CHART
# =========================================================

def build_live_chart(closed, forming, pending, show_ma=True):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.02,
    )

    # Closed candles
    fig.add_trace(go.Candlestick(
        x=closed["dt"],
        open=closed["open"], high=closed["high"],
        low=closed["low"], close=closed["close"],
        name="BTC/USDT",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Forming candle (live, semi-transparent)
    if forming is not None and len(forming):
        fig.add_trace(go.Candlestick(
            x=forming["dt"],
            open=forming["open"], high=forming["high"],
            low=forming["low"], close=forming["close"],
            name="Forming (live)",
            increasing_line_color="rgba(38,166,154,0.55)",
            decreasing_line_color="rgba(239,83,80,0.55)",
            increasing_fillcolor="rgba(38,166,154,0.35)",
            decreasing_fillcolor="rgba(239,83,80,0.35)",
        ), row=1, col=1)

    # Moving averages on closed candles only
    if show_ma and len(closed) >= 20:
        fig.add_trace(go.Scatter(
            x=closed["dt"],
            y=closed["close"].rolling(20).mean(),
            mode="lines", name="MA20",
            line=dict(width=1, color="#ffb74d"),
        ), row=1, col=1)

        if len(closed) >= 50:
            fig.add_trace(go.Scatter(
                x=closed["dt"],
                y=closed["close"].rolling(50).mean(),
                mode="lines", name="MA50",
                line=dict(width=1, color="#42a5f5"),
            ), row=1, col=1)

    # Anchor lines for pending predictions
    colors = {
        "UP": "#26a69a",
        "DOWN": "#ef5350",
        "FLAT": "#9e9e9e",
    }
    for p in pending:
        fig.add_hline(
            y=p["ref_price"],
            line_dash="dash",
            line_color=colors.get(p["predicted"], "gray"),
            annotation_text=(
                f"{p['predicted']} @ {p['ref_price']:,.2f}"
            ),
            annotation_position="right",
            row=1, col=1,
        )

    # Volume
    vol_colors = np.where(
        closed["close"] >= closed["open"],
        "#26a69a", "#ef5350",
    )
    fig.add_trace(go.Bar(
        x=closed["dt"], y=closed["volume"],
        marker_color=vol_colors, name="Volume",
        showlegend=False,
    ), row=2, col=1)

    if forming is not None and len(forming):
        f_colors = np.where(
            forming["close"] >= forming["open"],
            "rgba(38,166,154,0.55)", "rgba(239,83,80,0.55)",
        )
        fig.add_trace(go.Bar(
            x=forming["dt"], y=forming["volume"],
            marker_color=f_colors, showlegend=False,
        ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=560,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.04),
    )
    fig.update_yaxes(title_text="USDT", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)

    return fig


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🟢 BTC/USDT Live Sim")
st.sidebar.caption(
    "Practice predictions against the live market."
)

symbol = st.sidebar.text_input(
    "Symbol", value="BTCUSDT"
).upper()

interval = st.sidebar.selectbox(
    "Candle interval", ["1m", "5m", "15m"], index=0
)
interval_ms = interval_to_ms(interval)

chart_window = st.sidebar.slider(
    "Candles shown on chart", 50, 300, 120, step=10
)

horizon = st.sidebar.selectbox(
    "Prediction horizon (candles)", [1, 3, 5, 15], index=2
)

threshold_pct = st.sidebar.number_input(
    "FLAT threshold (%)",
    min_value=0.0, max_value=2.0,
    value=0.03, step=0.01,
    format="%.2f",
)
threshold = threshold_pct / 100

refresh_seconds = st.sidebar.slider(
    "Auto-refresh (seconds)", 5, 60, 10, step=5
)

show_ma = st.sidebar.checkbox("Show MA20 / MA50", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Actions")

if st.sidebar.button("🧹 Clear pending predictions"):
    st.session_state.predictions = [
        p for p in st.session_state.predictions if p["settled"]
    ]
    st.rerun()

if st.sidebar.button("🗑️ Clear all history"):
    st.session_state.predictions = []
    st.rerun()


# =========================================================
# AUTO-REFRESH
# =========================================================

st_autorefresh(
    interval=refresh_seconds * 1000,
    key="live_autorefresh",
)


# =========================================================
# FETCH LIVE DATA
# =========================================================

try:
    raw, host_used = fetch_klines(
        symbol, interval, limit=999
    )
    server_now_ms = fetch_server_time()
except RuntimeError as exc:
    st.error(f"Data fetch failed: {exc}")
    st.stop()

is_closed = raw["close_time"] < server_now_ms

closed = (
    raw.loc[is_closed]
    .reset_index(drop=True)
)
forming = (
    raw.loc[~is_closed]
    .reset_index(drop=True)
)

if len(closed) < 10:
    st.error("Not enough closed candles yet.")
    st.stop()

# Settle anything that has matured
settled_now, just_settled_ids = settle_predictions(
    st.session_state.predictions,
    closed,
    server_now_ms,
)


# =========================================================
# HEADER
# =========================================================

live_price = float(forming.iloc[-1]["close"]) if len(forming) else float(closed.iloc[-1]["close"])
last_closed = closed.iloc[-1]
next_close_ms = int(forming.iloc[-1]["close_time"]) + 1 if len(forming) else int(last_closed["close_time"]) + 1
seconds_to_close = max(0, (next_close_ms - server_now_ms) / 1000)

pending_count = sum(
    1 for p in st.session_state.predictions if not p["settled"]
)
settled_count = sum(
    1 for p in st.session_state.predictions if p["settled"]
)

h1, h2, h3, h4 = st.columns(4)
h1.metric(
    "Server time",
    utc_short(server_now_ms) + " UTC",
)
h2.metric(
    "Live price",
    f"${live_price:,.2f}",
)
h3.metric(
    "Next candle close",
    f"{seconds_to_close:.0f}s",
)
h4.metric(
    "Pending / Settled",
    f"{pending_count} / {settled_count}",
)

st.caption(
    f"Data host: `{host_used}` | "
    f"Interval: `{interval}` | "
    f"Auto-refresh every {refresh_seconds}s"
)


# =========================================================
# CHART
# =========================================================

chart_start = max(0, len(closed) - chart_window)
visible_closed = closed.iloc[chart_start:]

pending = [
    p for p in st.session_state.predictions if not p["settled"]
]
recent_settled = [
    p for p in st.session_state.predictions if p["settled"]
][-5:]

fig = build_live_chart(
    visible_closed,
    forming,
    pending,
    show_ma=show_ma,
)
st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PREDICTION PANEL
# =========================================================

st.subheader("🎯 Make a prediction")

st.write(
    f"Anchor: **close of the last closed candle** "
    f"(`{utc_str(int(last_closed['open_time']) + interval_ms)}`) "
    f"@ **${last_closed['close']:,.2f}**\n\n"
    f"Horizon: **{horizon} × {interval}**. "
    f"FLAT if change within ±{threshold_pct:.2f}%."
)

b1, b2, b3 = st.columns(3)
pred_up = b1.button("📈 UP", use_container_width=True)
pred_flat = b2.button("➡️ FLAT", use_container_width=True)
pred_down = b3.button("📉 DOWN", use_container_width=True)

clicked = (
    "UP" if pred_up else
    "FLAT" if pred_flat else
    "DOWN" if pred_down else None
)

if clicked:
    # Avoid two predictions on the same reference candle
    last_ref_open = int(last_closed["open_time"])
    already = any(
        p["ref_open"] == last_ref_open
        for p in st.session_state.predictions
    )
    if already:
        st.warning(
            "You already predicted on this reference candle."
        )
    else:
        new_pred = create_prediction(
            closed, interval_ms, horizon, clicked, threshold
        )
        st.session_state.predictions.append(new_pred)
        st.toast(
            f"Locked: {clicked} @ ${new_pred['ref_price']:,.2f}",
            icon="✅",
        )
        st.rerun()


# =========================================================
# PENDING PREDICTIONS
# =========================================================

st.markdown("---")
st.subheader("⏳ Pending predictions")

pending = [
    p for p in st.session_state.predictions if not p["settled"]
]

if not pending:
    st.caption(
        "No pending predictions. Use the buttons above to anchor one."
    )
else:
    pending_sorted = sorted(
        pending, key=lambda p: p["target_close_time"]
    )
    rows = []
    for p in pending_sorted:
        remaining = max(
            0, (p["target_close_time"] - server_now_ms) / 1000
        )
        rows.append({
            "Ref time": utc_short(p["ref_close_time"]),
            "Horizon": p["horizon"],
            "Predicted": p["predicted"],
            "Reference": f"${p['ref_price']:,.2f}",
            "Settles in": f"{remaining:.0f}s",
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# SETTLED HISTORY + STATS
# =========================================================

st.markdown("---")
st.subheader("📈 Settled history and statistics")

settled = [
    p for p in st.session_state.predictions if p["settled"]
]

if not settled:
    st.caption(
        "No settled predictions yet. "
        "Wait for the horizon to elapse."
    )
else:
    n = len(settled)
    k = sum(1 for p in settled if p["correct"])
    acc = k / n
    lo, hi = wilson_ci(k, n)

    # Majority baseline
    from collections import Counter
    actual_counts = Counter(p["actual"] for p in settled)
    majority_baseline = max(actual_counts.values()) / n

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Settled", n)
    s2.metric("Your accuracy", f"{acc:.1%}")
    s3.metric("95% CI", f"{lo:.1%} – {hi:.1%}")
    s4.metric(
        "Majority baseline",
        f"{majority_baseline:.1%}",
        help=(
            "Accuracy if you always guessed the most frequent "
            "outcome so far."
        ),
    )

    if n < 30:
        st.caption(
            "⚠️ Sample still small (<30). Do not draw conclusions — "
            "confidence interval is very wide."
        )
    elif lo > majority_baseline:
        st.success(
            "Your CI lower bound is above the majority baseline. "
            "Early positive sign — but still does not account for "
            "fees/slippage/payout if used for trading."
        )
    else:
        st.caption(
            "Not yet proven to beat the simple baseline. "
            "That is normal — and that is the point of the drill."
        )

    # Table of settled predictions, newest first
    rows = []
    for p in reversed(settled):
        rows.append({
            "Ref time": utc_str(p["ref_close_time"]),
            "Predicted": p["predicted"],
            "Reference": f"${p['ref_price']:,.2f}",
            "Target": f"${p['target_price']:,.2f}",
            "Change": f"{p['change_pct']:+.3f}%",
            "Actual": p["actual"],
            "Result": "✅" if p["correct"] else "❌",
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    # Accuracy by predicted direction
    st.write("**Accuracy by predicted direction:**")
    breakdown = {}
    for p in settled:
        d = breakdown.setdefault(
            p["predicted"], {"count": 0, "correct": 0}
        )
        d["count"] += 1
        if p["correct"]:
            d["correct"] += 1

    bd_rows = []
    for direction, d in breakdown.items():
        bd_rows.append({
            "Direction": direction,
            "Count": d["count"],
            "Correct": d["correct"],
            "Accuracy": f"{d['correct'] / d['count']:.1%}",
        })
    st.dataframe(
        pd.DataFrame(bd_rows),
        use_container_width=True,
        hide_index=True,
    )

    # Download
    export_rows = []
    for p in settled:
        export_rows.append({
            "ref_close_utc": utc_str(p["ref_close_time"]),
            "created_at_ms": p["created_at_ms"],
            "settled_at_ms": p["settled_at_ms"],
            "horizon": p["horizon"],
            "interval_ms": p["interval_ms"],
            "ref_price": p["ref_price"],
            "target_price": p["target_price"],
            "change_pct": p["change_pct"],
            "predicted": p["predicted"],
            "actual": p["actual"],
            "correct": p["correct"],
            "threshold": p["threshold"],
        })
    st.download_button(
        "💾 Download settled history (CSV)",
        pd.DataFrame(export_rows).to_csv(index=False),
        "live_prediction_history.csv",
        "text/csv",
    )


# =========================================================
# FOOTER
# =========================================================

st.caption(
    "Live data from Binance public API. "
    "This tool is for practicing price-action reading — "
    "not a signal, and it does not promise profit. "
    "Settlement uses close-to-close, so it is not the same as "
    "an executable entry/exit price."
)
