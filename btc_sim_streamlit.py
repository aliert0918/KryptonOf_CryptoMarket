#!/usr/bin/env python3
"""
BTC/USDT MARKET SIMULATION LAB
==============================
Replay candle historis Binance + latihan prediksi arah.

- Data NYATA dari Binance (CSV upload atau fetch API).
- Candle masa depan disembunyikan sampai kamu kunci prediksi.
- Skoring jujur: dibandingkan baseline random & majority class.

Ini alat latihan, BUKAN sinyal trading.
Akurasi prediksi di sini TIDAK memperhitungkan fee, spread,
slippage, ataupun payout — jadi bukan simulasi profit.

Jalankan:
    streamlit run btc_sim_streamlit.py
"""

import io
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="BTC/USDT Simulation Lab",
    page_icon="📊",
    layout="wide",
)

MINUTE_MS = 60_000
BINANCE_URL = "https://api.binance.com"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "qav", "num_trades",
    "taker_buy_vol", "taker_quote_vol", "ignore",
]


# =========================================================
# UTIL DATA
# =========================================================

def validate_ohlcv(df):
    """
    Validasi dan normalisasi data OHLCV.
    Menerima open_time dalam ms, detik, atau string tanggal.
    Return: (dataframe bersih, jumlah gap)
    """
    required = ["open_time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom kurang: {missing}")

    d = df.copy()

    # Normalisasi open_time -> milidetik
    if not np.issubdtype(d["open_time"].dtype, np.number):
        d["open_time"] = (
            pd.to_datetime(d["open_time"], utc=True)
            .astype("int64") // 1_000_000
        )
    else:
        d["open_time"] = pd.to_numeric(
            d["open_time"], errors="raise"
        ).astype("int64")
        # Kalau terlihat seperti detik, konversi ke ms
        if d["open_time"].max() < 100_000_000_000:
            d["open_time"] *= 1000

    for col in ["open", "high", "low", "close", "volume"]:
        d[col] = pd.to_numeric(d[col], errors="raise").astype(float)

    if "taker_buy_vol" in d.columns:
        d["taker_buy_vol"] = pd.to_numeric(
            d["taker_buy_vol"], errors="coerce"
        ).astype(float)
    else:
        d["taker_buy_vol"] = np.nan

    d = (
        d.sort_values("open_time")
        .drop_duplicates("open_time")
        .reset_index(drop=True)
    )

    if not np.isfinite(
        d[["open", "high", "low", "close", "volume"]].to_numpy()
    ).all():
        raise ValueError("Ada nilai NaN/infinity di data harga.")

    if (d[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Ada harga <= 0.")

    if (d["high"] < d["low"]).any():
        raise ValueError("Ada candle dengan high < low.")

    if (d["volume"] < 0).any():
        raise ValueError("Ada volume negatif.")

    # Deteksi gap (peringatan saja, tidak fatal)
    diffs = d["open_time"].diff().dropna()
    if len(diffs):
        expected = diffs.median()
        gap_count = int((diffs != expected).sum())
    else:
        gap_count = 0

    # Kolom datetime untuk plotting
    d["dt"] = pd.to_datetime(d["open_time"], unit="ms", utc=True)

    return d, gap_count


@st.cache_data(ttl=600, show_spinner="Mengambil data dari Binance...")
def fetch_binance(symbol, interval, limit):
    """Ambil candle dari Binance public API (tanpa API key)."""
    response = requests.get(
        f"{BINANCE_URL}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=15,
    )
    response.raise_for_status()

    raw = pd.DataFrame(response.json(), columns=KLINE_COLUMNS)

    # Buang candle yang masih berjalan (belum close)
    now_ms = int(time.time() * 1000)
    raw["close_time"] = pd.to_numeric(raw["close_time"])
    raw = raw.loc[raw["close_time"] < now_ms]

    keep = ["open_time", "open", "high", "low", "close",
            "volume", "taker_buy_vol"]
    return validate_ohlcv(raw[keep].copy())


@st.cache_data(show_spinner="Membaca CSV...")
def parse_csv(content):
    df = pd.read_csv(io.BytesIO(content))
    return validate_ohlcv(df)


# =========================================================
# UTIL STATISTIK
# =========================================================

def classify(ref_price, target_price, threshold):
    """Klasifikasi arah berdasarkan threshold FLAT."""
    change = target_price / ref_price - 1
    if change > threshold:
        return "UP", change
    if change < -threshold:
        return "DOWN", change
    return "FLAT", change


def wilson_ci(k, n, z=1.96):
    """Confidence interval 95% untuk proporsi (Wilson)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - margin, center + margin


# =========================================================
# CHART
# =========================================================

def build_chart(visible, future=None, show_ma=True, show_taker=False):
    n_rows = 3 if show_taker else 2
    heights = [0.68, 0.16, 0.16] if show_taker else [0.78, 0.22]

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=heights, vertical_spacing=0.02,
    )

    # Candle yang terlihat (masa lalu sampai posisi sekarang)
    fig.add_trace(go.Candlestick(
        x=visible["dt"],
        open=visible["open"], high=visible["high"],
        low=visible["low"], close=visible["close"],
        name="BTC/USDT",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Candle masa depan (hanya setelah reveal), dibuat transparan
    if future is not None and len(future):
        fig.add_trace(go.Candlestick(
            x=future["dt"],
            open=future["open"], high=future["high"],
            low=future["low"], close=future["close"],
            name="Hasil (reveal)",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            opacity=0.45,
            showlegend=True,
        ), row=1, col=1)

        fig.add_vline(
            x=future["dt"].iloc[0],
            line_dash="dash", line_color="#ffd54f",
            row=1, col=1,
        )

    if show_ma:
        ma20 = visible["close"].rolling(20).mean()
        ma50 = visible["close"].rolling(50).mean()
        fig.add_trace(go.Scatter(
            x=visible["dt"], y=ma20, mode="lines",
            name="MA20", line=dict(width=1, color="#ffb74d"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=visible["dt"], y=ma50, mode="lines",
            name="MA50", line=dict(width=1, color="#42a5f5"),
        ), row=1, col=1)

    # Volume
    colors = np.where(
        visible["close"] >= visible["open"], "#26a69a", "#ef5350"
    )
    fig.add_trace(go.Bar(
        x=visible["dt"], y=visible["volume"],
        marker_color=colors, name="Volume", showlegend=False,
    ), row=2, col=1)

    # Taker buy ratio (opsional; fitur yang juga dipakai model V13)
    if show_taker and visible["taker_buy_vol"].notna().any():
        ratio = (
            visible["taker_buy_vol"]
            / visible["volume"].replace(0, np.nan)
        )
        fig.add_trace(go.Scatter(
            x=visible["dt"], y=ratio, mode="lines",
            name="Taker buy ratio",
            line=dict(width=1, color="#ab47bc"),
        ), row=3, col=1)
        fig.add_hline(
            y=0.5, line_dash="dot", line_color="gray", row=3, col=1
        )
        fig.update_yaxes(title_text="Taker", row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=620,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.04),
    )
    fig.update_yaxes(title_text="USDT", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)

    return fig


# =========================================================
# SESSION STATE
# =========================================================

def init_state():
    defaults = {
        "idx": None,        # posisi candle terakhir yang terlihat
        "pending": None,    # prediksi yang terkunci, belum di-reveal
        "reveal": None,     # (start_iloc, end_iloc) candle hasil
        "history": [],      # riwayat prediksi
        "data_id": None,    # penanda sumber data aktif
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# =========================================================
# SIDEBAR: SUMBER DATA
# =========================================================

st.sidebar.title("📊 BTC/USDT Sim Lab")
st.sidebar.caption("Replay data nyata Binance untuk latihan.")

source = st.sidebar.radio(
    "Sumber data",
    ["Fetch Binance API", "Upload CSV"],
)

df = None
data_id = None

if source == "Fetch Binance API":
    symbol = st.sidebar.text_input("Symbol", value="BTCUSDT").upper()
    interval = st.sidebar.selectbox(
        "Interval", ["1m", "5m", "15m", "1h"], index=0
    )
    limit = st.sidebar.slider(
        "Jumlah candle", 500, 1500, 1000, step=100
    )

    if st.sidebar.button("🔄 Ambil data", use_container_width=True):
        try:
            df_new, gaps = fetch_binance(symbol, interval, limit)
            st.session_state["_data"] = df_new
            st.session_state["_gaps"] = gaps
            st.session_state["_data_id"] = f"{symbol}-{interval}-{limit}"
        except requests.RequestException as e:
            st.sidebar.error(f"Gagal fetch: {e}")

    if "_data" in st.session_state and st.session_state.get(
        "_data_id", ""
    ).startswith(f"{symbol}-{interval}"):
        df = st.session_state["_data"]
        data_id = st.session_state["_data_id"]
    else:
        st.info("Klik **Ambil data** di sidebar untuk mulai.")

else:
    uploaded = st.sidebar.file_uploader(
        "Upload CSV (format Binance klines)", type=["csv"]
    )
    st.sidebar.caption(
        "Kolom minimal: open_time, open, high, low, close, volume. "
        "File BTCUSDT_1m_50k.csv milikmu kompatibel."
    )

    if uploaded is not None:
        try:
            df, gaps = parse_csv(uploaded.getvalue())
            st.session_state["_gaps"] = gaps
            data_id = f"csv-{uploaded.name}-{uploaded.size}"
        except Exception as e:
            st.sidebar.error(f"CSV tidak valid: {e}")

# Reset posisi kalau sumber data berganti
if df is not None and data_id != st.session_state.data_id:
    st.session_state.data_id = data_id
    st.session_state.idx = min(200, len(df) - 20)
    st.session_state.pending = None
    st.session_state.reveal = None
    st.session_state.history = []

if df is None:
    st.title("📊 BTC/USDT Market Simulation Lab")
    st.write(
        "Pilih sumber data di sidebar untuk mulai. "
        "Candle masa depan akan disembunyikan — tugasmu menebak arahnya."
    )
    st.stop()

if st.session_state.get("_gaps", 0) > 0:
    st.warning(
        f"Data mengandung {st.session_state['_gaps']} gap candle. "
        "Hati-hati: loncatan waktu bisa memengaruhi pembacaan."
    )


# =========================================================
# SIDEBAR: PENGATURAN
# =========================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Pengaturan")

window = st.sidebar.slider("Candle tampil di chart", 30, 300, 100, step=10)
horizon = st.sidebar.selectbox("Horizon prediksi (candle)", [1, 3, 5, 15], index=2)
threshold_pct = st.sidebar.number_input(
    "Batas FLAT (%)", min_value=0.0, max_value=1.0,
    value=0.03, step=0.01,
)
threshold = threshold_pct / 100

show_ma = st.sidebar.checkbox("Tampilkan MA20/MA50", value=True)
show_taker = st.sidebar.checkbox("Tampilkan taker buy ratio", value=False)

autoplay = st.sidebar.toggle("Auto-play", value=False)
speed = st.sidebar.select_slider(
    "Speed auto-play", options=[0.5, 1, 2, 5, 10], value=2
)

max_idx = len(df) - 1
if st.session_state.idx is None:
    st.session_state.idx = min(200, max_idx - horizon - 1)

idx = int(np.clip(st.session_state.idx, window, max_idx))
st.session_state.idx = idx

locked = st.session_state.pending is not None


# =========================================================
# SIDEBAR: NAVIGASI
# =========================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Navigasi")

nav1 = st.sidebar.columns(3)
nav2 = st.sidebar.columns(3)

def move(step):
    st.session_state.idx = int(
        np.clip(st.session_state.idx + step, window, max_idx)
    )
    st.session_state.pending = None
    st.session_state.reveal = None

if nav1[0].button("⏪ -100", disabled=locked, use_container_width=True):
    move(-100)
    st.rerun()
if nav1[1].button("◀ -10", disabled=locked, use_container_width=True):
    move(-10)
    st.rerun()
if nav1[2].button("-1", disabled=locked, use_container_width=True):
    move(-1)
    st.rerun()
if nav2[0].button("+1", disabled=locked, use_container_width=True):
    move(1)
    st.rerun()
if nav2[1].button("+10 ▶", disabled=locked, use_container_width=True):
    move(10)
    st.rerun()
if nav2[2].button("+100 ⏩", disabled=locked, use_container_width=True):
    move(100)
    st.rerun()

if st.sidebar.button(
    "🎲 Posisi acak", disabled=locked, use_container_width=True
):
    st.session_state.idx = int(
        np.random.randint(window, max_idx - horizon - 1)
    )
    st.session_state.pending = None
    st.session_state.reveal = None
    st.rerun()


# =========================================================
# PANEL UTAMA: INFO + CHART
# =========================================================

visible = df.iloc[max(0, idx - window):idx + 1]
future = None
if st.session_state.reveal is not None:
    s, e = st.session_state.reveal
    future = df.iloc[s:e]

last = visible.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Waktu candle", last["dt"].strftime("%Y-%m-%d %H:%M UTC"))
c2.metric("Close", f"${last['close']:,.2f}")
c3.metric("Posisi", f"{idx:,} / {len(df):,}")
c4.metric(
    "Sisa data tersembunyi",
    f"{len(df) - idx - 1:,} candle",
)

fig = build_chart(visible, future, show_ma, show_taker)
st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PANEL PREDIKSI
# =========================================================

st.subheader("🎯 Latihan prediksi")

can_predict = idx + horizon < len(df)

if not can_predict:
    st.info("Sudah di ujung data — tidak cukup candle untuk reveal.")

if st.session_state.pending is None:
    st.write(
        f"Tebak arah **{horizon} candle ke depan** "
        f"(FLAT jika perubahan dalam ±{threshold_pct:.2f}%):"
    )

    b1, b2, b3 = st.columns(3)
    pred_up = b1.button("📈 UP", disabled=not can_predict,
                        use_container_width=True)
    pred_flat = b2.button("➡️ FLAT", disabled=not can_predict,
                          use_container_width=True)
    pred_down = b3.button("📉 DOWN", disabled=not can_predict,
                          use_container_width=True)

    chosen = (
        "UP" if pred_up else
        "FLAT" if pred_flat else
        "DOWN" if pred_down else None
    )

    if chosen:
        st.session_state.pending = {
            "pred": chosen,
            "ref_idx": idx,
            "ref_price": float(last["close"]),
            "ref_time": str(last["dt"]),
        }
        st.rerun()

else:
    pending = st.session_state.pending
    st.write(
        f"Prediksi terkunci: **{pending['pred']}** "
        f"@ ${pending['ref_price']:,.2f} ({pending['ref_time']})"
    )

    if st.session_state.reveal is None:
        if st.button("🔓 Reveal hasil", type="primary"):
            s = pending["ref_idx"] + 1
            e = s + horizon
            target_price = float(df.iloc[e - 1]["close"])

            actual, change = classify(
                pending["ref_price"], target_price, threshold
            )

            st.session_state.history.append({
                "time": pending["ref_time"],
                "ref_price": pending["ref_price"],
                "target_price": target_price,
                "change_pct": change * 100,
                "pred": pending["pred"],
                "actual": actual,
                "correct": pending["pred"] == actual,
            })

            st.session_state.reveal = (s, e)
            st.rerun()

    else:
        record = st.session_state.history[-1]

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Harga target", f"${record['target_price']:,.2f}")
        r2.metric("Perubahan", f"{record['change_pct']:+.3f}%")
        r3.metric("Aktual", record["actual"])
        r4.metric(
            "Hasil",
            "✅ Benar" if record["correct"] else "❌ Salah",
        )

        if st.button("➡️ Lanjut", type="primary"):
            # Lompat sejauh horizon: window tidak tumpang tindih
            st.session_state.idx = min(
                pending["ref_idx"] + horizon, max_idx
            )
            st.session_state.pending = None
            st.session_state.reveal = None
            st.rerun()


# =========================================================
# STATISTIK
# =========================================================

st.markdown("---")
st.subheader("📈 Statistik latihan")

history = st.session_state.history

if not history:
    st.caption("Belum ada prediksi. Statistik muncul setelah prediksi pertama.")
else:
    h = pd.DataFrame(history)
    n = len(h)
    k = int(h["correct"].sum())
    acc = k / n
    lo, hi = wilson_ci(k, n)

    majority_baseline = h["actual"].value_counts(normalize=True).max()

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total prediksi", n)
    s2.metric("Akurasi kamu", f"{acc:.1%}")
    s3.metric("CI 95%", f"{lo:.1%} – {hi:.1%}")
    s4.metric(
        "Baseline majority",
        f"{majority_baseline:.1%}",
        help="Akurasi kalau kamu selalu menebak kelas yang paling sering muncul.",
    )

    if n < 30:
        st.caption(
            "⚠️ Sampel masih kecil (<30). Jangan simpulkan apa-apa dulu — "
            "interval kepercayaan masih sangat lebar."
        )
    elif lo > majority_baseline:
        st.success(
            "Batas bawah CI kamu di atas baseline majority. "
            "Ini tanda awal yang menarik — tapi tetap belum memperhitungkan "
            "fee/slippage/payout kalau dipakai trading."
        )
    else:
        st.caption(
            "Akurasi kamu belum terbukti melampaui baseline sederhana. "
            "Itu normal — dan justru pelajaran pentingnya."
        )

    t1, t2 = st.columns(2)

    with t1:
        st.write("**Akurasi per prediksi:**")
        per_pred = (
            h.groupby("pred")["correct"]
            .agg(["count", "mean"])
            .rename(columns={"count": "jumlah", "mean": "akurasi"})
        )
        per_pred["akurasi"] = per_pred["akurasi"].map("{:.1%}".format)
        st.dataframe(per_pred, use_container_width=True)

    with t2:
        st.write("**Prediksi vs aktual:**")
        crosstab = pd.crosstab(
            h["pred"], h["actual"], margins=True, margins_name="Total"
        )
        st.dataframe(crosstab, use_container_width=True)

    st.write("**Akurasi kumulatif:**")
    h["cum_acc"] = h["correct"].expanding().mean()
    st.line_chart(h["cum_acc"])

    d1, d2 = st.columns(2)
    d1.download_button(
        "💾 Unduh riwayat (CSV)",
        h.drop(columns=["cum_acc"]).to_csv(index=False),
        "riwayat_prediksi.csv",
        "text/csv",
    )
    if d2.button("🗑️ Reset statistik"):
        st.session_state.history = []
        st.rerun()


# =========================================================
# AUTO-PLAY
# =========================================================

if autoplay and not locked and idx < max_idx:
    time.sleep(1 / speed)
    st.session_state.idx = idx + 1
    st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.caption(
    "Data: Binance public API / CSV milikmu. "
    "fee, spread, slippage, dan payout tidak dimodelkan di sini."
)