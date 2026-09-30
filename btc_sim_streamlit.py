#!/usr/bin/env python3
"""
btc_market_lab.py — BTCUSDT Market Liquidity & Execution Simulator
Optimized for Streamlit with Geo-restriction handling (HTTP 451 fallback).
"""

import streamlit as st
import requests
import pandas as pd
import time
import sqlite3
from datetime import datetime, timezone

# =========================================================
# CONFIGURATION
# =========================================================
SYMBOL = "BTCUSDT"
TAKER_FEE_RATE = 0.001  # 0.1% (Standard Binance Spot Taker Fee)

# Binance API Endpoints (Global & US for Geo-fallback)
API_ENDPOINTS = [
    "https://api.binance.com",
    "https://api.binance.us"
]

MAX_CANDLES = 1000  # Binance API Limit per request

st.set_page_config(page_title="BTCUSDT Market Lab", layout="wide", initial_sidebar_state="expanded")

# =========================================================
# API HANDLING & DATA FETCHING
# =========================================================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_market_data(endpoint_type, symbol, interval=None, limit=None):
    """
    Fetch data from Binance API with Geo-blocking fallback.
    endpoint_type: 'klines', 'depth', or 'ticker'
    """
    last_error = None
    
    # Cap limit for klines
    if endpoint_type == "klines" and limit:
        limit = min(max(int(limit), 1), MAX_CANDLES)

    for base_url in API_ENDPOINTS:
        url = f"{base_url}/api/v3/{endpoint_type}"
        params = {"symbol": symbol.upper()}
        
        if endpoint_type == "klines":
            params["interval"] = interval
            params["limit"] = limit
        elif endpoint_type == "depth":
            params["limit"] = 100  # Max depth snapshot
            
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 451:
                last_error = f"Geo-blocked (451) on {base_url}"
                continue  # Try next endpoint
            
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, (list, dict)):
                raise RuntimeError(f"Invalid response format: {data}")
                
            return data, base_url
            
        except requests.RequestException as e:
            last_error = f"Connection error on {base_url}: {e}"
            
    raise RuntimeError(f"Failed to fetch data from all endpoints. Last error: {last_error}")


def get_order_book():
    """Fetch Order Book Snapshot"""
    data, source = fetch_market_data("depth", SYMBOL)
    
    bids = pd.DataFrame(data['bids'], columns=['Price', 'Volume']).astype(float)
    asks = pd.DataFrame(data['asks'], columns=['Price', 'Volume']).astype(float)
    
    return bids, asks, source


def get_ticker():
    """Fetch Best Bid/Ask Ticker"""
    data, source = fetch_market_data("ticker", SYMBOL)
    return data


def get_candles(symbol, interval, limit):
    """Fetch Historical Candles (Klines)"""
    data, source = fetch_market_data("klines", symbol, interval=interval, limit=limit)
    
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "taker_buy_vol", "taker_quote_vol", "ignore"
    ])
    
    numeric_cols = ["open", "high", "low", "close", "volume", "taker_buy_vol"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    # Convert timestamps
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
    df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
    
    return df, source

# =========================================================
# SIMULATION LOGIC (MARKET EXECUTION)
# =========================================================
def simulate_market_order(order_book_side, order_type, amount):
    """
    Simulate market execution against the order book.
    order_book_side: DataFrame asks (for Buy) or bids (for Sell)
    amount: USDT (for Buy) or BTC (for Sell)
    """
    total_cost = 0.0
    total_btc_filled = 0.0
    remaining_amount = amount
    
    levels = order_book_side.copy()
    
    for _, row in levels.iterrows():
        price = row['Price']
        volume = row['Volume']
        
        if order_type == "BUY":
            # Buying BTC with USDT
            level_capacity_usdt = price * volume
            
            if remaining_amount <= level_capacity_usdt:
                btc_bought = remaining_amount / price
                total_cost += remaining_amount
                total_btc_filled += btc_bought
                remaining_amount = 0
                break
            else:
                total_cost += level_capacity_usdt
                total_btc_filled += volume
                remaining_amount -= level_capacity_usdt
                
        elif order_type == "SELL":
            # Selling BTC for USDT
            if remaining_amount <= volume:
                usdt_received = remaining_amount * price
                total_cost += usdt_received
                total_btc_filled += remaining_amount
                remaining_amount = 0
                break
            else:
                usdt_received = volume * price
                total_cost += usdt_received
                total_btc_filled += volume
                remaining_amount -= volume

    if remaining_amount > 0:
        return None, None, None, "Order size exceeds available liquidity in order book (limit 100 levels)."
    
    avg_fill_price = total_cost / total_btc_filled
    return total_btc_filled, total_cost, avg_fill_price, "Success"

# =========================================================
# STREAMLIT UI
# =========================================================
def main():
    st.title(f"📊 {SYMBOL} Market Lab & Liquidity Simulator")
    
    # Sidebar Controls
    st.sidebar.header("Controls")
    st.sidebar.markdown("---")
    
    auto_refresh = st.sidebar.checkbox("Auto-Refresh Data (5s)", value=False)
    
    if st.sidebar.button("🔄 Refresh Data Manually") or auto_refresh:
        st.cache_data.clear()
        if auto_refresh:
            time.sleep(5)
            st.rerun()

    st.sidebar.info("Note: If you encounter '451 Client Error', your server location is restricted. The app will automatically try to connect via `api.binance.us`.")

    # Fetch Data
    try:
        bids, asks, ob_source = get_order_book()
        ticker = get_ticker()
        candles_df, kline_source = get_candles(SYMBOL, "1m", MAX_CANDLES)
        
        # Display Source Info
        st.success(f"Connected to: `{ob_source}` (Order Book) | `{kline_source}` (Candles)")
        
    except Exception as e:
        st.error(f"Failed to load market data: {e}")
        st.info("Tip: Try running locally or use a VPN if your server is geo-restricted.")
        return

    # --- METRICS ROW ---
    best_bid = float(ticker['bidPrice'])
    best_ask = float(ticker['askPrice'])
    mid_price = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mid Price", f"{mid_price:,.2f} USDT")
    col2.metric("Spread", f"{spread:,.2f} USDT")
    col3.metric("Best Bid", f"{best_bid:,.2f} USDT")
    col4.metric("Best Ask", f"{best_ask:,.2f} USDT")
    
    st.markdown("---")
    
    # --- TABS: SIMULATOR & CHARTS ---
    tab1, tab2 = st.tabs(["🛒 Execution Simulator", "📈 Market Structure"])
    
    # TAB 1: EXECUTION SIMULATOR
    with tab1:
        st.header("Market Order Execution Simulator")
        
        sim_col1, sim_col2 = st.columns([1, 2])
        
        with sim_col1:
            st.subheader("Order Parameters")
            order_type = st.radio("Order Type", ["BUY (Buy BTC)", "SELL (Sell BTC)"])
            
            if "BUY" in order_type:
                amount = st.number_input("Amount (USDT)", min_value=10.0, value=1000.0, step=100.0)
                btn_label = "Simulate Buy"
                book_side = asks
                best_price_ref = best_ask
            else:
                amount = st.number_input("Amount (BTC)", min_value=0.001, value=0.5, step=0.1, format="%.3f")
                btn_label = "Simulate Sell"
                book_side = bids
                best_price_ref = best_bid
                
            if st.button(btn_label, type="primary", use_container_width=True):
                btc_filled, total_cost, avg_price, status = simulate_market_order(
                    book_side, "BUY" if "BUY" in order_type else "SELL", amount
                )
                
                if status == "Success":
                    st.success("✅ Simulation Successful")
                    
                    res1, res2, res3 = st.columns(3)
                    
                    if "BUY" in order_type:
                        fee_btc = btc_filled * TAKER_FEE_RATE
                        net_btc = btc_filled - fee_btc
                        slippage_pct = ((avg_price - best_ask) / best_ask) * 100
                        
                        res1.metric("Total BTC (Gross)", f"{btc_filled:.8f}")
                        res2.metric("Taker Fee (0.1%)", f"{fee_btc:.8f}")
                        res3.metric("Total BTC (Net)", f"{net_btc:.8f}")
                        
                        st.markdown(f"**Average Fill Price:** `{avg_price:,.2f} USDT`")
                        st.markdown(f"**Slippage from Best Ask:** `{slippage_pct:.4f}%`")
                        
                    else:
                        fee_usdt = total_cost * TAKER_FEE_RATE
                        net_usdt = total_cost - fee_usdt
                        slippage_pct = ((best_bid - avg_price) / best_bid) * 100
                        
                        res1.metric("Total USDT (Gross)", f"{total_cost:,.2f}")
                        res2.metric("Taker Fee (0.1%)", f"{fee_usdt:,.2f}")
                        res3.metric("Total USDT (Net)", f"{net_usdt:,.2f}")
                        
                        st.markdown(f"**Average Fill Price:** `{avg_price:,.2f} USDT`")
                        st.markdown(f"**Slippage from Best Bid:** `{slippage_pct:.4f}%`")
                else:
                    st.error(f"❌ {status}")

        with sim_col2:
            st.subheader("Order Book (Top 15 Levels)")
            ob_col1, ob_col2 = st.columns(2)
            
            with ob_col1:
                st.markdown("**🟩 BIDS (Buyers)**")
                bids_display = bids.head(15).copy()
                bids_display['Total (USDT)'] = (bids_display['Price'] * bids_display['Volume']).round(2)
                st.dataframe(
                    bids_display.style.format({"Price": "{:,.2f}", "Volume": "{:,.4f}", "Total (USDT)": "{:,.2f}"}),
                    use_container_width=True, height=400
                )
                
            with ob_col2:
                st.markdown("**🟥 ASKS (Sellers)**")
                asks_display = asks.head(15).copy()
                asks_display['Total (USDT)'] = (asks_display['Price'] * asks_display['Volume']).round(2)
                st.dataframe(
                    asks_display.style.format({"Price": "{:,.2f}", "Volume": "{:,.4f}", "Total (USDT)": "{:,.2f}"}),
                    use_container_width=True, height=400
                )

    # TAB 2: MARKET STRUCTURE
    with tab2:
        st.header("Recent Market Structure (1m Candles)")
        
        if not candles_df.empty:
            # Show last 10 candles
            recent_candles = candles_df.tail(10).copy()
            recent_candles = recent_candles[["open_time", "open", "high", "low", "close", "volume"]]
            
            st.dataframe(
                recent_candles.style.format({
                    "open": "{:,.2f}", "high": "{:,.2f}", 
                    "low": "{:,.2f}", "close": "{:,.2f}", 
                    "volume": "{:,.4f}"
                }),
                use_container_width=True
            )
            
            st.caption(f"Data Source: {kline_source} | Limit: {MAX_CANDLES} candles")
        else:
            st.warning("No candle data available.")

    st.markdown("---")
    st.caption("⚠️ Disclaimer: This is a simulation based on order book snapshots. Real execution may vary due to latency and market volatility. Not financial advice.")

if __name__ == "__main__":
    main()
