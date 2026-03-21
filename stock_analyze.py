import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
import sys

# Optional Streamlit Import
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

UNIVERSE = [
    '2330.TW', '2317.TW', '2454.TW', '2308.TW', '2382.TW', 
    '2881.TW', '2882.TW', '2883.TW', '2884.TW', '2885.TW', 
    '2886.TW', '2887.TW', '2890.TW', '2891.TW', '2892.TW',
    '2603.TW', '2609.TW', '2615.TW', 
    '2606.TW', '2610.TW'
]

def calculate_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_poc(df, step=2.0):
    if df.empty: return np.nan
    top = df['High'].max()
    bottom = df['Low'].min()
    if pd.isna(top) or pd.isna(bottom) or top == bottom:
        return np.nan
        
    dynRows = max(1, int(round((top - bottom) / step)))
    actual_step = (top - bottom) / dynRows
    
    bins = np.zeros(dynRows)
    bin_tops = bottom + actual_step * np.arange(1, dynRows + 1)
    bin_bots = bottom + actual_step * np.arange(0, dynRows)
    
    for h, l, v in zip(df['High'], df['Low'], df['Volume']):
        if pd.isna(h) or pd.isna(l) or pd.isna(v) or h == l:
            continue
        intersect_mask = (bin_bots <= h) & (bin_tops >= l)
        intersect_count = np.sum(intersect_mask)
        if intersect_count > 0:
            bins[intersect_mask] += v / intersect_count
            
    max_bin_idx = np.argmax(bins)
    poc = (bin_tops[max_bin_idx] + bin_bots[max_bin_idx]) / 2.0
    return poc

def fetch_data(tickers, progress_callback=None):
    all_data = []
    chunk_size = 20
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        msg = f"Downloading batch {i//chunk_size + 1}/{max(1, (len(tickers)-1)//chunk_size + 1)}: {chunk}"
        if progress_callback:
            progress_callback(msg)
            
        df = yf.download(chunk, period="1y", group_by="ticker", progress=False)
        all_data.append(df)
        time.sleep(1)
        
    processed_dfs = []
    for chunk, df in zip([tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)], all_data):
        if len(chunk) == 1:
            ticker = chunk[0]
            df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
        processed_dfs.append(df)
        
    final_df = pd.concat(processed_dfs, axis=1)
    final_df.dropna(how='all', axis=1, inplace=True)
    return final_df

def apply_strategy(df_ticker):
    df = df_ticker.copy()
    if df.empty or len(df) < 60:
        return None
        
    df['ATR'] = calculate_atr(df)
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['POC'] = calculate_poc(df)
    
    df['Signal_Buy'] = (df['Close'].shift(1) <= df['POC']) & (df['Close'] > df['POC'])
    df['Score'] = (df['Close'] / df['MA60']) - 1
    
    stop_loss = np.full(len(df), np.nan)
    take_profit = np.full(len(df), np.nan)
    fib_618 = np.full(len(df), np.nan)
    
    global_high = -np.inf
    high_idx = 0
    swing_low = np.inf
    
    for i in range(1, len(df)):
        today_high = df['High'].iloc[i]
        today_low = df['Low'].iloc[i]
        today_atr = df['ATR'].iloc[i]
        
        if not np.isnan(today_atr) and not np.isnan(today_high):
            current_sl = today_high - 2 * today_atr
            if np.isnan(stop_loss[i-1]):
                stop_loss[i] = current_sl
            else:
                stop_loss[i] = max(stop_loss[i-1], current_sl)
        else:
            stop_loss[i] = stop_loss[i-1]
            
        if not np.isnan(today_high):
            if today_high > global_high:
                old_high_idx = high_idx
                global_high = today_high
                high_idx = i
                
                if old_high_idx < high_idx and old_high_idx > 0:
                    swing_low = df['Low'].iloc[old_high_idx:high_idx+1].min()
                else:
                    swing_low = df['Low'].iloc[:high_idx+1].min()
            
            if global_high != -np.inf and swing_low != np.inf:
                diff = global_high - swing_low
                fib_618[i] = global_high - 0.618 * diff
                take_profit[i] = global_high
    
    df['Stop_Loss'] = stop_loss
    df['Take_Profit'] = take_profit
    df['Fib_0.618'] = fib_618
    return df

def run_strategy(progress_callback=None):
    if progress_callback:
        progress_callback("Fetching data from Yahoo Finance...")
    df_all = fetch_data(UNIVERSE, progress_callback)
    results = {}
    
    if isinstance(df_all.columns, pd.MultiIndex):
        if len(df_all.columns.levels) == 2:
            if 'Close' in df_all.columns.levels[0]:
                is_price_first = True
                close_px = df_all.xs('Close', level=0, axis=1)
                vol = df_all.xs('Volume', level=0, axis=1)
            else:
                is_price_first = False
                close_px = df_all.xs('Close', level=1, axis=1)
                vol = df_all.xs('Volume', level=1, axis=1)
                
            trading_value = close_px * vol
            idx = -1
            latest_tv = trading_value.iloc[idx].dropna()
            while latest_tv.empty and abs(idx) < len(trading_value):
                idx -= 1
                latest_tv = trading_value.iloc[idx].dropna()
                
            top_100_tickers = latest_tv.nlargest(100).index.tolist()
            if progress_callback:
                progress_callback(f"--- 篩選成交值前 100 名熱門股 ---\n共擷取 {len(top_100_tickers)} 檔")
            
            for ticker in top_100_tickers:
                if ticker not in UNIVERSE: continue
                try:
                    df_ticker = df_all.xs(ticker, level=1, axis=1).copy() if is_price_first else df_all[ticker].copy()
                    df_ticker.dropna(how='all', inplace=True)
                    res = apply_strategy(df_ticker)
                    if res is not None:
                        results[ticker] = res.iloc[-1].to_dict()
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Error processing {ticker}: {e}")
    
    summary_df = pd.DataFrame(results).T
    if not summary_df.empty:
        if 'Score' in summary_df.columns:
            summary_df.sort_values(by='Score', ascending=False, inplace=True)
            
        cols_to_show = ['Close', 'Signal_Buy', 'POC', 'Score', 'Stop_Loss', 'Take_Profit', 'Fib_0.618']
        cols_to_show = [c for c in cols_to_show if c in summary_df.columns]
        summary_df = summary_df[cols_to_show]
    
    return summary_df

def terminal_main():
    summary_df = run_strategy(lambda msg: print(msg))
    if not summary_df.empty:
        print("\n--- Latest Trading Signals (Sorted by Strength Score) ---")
        print(summary_df)
        
        buy_candidates = summary_df[summary_df['Signal_Buy'] == True]
        if not buy_candidates.empty:
            print("\n💡 今日符合所有入場條件（收盤價向上穿越 52週 POC）可選購之股票：")
            print(", ".join(list(buy_candidates.index)))
        else:
            print("\n💡 今日無符合所有入場條件之股票。")
            
        summary_df.to_csv("trading_signals.csv")
        print("\nSaved to trading_signals.csv successfully.")
    else:
        print("No data processed.")

def streamlit_main():
    if not STREAMLIT_AVAILABLE:
        print("Streamlit 尚未安裝，請先執行 `pip install streamlit`")
        return

    st.set_page_config(page_title="台股交易策略: Volume Profile", layout="wide")
    st.title("📈 台股交易策略: VPFR 動態突破與籌碼分析")
    st.markdown("自動掃描市場熱門股，透過 **Volume Profile (POC)** 判斷籌碼密集區突破，並搭配 **ATR 移動停損** 與 **斐波納契回撤** 控制風險。")
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def cached_run_strategy():
        return run_strategy()
    
    if st.button("🚀 執行選股與更新資料", use_container_width=True) or 'df' in st.session_state:
        with st.spinner("從 Yahoo Finance 下載最新市場資料中...請稍候... (需花費幾秒鐘時間)"):
            summary_df = cached_run_strategy()
            st.session_state['df'] = summary_df
            
        if not summary_df.empty:
            buy_candidates = summary_df[summary_df['Signal_Buy'] == True]
            
            if not buy_candidates.empty:
                st.success("🎯 **今日發現符合進場條件的熱門股！**")
                st.write(", ".join(list(buy_candidates.index)))
            else:
                st.info("💡 **今日無符合「收盤價向上穿越 52週 POC」條件之股票。**")
            
            st.subheader("📊 最新策略訊號排名 (根據強勢度 Score 排列)")
            
            # 使用 dataframe 呈現，並標記買入訊號
            def highlight_buy(s):
                return ['background-color: #2e7d32; color: white' if v else '' for v in s]
            
            styled_df = summary_df.style.apply(highlight_buy, subset=['Signal_Buy'] if 'Signal_Buy' in summary_df.columns else None)
            st.dataframe(styled_df, use_container_width=True, height=600)
            
            csv = summary_df.to_csv().encode('utf-8-sig')
            st.download_button(
                label="📥 下載 CSV 訊號報表",
                data=csv,
                file_name='trading_signals.csv',
                mime='text/csv',
            )
        else:
            st.warning("無資料可顯示。")

if __name__ == "__main__":
    is_streamlit = False
    if sys.argv and len(sys.argv) > 0:
        if 'streamlit' in sys.argv[0] or 'streamlit' in sys.modules:
            is_streamlit = True
            
    if 'STREAMLIT_RUNTIME' in os.environ:
        is_streamlit = True
        
    if is_streamlit:
        streamlit_main()
    else:
        terminal_main()
