import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pygad
from scipy import stats
import plotly.graph_objects as go

class QuantGAEngine:
    def __init__(self, tickers, period="1mo", interval="5m"):
        self.tickers = [t.strip().upper() for t in tickers]
        self.period = period
        self.interval = interval
        self.data_pool = {}

    def download_data(self):
        """V1.3 終極暴力修復：強制結構歸一化"""
        st.info(f"正在獲取數據：{', '.join(self.tickers)}...")
        
        try:
            # 1. 直接下載，不帶任何 group_by 參數，讓 yf 自行決定
            raw_data = yf.download(
                tickers=self.tickers, 
                period=self.period, 
                interval=self.interval, 
                auto_adjust=True,
                threads=True
            )
            
            if raw_data.empty:
                st.error("❌ Yahoo Finance 回傳空數據。請嘗試縮短時段（如 1mo）或更換頻率（如 1h）。")
                return 0

            # 2. 處理 MultiIndex (多支股票時常見)
            # 我們將其「拍平」，讓欄位變成簡單的字串
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in raw_data.columns.values]
            
            valid_count = 0
            for ticker in self.tickers:
                # 尋找屬於該 Ticker 的欄位
                # 例如欄位可能叫 'AAPL_Close' 或直接叫 'Close' (單支股票時)
                ticker_cols = [c for c in raw_data.columns if ticker in c]
                
                if ticker_cols:
                    # 提取該 Ticker 的數據並重命名
                    sub_df = raw_data[ticker_cols].copy()
                    sub_df.columns = [c.replace(f"{ticker}_", "") for c in sub_df.columns]
                else:
                    # 如果找不到帶 Ticker 名稱的欄位，且只有一個 Ticker，則直接使用 raw_data
                    if len(self.tickers) == 1:
                        sub_df = raw_data.copy()
                    else:
                        continue

                # 3. 強制標準化欄位名
                sub_df.columns = [c.capitalize() for c in sub_df.columns]
                
                # 檢查必要欄位
                required = ['High', 'Low', 'Close', 'Volume']
                if all(col in sub_df.columns for col in required):
                    sub_df = sub_df.dropna()
                    if len(sub_df) > 20:
                        sub_df['TP'] = (sub_df['High'] + sub_df['Low'] + sub_df['Close']) / 3
                        self.data_pool[ticker] = sub_df
                        valid_count += 1
                else:
                    st.warning(f"⚠️ {ticker} 欄位不完整。現有欄位: {list(sub_df.columns)}")

            return valid_count
            
        except Exception as e:
            st.error(f"❌ 數據處理失敗: {str(e)}")
            return 0

    @staticmethod
    def backtest_logic(df, window, buy_t, sell_t, commission, slippage):
        window = int(window)
        tp_v = (df['TP'] * df['Volume']).rolling(window=window).sum()
        v_sum = df['Volume'].rolling(window=window).sum()
        vwap = tp_v / v_sum
        dist = (df['Close'] - vwap) / vwap
        returns = []
        pos = 0 
        entry_price = 0
        prices = df['Close'].values
        dists = dist.values
        
        for i in range(window, len(df)):
            if pos == 0 and dists[i] < -buy_t:
                pos = 1
                entry_price = prices[i] * (1 + slippage)
            elif pos == 1 and (dists[i] > sell_t or i == len(df)-1):
                exit_price = prices[i] * (1 - slippage)
                net_ret = ((exit_price - entry_price) / entry_price) - (commission * 2)
                returns.append(net_ret)
                pos = 0
        return returns

# --- Streamlit UI ---
st.set_page_config(page_title="VWAP GA Optimizer V1.3", layout="wide")
st.title("🛡️ 專業級遺傳算法策略優化器 V1.3")
st.caption("終極兼容版：解決單/多標的數據結構報錯")

with st.sidebar:
    st.header("🛠️ 系統配置")
    tickers_input = st.text_input("股票代碼", value="2330.TW")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    period = st.selectbox("回測時段", ["1mo", "60d"], index=0)
    interval = st.selectbox("K線頻率", ["5m", "15m", "30m", "1h"], index=1) # 預設改 15m 較穩
    comm = st.number_input("單邊手續費 (%)", value=0.030, format="%.3f") / 100
    slip = st.number_input("單邊滑點 (%)", value=0.010, format="%.3f") / 100
    num_gen = st.slider("演化代數", 5, 50, 15)
    pop_size = st.slider("種群大小", 10, 100, 20)

if st.button("🔥 啟動全域優化引擎"):
    if not tickers:
        st.warning("請先輸入股票代碼。")
    else:
        engine = QuantGAEngine(tickers, period, interval)
        num_ready = engine.download_data()
        
        if num_ready > 0:
            st.success(f"✅ 成功加載 {num_ready} 隻股票。開始進化...")
            progress_bar = st.progress(0)
            
            def fitness_func(ga_instance, solution, solution_idx):
                win, b_t, s_t = solution
                all_rets = []
                for t in engine.data_pool:
                    rets = engine.backtest_logic(engine.data_pool[t], win, b_t, s_t, comm, slip)
                    all_rets.extend(rets)
                if len(all_rets) < 15: return -5.0
                avg_ret = np.mean(all_rets)
                std_ret = np.std(all_rets)
                t_stat, p_val = stats.ttest_1samp(all_rets, 0, alternative='greater')
                sharpe = (avg_ret / std_ret) * np.sqrt(len(all_rets)) if std_ret > 0 else 0
                fitness = sharpe * (1 - p_val)
                if p_val > 0.05: fitness *= 0.1
                return float(fitness)

            gene_space = [range(10, 101, 10), np.linspace(0.001, 0.015, 15), np.linspace(0.001, 0.015, 15)]

            ga_instance = pygad.GA(
                num_generations=num_gen,
                num_parents_mating=int(pop_size/2),
                fitness_func=fitness_func,
                sol_per_pop=pop_size,
                num_genes=3,
                gene_space=gene_space,
                parent_selection_type="rank",
                on_generation=lambda ga: progress_bar.progress(ga.generations_completed / num_gen)
            )

            with st.spinner("🧠 正在演化搜尋..."):
                ga_instance.run()

            solution, sol_fitness, _ = ga_instance.best_solution()
            st.markdown("---")
            st.header("🎯 最佳參數結果")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("VWAP Window", f"{int(solution[0])}")
            c2.metric("買入閾值", f"{solution[1]:.3%}")
            c3.metric("賣出閾值", f"{solution[2]:.3%}")
            c4.metric("最終適應度", f"{sol_fitness:.4f}")
            
            fig = go.Figure(go.Scatter(y=ga_instance.best_solutions_fitness, mode='lines+markers'))
            fig.update_layout(template="plotly_dark", title="進化歷史")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ 無法讀取數據。建議：1.檢查代碼是否正確 2.將時段縮短為 1mo 3.更換頻率為 1h 測試。")
