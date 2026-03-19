import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pygad
from scipy import stats
import plotly.graph_objects as go

class QuantGAEngine:
    def __init__(self, tickers, period="1mo", interval="1h"):
        self.tickers = [t.strip().upper() for t in tickers]
        self.period = period
        self.interval = interval
        self.data_pool = {}

    def download_data(self):
        """V1.4 容錯加強版：自動頻率回退機制"""
        st.info(f"正在連線 Yahoo Finance 獲取：{', '.join(self.tickers)}...")
        
        try:
            # 第一步：嘗試原始下載
            raw_data = yf.download(
                tickers=self.tickers, 
                period=self.period, 
                interval=self.interval, 
                threads=True
            )
            
            # 如果失敗，自動嘗試 1h 頻率 (最穩定的頻率)
            if raw_data.empty and self.interval != "1h":
                st.warning(f"⚠️ {self.interval} 數據獲取失敗，自動回退至 1h 頻率重試...")
                self.interval = "1h"
                raw_data = yf.download(self.tickers, period=self.period, interval="1h", threads=True)

            if raw_data.empty:
                st.error("❌ 無法獲取任何數據。請檢查 Ticker 名稱 (如 2330.TW) 或網絡。")
                return 0

            # 歸一化結構：確保它是 MultiIndex 方便處理
            if len(self.tickers) == 1:
                # 單支股票下載時 yf 返回單層，我們手動轉為 MultiIndex 格式一致化
                raw_data.columns = pd.MultiIndex.from_product([[self.tickers[0]], raw_data.columns])

            valid_count = 0
            for ticker in self.tickers:
                if ticker in raw_data.columns.get_level_values(0):
                    df = raw_data[ticker].copy().dropna()
                    
                    # 統一欄位大小寫
                    df.columns = [c.capitalize() for c in df.columns]
                    
                    # 針對 yfinance 不同版本，Close 可能叫 Adj close，做修正
                    if 'Close' not in df.columns and 'Adj close' in df.columns:
                        df.rename(columns={'Adj close': 'Close'}, inplace=True)

                    required = ['High', 'Low', 'Close', 'Volume']
                    if all(col in df.columns for col in required) and len(df) > 10:
                        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
                        self.data_pool[ticker] = df
                        valid_count += 1
            
            return valid_count
            
        except Exception as e:
            st.error(f"❌ 數據引擎異常: {str(e)}")
            return 0

    @staticmethod
    def backtest_logic(df, window, buy_t, sell_t, commission, slippage):
        window = int(window)
        tp_v = (df['TP'] * df['Volume']).rolling(window=window).sum()
        v_sum = df['Volume'].rolling(window=window).sum()
        vwap = tp_v / v_sum
        dist = (df['Close'] - vwap) / vwap
        returns = []
        pos, entry_p = 0, 0
        prices, dists = df['Close'].values, dist.values
        
        for i in range(window, len(df)):
            if pos == 0 and dists[i] < -buy_t:
                pos, entry_p = 1, prices[i] * (1 + slippage)
            elif pos == 1 and (dists[i] > sell_t or i == len(df)-1):
                net_ret = ((prices[i]*(1-slippage) - entry_p) / entry_p) - (commission*2)
                returns.append(net_ret)
                pos = 0
        return returns

# --- Streamlit UI ---
st.set_page_config(page_title="VWAP GA Optimizer V1.4", layout="wide")
st.title("🛡️ 專業級遺傳算法策略優化器 V1.4")
st.caption("生存者版本：具備頻率自動回退功能")

with st.sidebar:
    st.header("🛠️ 系統配置")
    ticker_input = st.text_input("股票代碼", value="2330.TW")
    tickers = [ticker_input.strip()] if ticker_input else []
    
    # 這裡強制鎖定最穩定的 Period
    period = "1mo" 
    interval = st.selectbox("K線頻率 (建議 1h 最穩)", ["1h", "30m", "15m", "5m"], index=0)
    
    st.divider()
    comm = st.number_input("單邊手續費 (%)", value=0.03, format="%.3f") / 100
    slip = st.number_input("單邊滑點 (%)", value=0.01, format="%.3f") / 100
    num_gen = st.slider("演化代數", 5, 50, 10)
    pop_size = st.slider("種群大小", 10, 50, 20)

if st.button("🔥 啟動優化引擎"):
    if not tickers:
        st.error("請輸入代碼")
    else:
        engine = QuantGAEngine(tickers, period, interval)
        num_ready = engine.download_data()
        
        if num_ready > 0:
            st.success(f"✅ 成功獲取數據！當前頻率：{engine.interval}")
            prog = st.progress(0)
            
            def fitness_func(ga, sol, idx):
                win, b_t, s_t = sol
                all_rets = []
                for t in engine.data_pool:
                    all_rets.extend(engine.backtest_logic(engine.data_pool[t], win, b_t, s_t, comm, slip))
                if len(all_rets) < 10: return -5.0
                t_stat, p_val = stats.ttest_1samp(all_rets, 0, alternative='greater')
                sharpe = (np.mean(all_rets) / np.std(all_rets)) * np.sqrt(len(all_rets)) if np.std(all_rets) > 0 else 0
                return float(sharpe * (1 - p_val) * (0.1 if p_val > 0.05 else 1.0))

            ga = pygad.GA(
                num_generations=num_gen, num_parents_mating=int(pop_size/2),
                fitness_func=fitness_func, sol_per_pop=pop_size, num_genes=3,
                gene_space=[range(10, 61, 5), np.linspace(0.001, 0.01, 10), np.linspace(0.001, 0.01, 10)],
                on_generation=lambda g: prog.progress(g.generations_completed / num_gen)
            )
            ga.run()
            
            sol, fit, _ = ga.best_solution()
            st.divider()
            st.header("🎯 最佳優化參數")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("VWAP Window", int(sol[0]))
            c2.metric("買入閾值", f"{sol[1]:.2%}")
            c3.metric("賣出閾值", f"{sol[2]:.2%}")
            c4.metric("適應度", f"{fit:.4f}")
            
            st.plotly_chart(go.Figure(go.Scatter(y=ga.best_solutions_fitness)), use_container_width=True)
        else:
            st.error("❌ 仍無法讀取數據。這通常是 Yahoo API 暫時封鎖 IP，建議換個網路環境測試。")
