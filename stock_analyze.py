import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pygad
from scipy import stats
import plotly.graph_objects as go
from tqdm import tqdm

# --- 1. 量化回測引擎 ---
class QuantGAEngine:
    def __init__(self, tickers, period="1mo", interval="5m"):
        self.tickers = tickers
        self.period = period
        self.interval = interval
        self.data_pool = {}

    def download_data(self):
        """分批下載數據並進行初步清洗"""
        st.info(f"正在獲取 {len(self.tickers)} 隻股票數據 (Period: {self.period})...")
        # yfinance 批量下載
        raw_data = yf.download(self.tickers, period=self.period, interval=self.interval, 
                               group_by='ticker', silent=True, threads=True)
        
        valid_count = 0
        for ticker in self.tickers:
            df = raw_data[ticker].dropna() if len(self.tickers) > 1 else raw_data.dropna()
            if not df.empty and len(df) > 100:
                # 預計算 Typical Price 減少 GA 內部運算量
                df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
                self.data_pool[ticker] = df
                valid_count += 1
        return valid_count

    @staticmethod
    def backtest_logic(df, window, buy_t, sell_t, commission, slippage):
        """核心策略：VWAP 偏離回歸"""
        window = int(window)
        # 計算 Rolling VWAP
        tp_v = (df['TP'] * df['Volume']).rolling(window=window).sum()
        v_sum = df['Volume'].rolling(window=window).sum()
        vwap = tp_v / v_sum
        
        # 偏離率
        dist = (df['Close'] - vwap) / vwap
        
        returns = []
        pos = 0
        entry_price = 0
        
        prices = df['Close'].values
        dists = dist.values
        
        for i in range(window, len(df)):
            # 買入邏輯 (考慮滑點)
            if pos == 0 and dists[i] < -buy_t:
                pos = 1
                entry_price = prices[i] * (1 + slippage)
            
            # 賣出邏輯 (考慮滑點與強制平倉)
            elif pos == 1 and (dists[i] > sell_t or i == len(df)-1):
                exit_price = prices[i] * (1 - slippage)
                # 淨收益 = 價差收益 - 雙邊手續費
                net_ret = ((exit_price - entry_price) / entry_price) - (commission * 2)
                returns.append(net_ret)
                pos = 0
        return returns

# --- 2. Streamlit 介面與配置 ---
st.set_page_config(page_title="VWAP GA Optimizer V1", layout="wide")
st.title("🤖 遺傳算法策略優化器 V1")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ 設定參數")
    tickers_str = st.text_input("股票代碼 (逗號隔開)", value="AAPL, TSLA, NVDA, AMD, MSFT")
    tickers = [t.strip().upper() for t in tickers_str.split(",")]
    
    st.subheader("數據範圍")
    period = st.selectbox("歷史長度", ["1mo", "3mo", "6mo"], index=0)
    interval = st.selectbox("K線頻率", ["5m", "15m", "30m"], index=0)
    
    st.subheader("交易成本")
    comm = st.number_input("單邊手續費", value=0.0003, format="%.4f")
    slip = st.number_input("單邊滑點", value=0.0001, format="%.4f")
    
    st.subheader("遺傳算法配置")
    num_gen = st.slider("演化代數", 5, 50, 20)
    pop_size = st.slider("種群大小", 10, 100, 30)

# --- 3. 執行優化 ---
if st.button("開始執行系統優化"):
    engine = QuantGAEngine(tickers, period, interval)
    num_ready = engine.download_data()
    
    if num_ready > 0:
        st.success(f"數據準備就緒，共 {num_ready} 隻有效標的。")
        
        progress_bar = st.progress(0)
        
        def fitness_func(ga_instance, solution, solution_idx):
            win, b_t, s_t = solution
            all_rets = []
            
            for t in engine.data_pool:
                rets = engine.backtest_logic(engine.data_pool[t], win, b_t, s_t, comm, slip)
                all_rets.extend(rets)
            
            # 統計過濾：樣本不足
            if len(all_rets) < 30:
                return -10.0
            
            # 計算統計量
            avg_ret = np.mean(all_rets)
            std_ret = np.std(all_rets)
            t_stat, p_val = stats.ttest_1samp(all_rets, 0, alternative='greater')
            sharpe = (avg_ret / std_ret) * np.sqrt(len(all_rets)) if std_ret > 0 else 0
            
            # 適應度函數：Sharpe Ratio * 統計置信度 (1 - p_value)
            fitness = sharpe * (1 - p_val)
            
            # 顯著性處罰 (p-value > 0.05 視為隨機)
            if p_val > 0.05:
                fitness *= 0.1
                
            return float(fitness)

        # 基因空間
        gene_space = [
            range(10, 151, 10),            # VWAP Window
            np.linspace(0.001, 0.02, 20),  # Buy Threshold
            np.linspace(0.001, 0.02, 20)   # Sell Threshold
        ]

        ga_instance = pygad.GA(
            num_generations=num_gen,
            num_parents_mating=int(pop_size/2),
            fitness_func=fitness_func,
            sol_per_pop=pop_size,
            num_genes=3,
            gene_space=gene_space,
            parent_selection_type="rank",
            crossover_type="uniform",
            mutation_type="random",
            mutation_probability=0.1,
            on_generation=lambda ga: progress_bar.progress(ga.generations_completed / num_gen)
        )

        with st.spinner("遺傳算法計算中..."):
            ga_instance.run()

        # --- 4. 結果展示 ---
        solution, sol_fitness, _ = ga_instance.best_solution()
        
        st.markdown("### 🏆 最佳參數結果")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VWAP Window", f"{int(solution[0])}")
        c2.metric("Buy Threshold", f"{solution[1]:.4%}")
        c3.metric("Sell Threshold", f"{solution[2]:.4%}")
        c4.metric("Fitness Score", f"{sol_fitness:.4f}")

        # 歷史收斂圖
        st.subheader("📈 優化收斂曲線")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=ga_instance.best_solutions_fitness, mode='lines+markers', line=dict(color='#00ffcc')))
        fig.update_layout(template="plotly_dark", xaxis_title="Generation", yaxis_title="Fitness")
        st.plotly_chart(fig, use_container_width=True)

        st.warning("⚠️ 注意：此結果基於歷史數據。高統計顯著性 (p < 0.05) 雖降低了隨機性，但仍需注意市場結構轉變風險。")
    else:
        st.error("無法獲取數據，請檢查網絡或股票代碼。")
