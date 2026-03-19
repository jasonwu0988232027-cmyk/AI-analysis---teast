import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pygad
from scipy import stats
import plotly.graph_objects as go

# --- 1. 量化回測引擎 ---
class QuantGAEngine:
    def __init__(self, tickers, period="1mo", interval="5m"):
        self.tickers = tickers
        self.period = period
        self.interval = interval
        self.data_pool = {}

    def download_data(self):
        """修復版：精確處理 yfinance 返回的單/多標的結構差異"""
        st.info(f"正在獲取數據：{', '.join(self.tickers)} (時段: {self.period})")
        
        try:
            # yfinance 批量下載，auto_adjust=True 處理除權息
            raw_data = yf.download(
                tickers=self.tickers, 
                period=self.period, 
                interval=self.interval, 
                group_by='ticker', 
                auto_adjust=True, 
                threads=True
            )
            
            if raw_data.empty:
                st.error("❌ 下載結果為空，請檢查代碼或該時段是否有交易數據。")
                return 0

            valid_count = 0
            
            # 判斷是多標的還是單標的
            if len(self.tickers) > 1:
                # 情況 A: 多標的 (MultiIndex 結構)
                returned_tickers = raw_data.columns.get_level_values(0).unique()
                for ticker in self.tickers:
                    if ticker in returned_tickers:
                        df = raw_data[ticker].copy().dropna()
                        if not df.empty and len(df) > 50:
                            df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
                            self.data_pool[ticker] = df
                            valid_count += 1
            else:
                # 情況 B: 單標的 (Flat 結構)
                ticker = self.tickers[0]
                df = raw_data.copy().dropna()
                if not df.empty and len(df) > 50:
                    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
                    self.data_pool[ticker] = df
                    valid_count += 1
            
            return valid_count
            
        except Exception as e:
            st.error(f"❌ 數據下載或處理異常: {str(e)}")
            return 0

    @staticmethod
    def backtest_logic(df, window, buy_t, sell_t, commission, slippage):
        """核心策略：VWAP 偏離回歸 + 交易摩擦模型"""
        window = int(window)
        # 計算 Rolling VWAP
        tp_v = (df['TP'] * df['Volume']).rolling(window=window).sum()
        v_sum = df['Volume'].rolling(window=window).sum()
        vwap = tp_v / v_sum
        
        # 偏離率 (Price / VWAP - 1)
        dist = (df['Close'] - vwap) / vwap
        
        returns = []
        pos = 0 # 0: 空倉, 1: 持倉
        entry_price = 0
        
        prices = df['Close'].values
        dists = dist.values
        
        for i in range(window, len(df)):
            # 買入 (考慮滑點：買貴)
            if pos == 0 and dists[i] < -buy_t:
                pos = 1
                entry_price = prices[i] * (1 + slippage)
            # 賣出 (考慮滑點：賣便宜 + 收盤強平)
            elif pos == 1 and (dists[i] > sell_t or i == len(df)-1):
                exit_price = prices[i] * (1 - slippage)
                net_ret = ((exit_price - entry_price) / entry_price) - (commission * 2)
                returns.append(net_ret)
                pos = 0
                
        return returns

# --- 2. Streamlit 介面 ---
st.set_page_config(page_title="VWAP GA Optimizer V1.1", layout="wide")
st.title("🛡️ 專業級遺傳算法策略優化器")
st.caption("版本 V1.1 (修復單標的下載 Bug) | 整合統計檢驗與交易成本")

with st.sidebar:
    st.header("🛠️ 系統配置")
    tickers_input = st.text_input("股票代碼 (例: 2330.TW, AAPL)", value="2330.TW")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    st.subheader("1. 數據設定")
    period = st.selectbox("回測時段", ["1mo", "60d", "max"], index=0)
    interval = st.selectbox("K線頻率", ["5m", "15m", "30m", "1h"], index=0)
    
    st.subheader("2. 模擬成本")
    comm = st.number_input("單邊手續費 (%)", value=0.030, format="%.3f") / 100
    slip = st.number_input("單邊滑點 (%)", value=0.010, format="%.3f") / 100
    
    st.subheader("3. 進化參數")
    num_gen = st.slider("演化代數", 5, 50, 20)
    pop_size = st.slider("種群大小", 10, 100, 30)

# --- 3. 執行流程 ---
if st.button("🔥 啟動全域優化引擎"):
    if not tickers:
        st.warning("請輸入股票代碼。")
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
                
                if len(all_rets) < 20: return -5.0
                
                avg_ret = np.mean(all_rets)
                std_ret = np.std(all_rets)
                t_stat, p_val = stats.ttest_1samp(all_rets, 0, alternative='greater')
                sharpe = (avg_ret / std_ret) * np.sqrt(len(all_rets)) if std_ret > 0 else 0
                
                # 適應度公式
                fitness = sharpe * (1 - p_val)
                if p_val > 0.05: fitness *= 0.1
                return float(fitness)

            gene_space = [range(10, 151, 10), np.linspace(0.001, 0.02, 20), np.linspace(0.001, 0.02, 20)]

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

            with st.spinner("🧠 正在進行全域統計搜尋..."):
                ga_instance.run()

            # 結果展示
            solution, sol_fitness, _ = ga_instance.best_solution()
            st.markdown("---")
            st.header("🎯 最佳參數結果")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("VWAP Window", f"{int(solution[0])}")
            c2.metric("買入閾值", f"{solution[1]:.3%}")
            c3.metric("賣出閾值", f"{solution[2]:.3%}")
            c4.metric("最終適應度", f"{sol_fitness:.4f}")

            # 繪製曲線
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=ga_instance.best_solutions_fitness, mode='lines+markers'))
            fig.update_layout(template="plotly_dark", title="進化歷史", xaxis_title="Generation", yaxis_title="Fitness")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ 數據池準備失敗。")
