import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pygad
from scipy import stats
import plotly.graph_objects as go
from datetime import datetime

# --- 1. 量化回測引擎 ---
class QuantGAEngine:
    def __init__(self, tickers, period="1mo", interval="5m"):
        self.tickers = tickers
        self.period = period
        self.interval = interval
        self.data_pool = {}

    def download_data(self):
        """修復版：處理 yfinance 的 MultiIndex 結構與單/多標的兼容性"""
        st.info(f"正在獲取數據：{', '.join(self.tickers)} (時段: {self.period})")
        
        try:
            # 5m/15m 數據在 yfinance 有長度限制，若 period 太長會報錯，這裡做基本防禦
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
            for ticker in self.tickers:
                # 處理多標的 (MultiIndex) 與單標的 (DataFrame) 的差異
                if len(self.tickers) > 1:
                    if ticker in raw_data.columns.get_level_values(0):
                        df = raw_data[ticker].copy()
                    else:
                        continue
                else:
                    df = raw_data.copy()

                df = df.dropna()
                if not df.empty and len(df) > 50:
                    # 預計算 Typical Price
                    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
                    self.data_pool[ticker] = df
                    valid_count += 1
            
            return valid_count
            
        except Exception as e:
            st.error(f"❌ 數據下載異常: {str(e)}")
            return 0

    @staticmethod
    def backtest_logic(df, window, buy_t, sell_t, commission, slippage):
        """核心策略：VWAP 偏離回歸 + 交易成本模型"""
        window = int(window)
        # 計算 Rolling VWAP
        tp_v = (df['TP'] * df['Volume']).rolling(window=window).sum()
        v_sum = df['Volume'].rolling(window=window).sum()
        vwap = tp_v / v_sum
        
        # 偏離率
        dist = (df['Close'] - vwap) / vwap
        
        returns = []
        pos = 0 # 0: 空倉, 1: 持倉
        entry_price = 0
        
        prices = df['Close'].values
        dists = dist.values
        
        # 從 window 之後開始跑，避免 NaN
        for i in range(window, len(df)):
            # 買入信號 (考慮買入滑點)
            if pos == 0 and dists[i] < -buy_t:
                pos = 1
                entry_price = prices[i] * (1 + slippage)
            
            # 賣出信號 (考慮賣出滑點 + 收盤強制平倉)
            elif pos == 1 and (dists[i] > sell_t or i == len(df)-1):
                exit_price = prices[i] * (1 - slippage)
                # 淨收益計算 (扣除雙邊手續費)
                net_ret = ((exit_price - entry_price) / entry_price) - (commission * 2)
                returns.append(net_ret)
                pos = 0
                
        return returns

# --- 2. Streamlit 介面渲染 ---
st.set_page_config(page_title="VWAP GA Optimizer V1.1", layout="wide")
st.title("🛡️ 專業級遺傳算法策略優化器")
st.caption("版本 V1.1 | 整合統計顯著性檢驗與交易摩擦成本")

with st.sidebar:
    st.header("🛠️ 系統配置")
    tickers_input = st.text_input("股票代碼 (例如: AAPL,TSLA,NVDA)", value="AAPL,TSLA,NVDA")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    st.subheader("1. 數據設定")
    period = st.selectbox("回測時長", ["1mo", "60d", "max"], index=0)
    interval = st.selectbox("K線頻率", ["5m", "15m", "30m", "1h"], index=0)
    
    st.subheader("2. 模擬成本")
    comm = st.number_input("單邊手續費 (%)", value=0.03, format="%.3f") / 100
    slip = st.number_input("單邊滑點 (%)", value=0.01, format="%.3f") / 100
    
    st.subheader("3. 進化參數")
    num_gen = st.slider("演化代數", 5, 50, 20)
    pop_size = st.slider("種群大小", 10, 100, 30)

# --- 3. 執行優化工作流 ---
if st.button("🔥 啟動全域優化引擎"):
    if not tickers:
        st.warning("請輸入至少一個股票代碼。")
    else:
        engine = QuantGAEngine(tickers, period, interval)
        num_ready = engine.download_data()
        
        if num_ready > 0:
            st.success(f"✅ 成功加載 {num_ready} 隻股票標的。")
            progress_bar = st.progress(0)
            
            # 定義 GA 適應度函數
            def fitness_func(ga_instance, solution, solution_idx):
                win, b_t, s_t = solution
                all_rets = []
                
                # 計算股票池總收益樣本
                for t in engine.data_pool:
                    rets = engine.backtest_logic(engine.data_pool[t], win, b_t, s_t, comm, slip)
                    all_rets.extend(rets)
                
                # 樣本量過低懲罰
                if len(all_rets) < 20:
                    return -5.0
                
                # 統計指標計算
                avg_ret = np.mean(all_rets)
                std_ret = np.std(all_rets)
                t_stat, p_val = stats.ttest_1samp(all_rets, 0, alternative='greater')
                sharpe = (avg_ret / std_ret) * np.sqrt(len(all_rets)) if std_ret > 0 else 0
                
                # 適應度公式：Sharpe * (1 - p-value)
                fitness = sharpe * (1 - p_val)
                
                # 統計顯著性過濾 (Alpha = 0.05)
                if p_val > 0.05:
                    fitness *= 0.1 # 非顯著獲利給予重罰
                    
                return float(fitness)

            # 參數空間 [Window, Buy_Thres, Sell_Thres]
            gene_space = [
                range(10, 151, 10),
                np.linspace(0.001, 0.02, 20),
                np.linspace(0.001, 0.02, 20)
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
                mutation_probability=0.1,
                on_generation=lambda ga: progress_bar.progress(ga.generations_completed / num_gen)
            )

            with st.spinner("🧠 進化計算中... 這可能需要一點時間"):
                ga_instance.run()

            # --- 4. 結果展示 ---
            solution, sol_fitness, _ = ga_instance.best_solution()
            
            st.markdown("---")
            st.header("🎯 最佳參數組合")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("VWAP Window", f"{int(solution[0])}")
            col2.metric("買入閾值", f"{solution[1]:.3%}")
            col3.metric("賣出閾值", f"{solution[2]:.3%}")
            col4.metric("最終適應度", f"{sol_fitness:.4f}")

            # 繪製收斂曲線
            st.subheader("📈 策略演化進程 (Evolution History)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=ga_instance.best_solutions_fitness, mode='lines+markers', name='Fitness Score'))
            fig.update_layout(template="plotly_dark", xaxis_title="Generation", yaxis_title="Fitness")
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("🏁 優化完成！建議將此參數應用於樣本外（Out-of-Sample）數據進行最終驗證。")
        else:
            st.error("❌ 數據池準備失敗，請檢查網路連接或 Tickers 名稱。")
