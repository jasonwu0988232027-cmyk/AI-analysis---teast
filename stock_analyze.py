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
        self.tickers = [t.strip().upper() for t in tickers]
        self.period = period
        self.interval = interval
        self.data_pool = {}

    def download_data(self):
        """V1.2 終極修復：自動偵測並強制歸一化單/多標的數據結構"""
        st.info(f"正在獲取數據：{', '.join(self.tickers)}...")
        
        try:
            # 執行下載 (不強制 group_by，交由後續邏輯判斷結構)
            raw_data = yf.download(
                tickers=self.tickers, 
                period=self.period, 
                interval=self.interval, 
                auto_adjust=True, 
                threads=True
            )
            
            if raw_data.empty:
                st.error("❌ 下載結果為空，請檢查網絡或標的代碼。")
                return 0

            valid_count = 0
            
            for ticker in self.tickers:
                df = None
                
                # 結構 A: 多層索引 (MultiIndex) - 當下載多支股票時
                if isinstance(raw_data.columns, pd.MultiIndex):
                    if ticker in raw_data.columns.get_level_values(0):
                        df = raw_data[ticker].copy()
                
                # 結構 B: 扁平索引 - 當下載單支股票時 yfinance 的行為
                else:
                    if len(self.tickers) == 1:
                        df = raw_data.copy()
                
                # 數據清洗與欄位標準化
                if df is not None and not df.empty:
                    df = df.dropna()
                    
                    # 檢查核心價格欄位是否存在 (處理 yfinance 可能返回的欄位名稱差異)
                    required = ['High', 'Low', 'Close', 'Volume']
                    if all(col in df.columns for col in required):
                        # 預計算 Typical Price
                        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
                        self.data_pool[ticker] = df
                        valid_count += 1
                    else:
                        st.warning(f"⚠️ {ticker} 數據格式不全，缺少必要價格欄位。")
            
            return valid_count
            
        except Exception as e:
            st.error(f"❌ 嚴重異常: {str(e)}")
            return 0

    @staticmethod
    def backtest_logic(df, window, buy_t, sell_t, commission, slippage):
        """核心策略：VWAP 偏離回歸"""
        window = int(window)
        # 滾動計算 VWAP
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
            # 買入 (考慮滑點)
            if pos == 0 and dists[i] < -buy_t:
                pos = 1
                entry_price = prices[i] * (1 + slippage)
            # 賣出 (考慮滑點 + 收盤強平)
            elif pos == 1 and (dists[i] > sell_t or i == len(df)-1):
                exit_price = prices[i] * (1 - slippage)
                net_ret = ((exit_price - entry_price) / entry_price) - (commission * 2)
                returns.append(net_ret)
                pos = 0
        return returns

# --- 2. Streamlit 介面 ---
st.set_page_config(page_title="VWAP GA Optimizer V1.2", layout="wide")
st.title("🛡️ 專業級遺傳算法策略優化器")
st.caption("版本 V1.2 | 已修復單/多標的 Key Error | 整合統計顯著性與交易成本")

with st.sidebar:
    st.header("🛠️ 系統配置")
    tickers_input = st.text_input("股票代碼 (例: 2330.TW, AAPL)", value="2330.TW")
    # 清理輸入字符
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

# --- 3. 優化執行 ---
if st.button("🔥 啟動全域優化引擎"):
    if not tickers:
        st.warning("請先輸入股票代碼。")
    else:
        engine = QuantGAEngine(tickers, period, interval)
        num_ready = engine.download_data()
        
        if num_ready > 0:
            st.success(f"✅ 成功加載 {num_ready} 隻股票。進化算法啟動...")
            progress_bar = st.progress(0)
            
            def fitness_func(ga_instance, solution, solution_idx):
                win, b_t, s_t = solution
                all_rets = []
                for t in engine.data_pool:
                    rets = engine.backtest_logic(engine.data_pool[t], win, b_t, s_t, comm, slip)
                    all_rets.extend(rets)
                
                # 樣本量過低懲罰
                if len(all_rets) < 20: return -5.0
                
                avg_ret = np.mean(all_rets)
                std_ret = np.std(all_rets)
                # t-檢驗：測試收益是否顯著 > 0
                t_stat, p_val = stats.ttest_1samp(all_rets, 0, alternative='greater')
                sharpe = (avg_ret / std_ret) * np.sqrt(len(all_rets)) if std_ret > 0 else 0
                
                # 適應度公式：Sharpe * 信心指數
                fitness = sharpe * (1 - p_val)
                if p_val > 0.05: fitness *= 0.1 # 懲罰不顯著的獲利
                return float(fitness)

            # 搜尋空間
            gene_space = [
                range(10, 151, 10),            # Window
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
                on_generation=lambda ga: progress_bar.progress(ga.generations_completed / num_gen)
            )

            with st.spinner("🧠 正在進行全域參數演化搜尋..."):
                ga_instance.run()

            # --- 4. 結果展示 ---
            solution, sol_fitness, _ = ga_instance.best_solution()
            st.markdown("---")
            st.header("🎯 最佳統計參數組合")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("VWAP Window", f"{int(solution[0])}")
            c2.metric("買入閾值", f"{solution[1]:.3%}")
            c3.metric("賣出閾值", f"{solution[2]:.3%}")
            c4.metric("適應度得分", f"{sol_fitness:.4f}")

            # 進化歷史圖表
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=ga_instance.best_solutions_fitness, mode='lines+markers', name='Fitness Score'))
            fig.update_layout(template="plotly_dark", title="演化歷史曲線", xaxis_title="Generation", yaxis_title="Fitness")
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("🏁 優化完成！")
        else:
            st.error("❌ 無法讀取有效的數據，請檢查代碼或網路。")
