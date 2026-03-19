import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pygad
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- 1. 數據獲取模塊 (參考來源 [1]) ---
@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    """下載指定股票數據"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"數據下載失敗: {e}")
        return None

# --- 2. 交易回測邏輯 (參考來源 [2, 4]) ---
def run_backtest(data, short_window, long_window):
    """
    執行簡單的均線交叉策略
    short_window: 短期均線天數 (基因1)
    long_window: 長期均線天數 (基因2)
    """
    df = data.copy()
    # 確保窗口為整數且短期小於長期
    short_window = int(short_window)
    long_window = int(long_window)
    
    if short_window >= long_window:
        return -100 # 不合理的參數給予懲罰
        
    df['SMA_S'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_L'] = df['Close'].rolling(window=long_window).mean()
    
    # 產生信號
    df['Signal'] = 0.0
    df.loc[df.index[short_window:], 'Signal'] = np.where(
        df['SMA_S'][short_window:] > df['SMA_L'][short_window:], 1.0, 0.0
    )
    df['Position'] = df['Signal'].diff()
    
    # 計算收益 (假設初始資金 10000)
    df['Market_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Market_Return'] * df['Signal'].shift(1)
    
    # 計算累計收益
    cumulative_return = (1 + df['Strategy_Return'].fillna(0)).cumprod()
    total_return = cumulative_return.iloc[-1] - 1
    
    # 計算最大回撤 (來源 [2] 強調考慮風險)
    peak = cumulative_return.expanding(min_periods=1).max()
    drawdown = (cumulative_return/peak) - 1
    max_drawdown = drawdown.min()
    
    # 適應度函數：不僅看收益，還看風險 (收益 / |最大回撤|)
    # 來源 [3] 提到捕捉低效市場時需穩健性
    fitness = total_return / (abs(max_drawdown) + 0.01) 
    
    return fitness, cumulative_return, total_return, max_drawdown

# --- 3. 遺傳算法設置 (參考來源 [2]) ---
def fitness_func(ga_instance, solution, solution_idx):
    """遺傳算法的適應度評估函數"""
    short_w, long_w = solution
    # 使用全域變數中的數據進行回測
    fitness_value, _, _, _ = run_backtest(st.session_state.market_data, short_w, long_w)
    return fitness_value

# --- 4. Streamlit UI 界面 ---
st.title("🧬 AI 遺傳算法自動構建交易策略")
st.write("結合 DeepSeek 思路與遺傳算法，自動搜索最優策略參數 [2, 4]")

# 側邊欄配置
st.sidebar.header("1. 數據設定")
ticker = st.sidebar.text_input("股票代碼 (e.g. AAPL, TSLA)", "AAPL")
days = st.sidebar.slider("回測天數", 365, 1825, 730)

st.sidebar.header("2. 遺傳算法參數")
num_generations = st.sidebar.slider("進化代數 (Generations)", 10, 100, 20)
sol_per_pop = st.sidebar.slider("種群大小 (Population Size)", 10, 50, 20)

# 執行流程
end_date = datetime.now()
start_date = end_date - timedelta(days=days)

if st.sidebar.button("開始進化策略"):
    with st.spinner('正在下載數據並執行遺傳進化...'):
        data = fetch_stock_data(ticker, start_date, end_date)
        
        if data is not None:
            st.session_state.market_data = data
            
            # 初始化 PyGAD
            # 基因空間：短期均線 (5-50), 長期均線 (50-200)
            ga_instance = pygad.GA(
                num_generations=num_generations,
                num_parents_mating=5,
                fitness_func=fitness_func,
                sol_per_pop=sol_per_pop,
                num_genes=2,
                gene_space=[range(5, 51), range(51, 201)],
                mutation_percent_genes=10
            )
            
            # 執行進化
            ga_instance.run()
            
            # 獲取最佳結果
            solution, solution_fitness, _ = ga_instance.best_solution()
            best_short, best_long = solution
            
            # 重新運行回測以獲取圖表數據
            _, final_curve, final_ret, final_mdd = run_backtest(data, best_short, best_long)
            
            # --- 顯示結果 ---
            st.success("🎉 進化完成！")
            col1, col2, col3 = st.columns(3)
            col1.metric("最佳短期均線", f"{int(best_short)} 天")
            col2.metric("最佳長期均線", f"{int(best_long)} 天")
            col3.metric("適應度評分", f"{solution_fitness:.2f}")
            
            st.metric("累計收益率", f"{final_ret*100:.2f}%")
            st.metric("最大回撤 (Risk)", f"{final_mdd*100:.2f}%")
            
            # 繪製曲線圖
            st.subheader("策略收益曲線")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(final_curve, label='Strategy (GA Optimized)', color='orange')
            ax.plot((1 + data['Close'].pct_change().fillna(0)).cumprod(), label='Market (Buy & Hold)', alpha=0.5)
            ax.legend()
            st.pyplot(fig)
            
            # 顯示進化歷史 (收斂情況)
            st.subheader("遺傳算法收斂歷史")
            fig_fitness = ga_instance.plot_fitness()
            st.pyplot(fig_fitness)
            
        else:
            st.error("無法獲取數據，請檢查代碼或網路連接。")
else:
    st.info("👈 請在左側設定參數並點擊『開始進化策略』")
