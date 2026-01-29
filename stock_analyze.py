import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

# --- 頁面配置（必須在最前面）---
st.set_page_config(page_title="AI 股市預測專家 Pro", layout="wide", initial_sidebar_state="expanded")

# 隱藏不必要的警告
warnings.filterwarnings('ignore')

# 嘗試導入機器學習與技術分析套件
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# --- API 設定 ---
FINNHUB_API_KEY = "d5t2rvhr01qt62ngu1kgd5t2rvhr01qt62ngu1l0"

# ==================== 1. 數據獲取模組 ====================

@st.cache_data(ttl=3600)
def get_stock_data(symbol, period="1y"):
    """獲取歷史股價數據"""
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False, timeout=10)
        if df.empty: return None
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df.reset_index()
    except Exception as e:
        st.error(f"數據獲取失敗: {str(e)}")
        return None

@st.cache_data(ttl=86400)
def get_fundamental_data(symbol):
    """獲取基本面數據"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if not info: return None
        return {
            'PE Ratio': info.get('trailingPE', 'N/A'),
            'ROE': info.get('returnOnEquity', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            'Industry': info.get('industry', '未知')
        }
    except:
        return None

# ==================== 2. 技術指標計算 ====================

def calculate_indicators(df):
    """計算 RSI, MACD, 布林通道, KD (支援 ta 套件或手動計算)"""
    d = df.copy()
    if TA_AVAILABLE:
        # 使用專業 ta 套件
        d['RSI'] = ta.momentum.RSIIndicator(d['Close']).rsi()
        macd = ta.trend.MACD(d['Close'])
        d['MACD_Diff'] = macd.macd_diff()
        bollinger = ta.volatility.BollingerBands(d['Close'])
        d['BB_High'], d['BB_Low'] = bollinger.bollinger_hband(), bollinger.bollinger_lband()
        stoch = ta.momentum.StochasticOscillator(d['High'], d['Low'], d['Close'])
        d['K'], d['D'] = stoch.stoch(), stoch.stoch_signal()
        d['SMA_20'] = ta.trend.SMAIndicator(d['Close'], window=20).sma_indicator()
        d['SMA_50'] = ta.trend.SMAIndicator(d['Close'], window=50).sma_indicator()
    else:
        # 手動計算邏輯 (備援)
        delta = d['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        d['RSI'] = 100 - (100 / (1 + (gain / loss)))
        d['SMA_20'] = d['Close'].rolling(20).mean()
        d['SMA_50'] = d['Close'].rolling(50).mean()
        std = d['Close'].rolling(20).std()
        d['BB_High'], d['BB_Low'] = d['SMA_20'] + (std * 2), d['SMA_20'] - (std * 2)
        d['MACD_Diff'] = d['Close'].ewm(span=12).mean() - d['Close'].ewm(span=26).mean()
        low_14, high_14 = d['Low'].rolling(14).min(), d['High'].rolling(14).max()
        d['K'] = 100 * ((d['Close'] - low_14) / (high_14 - low_14))
        d['D'] = d['K'].rolling(3).mean()
    
    return d.bfill().ffill()

# ==================== 3. AI 預測模型 (LSTM) ====================

@st.cache_resource
def train_lstm_model(df, epochs=50):
    """訓練 LSTM 並修正 Retracing 警告"""
    if not TF_AVAILABLE or not SKLEARN_AVAILABLE:
        return None, None, None
    
    # 關鍵：清除 Keras 舊 Session 防止警告與記憶體洩漏
    tf.keras.backend.clear_session()
    
    lookback = 60
    data = df[['Close']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    
    return model, scaler, lookback

# ==================== 4. 主介面邏輯 ====================

def main():
    st.title("📈 AI 股市全方位預測系統 Pro")
    
    # 側邊欄
    st.sidebar.header("🔍 參數設定")
    symbol = st.sidebar.text_input("股票代碼", "2330.TW").upper()
    forecast_days = st.sidebar.slider("預測天數", 5, 14, 7)
    use_lstm = st.sidebar.toggle("啟用 LSTM 深度學習預測", value=True)
    show_fundamentals = st.sidebar.toggle("顯示公司基本面", value=False)

    with st.spinner('數據計算中...'):
        df_raw = get_stock_data(symbol)
        if df_raw is None:
            st.error("找不到股票數據，請檢查代碼。")
            return

        df = calculate_indicators(df_raw)
        
        # --- 預測邏輯 ---
        if use_lstm and TF_AVAILABLE:
            model, scaler, lookback = train_lstm_model(df)
            last_60 = df[['Close']].tail(lookback).values
            scaled_last = scaler.transform(last_60)
            
            preds = []
            curr_seq = scaled_last.reshape(1, lookback, 1)
            for _ in range(forecast_days):
                p = model.predict(curr_seq, verbose=0)
                preds.append(p[0,0])
                curr_seq = np.append(curr_seq[:,1:,:], p.reshape(1,1,1), axis=1)
            
            future_prices = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
        else:
            # 隨機漫步回歸預測 (備援)
            last_p = df['Close'].iloc[-1]
            future_prices = [last_p * (1 + np.random.normal(0, 0.01)) for _ in range(forecast_days)]

        # --- 繪圖 (修正 width='stretch') ---
        fig = go.Figure()
        d_plot = df.tail(100)
        fig.add_trace(go.Candlestick(x=d_plot['Date'], open=d_plot['Open'], high=d_plot['High'], low=d_plot['Low'], close=d_plot['Close'], name="歷史K線"))
        
        # 預測線
        pred_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
        fig.add_trace(go.Scatter(x=[df['Date'].iloc[-1]] + pred_dates, y=[df['Close'].iloc[-1]] + list(future_prices), 
                                 line=dict(color='orange', width=3, dash='dot'), name="AI 預測趨勢"))
        
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width='stretch') # 修正警告：使用 width='stretch'

        # --- 指標面板 ---
        c1, c2, c3 = st.columns(3)
        curr_p = df['Close'].iloc[-1]
        targ_p = future_prices[-1]
        c1.metric("當前價格", f"${curr_p:.2f}")
        c2.metric(f"{forecast_days}日預測", f"${targ_p:.2f}", f"{((targ_p-curr_p)/curr_p)*100:+.2f}%")
        c3.metric("RSI 強弱指標", f"{df['RSI'].iloc[-1]:.1f}")

        # --- 基本面 ---
        if show_fundamentals:
            st.divider()
            f_data = get_fundamental_data(symbol)
            if f_data:
                st.subheader(f"💼 {symbol} 公司概況 - {f_data['Industry']}")
                m1, m2, m3 = st.columns(3)
                m1.write(f"**本益比 (PE):** {f_data['PE Ratio']}")
                m2.write(f"**ROE:** {f_data['ROE']}")
                m3.write(f"**殖利率:** {f_data['Dividend Yield']}")
            else:
                st.warning("暫時無法取得基本面數據。")

if __name__ == "__main__":
    main()