import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import time
import random
import urllib3
import sys

# --- 強制設定系統環境為 UTF-8 ---
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="GPT-4 量化混合專家 v28.0", layout="wide")

# ==================== 1. 金鑰讀取與安全初始化 ====================

def get_gspread_client():
    """修復 PEM 與編碼問題"""
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    if "gcp_service_account" in st.secrets:
        try:
            creds_info = dict(st.secrets["gcp_service_account"])
            # 深度清洗私鑰字串
            pk = str(creds_info["private_key"]).strip().strip('"').strip("'")
            pk = pk.replace("\\n", "\n")
            creds_info["private_key"] = pk
            
            creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
            return gspread.authorize(creds)
        except Exception as e:
            st.error(f"❌ Sheets 授權失敗: {e}")
            return None
    return None

# 初始化 OpenAI (確保 API Key 存在)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
ai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ==================== 2. 功能模組 ====================

def get_market_indicators(ticker, df):
    """計算技術面數據"""
    try:
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        trend = "黃金交叉(偏多)" if ma5 > ma20 else "死亡交叉(偏空)"
        pe = yf.Ticker(ticker).info.get('forwardPE', 'N/A')
        return f"趨勢:{trend}, PE:{pe}"
    except: return "分析失敗"

def fetch_news_summary(symbol):
    """新聞爬蟲"""
    stock_id = symbol.split('.')[0]
    try:
        url = "https://news.cnyes.com/news/cat/tw_stock_news"
        res = requests.get(url, timeout=5)
        res.encoding = 'utf-8' # 強制編碼
        soup = BeautifulSoup(res.text, 'html.parser')
        titles = [t.get_text().strip() for t in soup.find_all(['h3', 'a']) if stock_id in t.get_text()]
        return " | ".join(titles[:5])
    except: return "無相關新聞"

# ==================== 3. 主執行流程 ====================

st.title("🏆 GPT-4o 量化混合預測系統 v28.0")

if st.button("🚀 啟動全方位分析 (A-K 欄位自動同步)"):
    if not ai_client:
        st.error("❌ OpenAI API Key 缺失。")
        st.stop()

    gc = get_gspread_client()
    if gc:
        try:
            sh = gc.open("Stock_Predictions_History")
            ws = sh.get_worksheet(0)
            
            # --- 1. 自動建立 A-K 標題列 ---
            # 欄位定義：日期(A), 代號(B), 現價(C), 指標(D), 預測1-5(E-I), 情緒(J), 摘要(K)
            headers = ["日期", "股票代號", "收盤價格", "交易值指標", "5日預測-1", "5日預測-2", "5日預測-3", "5日預測-4", "5日預測-5", "AI情緒分", "AI短評摘要"]
            if not ws.row_values(1):
                ws.insert_row(headers, 1)
                st.info("✅ 已自動初始化 A-K 欄位標題。")

            # 2. 讀取 B 欄股票代號
            raw_data = ws.get_all_values()
            if len(raw_data) <= 1:
                st.warning("請先在 B 欄填入股票代號。")
                st.stop()

            # 使用 DataFrame 整理數據
            df_sheet = pd.DataFrame(raw_data[1:], columns=raw_data[0])
            tickers = df_sheet['股票代號'].dropna().astype(str).head(100).tolist()
            
            p_bar = st.progress(0)
            status = st.empty()
            
            # 批量下載數據
            all_hist = yf.download(tickers, period="3mo", group_by='ticker', progress=False)

            for idx, t in enumerate(tickers):
                try:
                    status.text(f"GPT-4 分析處理中 ({idx+1}/{len(tickers)}): {t}")
                    df = all_hist[t].dropna() if isinstance(all_hist.columns, pd.MultiIndex) else all_hist.dropna()
                    if df.empty: continue
                    
                    curr_p = round(float(df['Close'].iloc[-1]), 2)
                    tech_info = get_market_indicators(t, df)
                    news_info = fetch_news_summary(t)
                    
                    # 3. GPT-4o 混合分析
                    response = ai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "你是一位精通繁體中文的台股分析師，請確保回傳內容無亂碼。"},
                            {"role": "user", "content": f"標的:{t},現價:{curr_p},數據:{tech_info},新聞:{news_info}。請回傳:情緒分(-5到5),5日預測價(5個),30字摘要。格式:分數,價1,價2,價3,價4,價5,摘要"}
                        ]
                    )
                    
                    # 4. 資料清洗與寫入
                    res = response.choices[0].message.content.strip().split(',')
                    # 強制轉為字串並過濾非 ASCII 字符導致的寫入問題
                    sentiment_score = str(res[0])
                    prediction_prices = res[1:6]
                    ai_summary = str(res[6]) if len(res) > 6 else "分析完成"
                    
                    # 寫入 E-K 欄 (第5到第11欄)
                    final_data = prediction_prices + [f"GPT:{sentiment_score}", ai_summary]
                    ws.update(f"E{idx+2}:K{idx+2}", [final_data])
                    
                    time.sleep(1.0)
                except Exception as e:
                    st.warning(f"跳過 {t}: {e}")
                p_bar.progress((idx + 1) / len(tickers))
                
            st.success("🎉 全方位分析完成，Excel 數據已更新！")
        except Exception as e:
            st.error(f"操作失敗: {e}")
