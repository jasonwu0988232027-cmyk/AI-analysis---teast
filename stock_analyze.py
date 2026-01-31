import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import time
import os
import random
import urllib3

# --- 基礎配置 ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="AI 股市情緒分析系統 v21.0", layout="wide")

# --- 金鑰與參數 ---
SHEET_NAME = "Stock_Predictions_History"
CREDENTIALS_JSON = "eco-precept-485904-j5-7ef3cdda1b03.json"
# 預設使用您提供的 Gemini API KEY
DEFAULT_GEMINI_KEY = "AIzaSyDE4yDZMnniFaYLQd-LK7WSQpHh-6JRA3Q"
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", DEFAULT_GEMINI_KEY)

# ==================== 1. AI 模型初始化 (修正 404 問題) ====================

def init_ai_engine():
    """動態偵測並初始化 Gemini 模型，確保路徑正確"""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # 嘗試最相容的模型路徑
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"AI 引擎啟動失敗: {e}")
        return None

ai_engine = init_ai_engine()

# ==================== 2. 雲端連線與爬蟲模組 ====================

def get_gspread_client():
    """建立授權連線並修正私鑰格式"""
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    try:
        if "gcp_service_account" in st.secrets:
            creds_info = dict(st.secrets["gcp_service_account"])
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
            creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        elif os.path.exists(CREDENTIALS_JSON):
            creds = Credentials.from_service_account_file(CREDENTIALS_JSON, scopes=scopes)
        else: return None
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Sheets 授權失敗: {e}")
        return None

def scrape_stock_news(symbol):
    """針對特定代碼爬取新聞標題與摘要"""
    stock_id = symbol.split('.')[0]
    headers = {'User-Agent': 'Mozilla/5.0'}
    # 爬取鉅亨網與經濟日報之即時新聞
    url = f"https://news.cnyes.com/news/cat/tw_stock_news"
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        # 擷取包含代碼或相關字眼的標題
        titles = [t.get_text() for t in soup.find_all(['h3', 'a']) if stock_id in t.get_text()]
        return " ".join(titles[:8]) if titles else "目前查無重大相關新聞。"
    except:
        return "新聞獲取失敗。"

# ==================== 3. 主執行程序 ====================

st.title("🤖 AI 股市情緒預測系統 (Gemini 驅動)")

if st.button("🚀 執行近5日 Top 100 標的情緒分析"):
    if not ai_engine:
        st.error("AI 引擎未就緒。")
        st.stop()
        
    client = get_gspread_client()
    if client:
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        
        # 讀取 Excel A-D 欄資料
        raw_values = ws.get_all_values()
        if len(raw_values) <= 1:
            st.warning("表格內無資料。")
            st.stop()
            
        df_sheet = pd.DataFrame(raw_values[1:], columns=raw_values[0])
        
        # 篩選最近 5 日的資料
        df_sheet['日期'] = pd.to_datetime(df_sheet['日期'])
        five_days_ago = datetime.now() - timedelta(days=5)
        df_recent = df_sheet[df_sheet['日期'] >= five_days_ago].head(100)
        
        tickers = df_recent['股票代號'].tolist()
        st.info(f"偵測到 {len(tickers)} 檔近 5 日熱門標的，開始 AI 情緒掃描...")
        
        p_bar = st.progress(0)
        
        for idx, t in enumerate(tickers):
            try:
                # 1. 抓取新聞
                news_content = scrape_stock_news(t)
                
                # 2. 呼叫 AI 進行情緒評分
                # 提示詞設計：要求結構化輸出以利預測運算
                prompt = f"""
                你是資深台股分析師。請分析股票 {t} 的以下新聞：
                ---
                {news_content}
                ---
                請給出 -5 (極度利空) 到 5 (極度利多) 的情緒評分。
                並根據此評分與新聞，給出未來 5 個交易日的預測收盤價。
                格式要求：分數,價1,價2,價3,價4,價5 (僅回答數字與逗號)
                """
                response = ai_engine.generate_content(prompt)
                ai_output = response.text.strip().split(',')
                
                # 解析數據
                sentiment_score = float(ai_output[0])
                pred_prices = [float(p) for p in ai_output[1:6]]
                
                # 3. 寫入 Excel E-J 欄
                # E-I: 預測價, J: 分數(或誤差)
                ws.update(f"E{idx+2}:J{idx+2}", [pred_prices + [f"AI分:{sentiment_score}"]])
                
                st.write(f"✅ {t} 分析完成 | 情緒分: {sentiment_score}")
                
                # 智能延遲預防 API 封鎖
                time.sleep(random.uniform(1.5, 3.0))
                if (idx + 1) % 10 == 0:
                    time.sleep(15)
                    
            except Exception as e:
                st.warning(f"跳過 {t}: {e}")
                
            p_bar.progress((idx + 1) / len(tickers))
            
        st.success("🎉 AI 情緒分析與 5 日預測已全數同步至 Excel！")
