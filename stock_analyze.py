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
st.set_page_config(page_title="AI 股市情緒專家 v21.5", layout="wide")

# --- 金鑰與參數 ---
SHEET_NAME = "Stock_Predictions_History"
CREDENTIALS_JSON = "eco-precept-485904-j5-7ef3cdda1b03.json"
# 預設 Gemini API KEY
DEFAULT_GEMINI_KEY = "AIzaSyDE4yDZMnniFaYLQd-LK7WSQpHh-6JRA3Q"
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", DEFAULT_GEMINI_KEY)

# ==================== 1. AI 引擎初始化 ====================

def init_ai():
    """初始化 Gemini 並解決路徑識別問題"""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # 2026 年環境下穩定路徑
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"AI 初始化失敗: {e}")
        return None

ai_engine = init_ai()

# ==================== 2. 雲端連線與爬蟲 ====================

def get_gspread_client():
    """建立連線並修正私鑰格式錯誤"""
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

def fetch_deep_news(symbol):
    """搜尋四大新聞網相關報導"""
    stock_id = symbol.split('.')[0]
    headers = {'User-Agent': 'Mozilla/5.0'}
    urls = [
        "https://news.cnyes.com/news/cat/tw_stock_news", # 鉅亨
        "https://money.udn.com/money/index",           # 經濟
        "https://www.ftnn.com.tw/category/6"           # FTNN
    ]
    news_pool = []
    try:
        for url in random.sample(urls, 2): # 隨機抽樣避免封鎖
            res = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            titles = [t.get_text().strip() for t in soup.find_all(['h3', 'a']) if stock_id in t.get_text()]
            news_pool.extend(titles[:4])
    except: pass
    return " | ".join(news_pool) if news_pool else "暫無重大新聞。"

# ==================== 3. 主執行程序 ====================

st.title("🤖 AI 股市深度情緒分析系統")

if st.button("🚀 執行近5日標的情緒分析與預測"):
    if not ai_engine:
        st.error("AI 引擎未就緒，請檢查 API Key。")
        st.stop()
        
    client = get_gspread_client()
    if client:
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        
        # 讀取 Excel 資料並清洗
        raw_rows = ws.get_all_values()
        if len(raw_rows) <= 1:
            st.warning("表格內無資料。")
            st.stop()
            
        df_sheet = pd.DataFrame(raw_rows[1:], columns=raw_rows[0])
        df_sheet['日期'] = pd.to_datetime(df_sheet['日期'])
        
        # 步驟 1：抓取近 5 日交易值前 100 股票
        five_days_ago = datetime.now() - timedelta(days=5)
        df_recent = df_sheet[df_sheet['日期'] >= five_days_ago].head(100)
        
        tickers = df_recent['股票代號'].tolist()
        st.info(f"偵測到 {len(tickers)} 檔熱門標的，開始 AI 分析...")
        
        p_bar = st.progress(0)
        status = st.empty()
        
        for idx, t in enumerate(tickers):
            try:
                status.text(f"分析中 ({idx+1}/{len(tickers)}): {t}")
                
                # 步驟 2：搜尋相關新聞
                news_txt = fetch_deep_news(t)
                
                # 步驟 3：Gemini 情緒評分與預測
                prompt = f"""
                你是資深台股分析師。請針對 {t} 的新聞進行分析：
                新聞內容：{news_txt}
                
                請回傳：
                1. 情緒評分 (-5 到 5)
                2. 未來 5 日預測價格 (5個數值)
                3. 30字以內的利多/利空總結
                
                格式要求：評分,價1,價2,價3,價4,價5,總結
                """
                response = ai_engine.generate_content(prompt)
                ai_output = response.text.strip().split(',')
                
                score = ai_output[0]
                preds = ai_output[1:6]
                summary = ai_output[6] if len(ai_output) > 6 else "分析完畢"
                
                # 寫入 Excel E-K 欄
                # E-I: 預測價, J: 分數, K: 總結
                final_update = preds + [f"AI分:{score}", summary]
                ws.update(f"E{idx+2}:K{idx+2}", [final_update])
                
                time.sleep(random.uniform(1.5, 3.0)) # 抗封鎖延遲
                if (idx + 1) % 10 == 0: time.sleep(15)
                    
            except Exception as e:
                st.warning(f"跳過 {t}: {e}")
                
            p_bar.progress((idx + 1) / len(tickers))
            
        st.success("🎉 AI 情緒分析、5日預測與總結摘要已同步至 A-K 欄！")
