import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from openai import OpenAI  # 改用 OpenAI SDK
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
st.set_page_config(page_title="GPT-4 股市分析專家 v22.0", layout="wide")

# --- 金鑰與參數 ---
SHEET_NAME = "Stock_Predictions_History"
CREDENTIALS_JSON = "eco-precept-485904-j5-7ef3cdda1b03.json"

# 設定 OpenAI API Key (請在 Streamlit Secrets 設定 OPENAI_API_KEY)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "你的_OPENAI_API_KEY")
client_ai = OpenAI(api_key=OPENAI_API_KEY)

# ==================== 1. 雲端連線模組 ====================

def get_gspread_client():
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

# ==================== 2. 新聞爬蟲模組 ====================

def fetch_stock_news(symbol):
    """搜尋各大新聞網標的資訊"""
    stock_id = symbol.split('.')[0]
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    urls = [
        "https://news.cnyes.com/news/cat/tw_stock_news",
        "https://money.udn.com/money/index",
        "https://www.ftnn.com.tw/category/6"
    ]
    news_pool = []
    try:
        # 爬取新聞
        res = requests.get(urls[0], headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        titles = [t.get_text().strip() for t in soup.find_all(['h3', 'a']) if stock_id in t.get_text()]
        news_pool.extend(titles[:5])
    except: pass
    return " | ".join(news_pool) if news_pool else "暫無重大相關新聞。"

# ==================== 3. 主執行程序 (GPT-4 驅動) ====================

st.title("🚀 GPT-4 股市深度情緒分析系統")
st.info("模式：讀取近5日標的 -> 爬取新聞 -> GPT-4 評分與預測")

if st.button("啟動 GPT-4 全方位分析任務"):
    if not OPENAI_API_KEY:
        st.error("❌ 找不到 OPENAI_API_KEY，請在 Secrets 中設定。")
        st.stop()
        
    client_gs = get_gspread_client()
    if client_gs:
        sh = client_gs.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        
        # 讀取 Excel 並過濾近 5 日 
        raw_rows = ws.get_all_values()
        if len(raw_rows) <= 1:
            st.warning("表格內無資料。")
            st.stop()
            
        df_sheet = pd.DataFrame(raw_rows[1:], columns=raw_rows[0])
        df_sheet['日期'] = pd.to_datetime(df_sheet['日期'])
        
        five_days_ago = datetime.now() - timedelta(days=5)
        df_recent = df_sheet[df_sheet['日期'] >= five_days_ago].head(100)
        
        tickers = df_recent['股票代號'].tolist()
        st.info(f"偵測到 {len(tickers)} 檔近 5 日熱門標的，開始 AI 分析...")
        
        p_bar = st.progress(0)
        status = st.empty()
        
        for idx, t in enumerate(tickers):
            try:
                status.text(f"GPT-4 正在處理 ({idx+1}/{len(tickers)}): {t}")
                
                # 抓取新聞
                news_txt = fetch_stock_news(t)
                
                # 呼叫 GPT-4o 進行深度分析
                response = client_ai.chat.completions.create(
                    model="gpt-4o",  # 使用 GPT-4o 兼顧性能與成本
                    messages=[
                        {"role": "system", "content": "你是一位精通台股的資深投顧分析師。"},
                        {"role": "user", "content": f"""
                        分析股票 {t}。相關新聞如下：
                        {news_txt}
                        
                        請提供：
                        1. 情緒評分 (-5 到 5)
                        2. 未來 5 日預測價格 (5個數值)
                        3. 30字以內的利多/利空分析
                        
                        請嚴格按此格式回答：評分,價1,價2,價3,價4,價5,總結
                        """}
                    ],
                    temperature=0.3  # 降低隨機性，讓預測更穩健
                )
                
                ai_output = response.choices[0].message.content.strip().split(',')
                
                score = ai_output[0]
                preds = ai_output[1:6]
                summary = ai_output[6] if len(ai_output) > 6 else "分析完畢"
                
                # 更新 Excel E-K 欄
                final_update = preds + [f"GPT分:{score}", summary]
                ws.update(f"E{idx+2}:K{idx+2}", [final_update])
                
                # GPT-4 速度較快，延遲可稍微縮短
                time.sleep(1.0)
                    
            except Exception as e:
                st.warning(f"跳過 {t}: {e}")
                
            p_bar.progress((idx + 1) / len(tickers))
            
        st.success("🎉 GPT-4 分析任務已完成，數據已同步至 Excel！")
