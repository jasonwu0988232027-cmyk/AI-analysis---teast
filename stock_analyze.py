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
import os
import random
import urllib3

# --- 基礎配置 ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="GPT-4 量化混合系統 v25.0", layout="wide")

# --- 金鑰管理 ---
SHEET_NAME = "Stock_Predictions_History"
CREDENTIALS_JSON = "eco-precept-485904-j5-7ef3cdda1b03.json"

# 請優先使用 Streamlit Secrets 設定 OPENAI_API_KEY
# 若暫時測試可填入下方預設值，但建議部署時移除
DEFAULT_GPT_KEY = "sk-proj-QbBYyhf... (請更換為新產生的 Key)"
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", DEFAULT_GPT_KEY)
client_ai = OpenAI(api_key=OPENAI_API_KEY)

# ==================== 1. 雲端連線與初始化 ====================

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

# ==================== 2. 量化分析與新聞模組 ====================

def get_tech_indicators(ticker, df):
    """計算量化指標：提供數據背景給 AI"""
    try:
        # 計算 5/20 均線指標
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        status = "黃金交叉(偏多)" if ma5 > ma20 else "死亡交叉(偏空)"
        
        info = yf.Ticker(ticker).info
        pe = info.get('forwardPE', 'N/A')
        
        return f"技術面:{status}, 本益比:{pe}"
    except:
        return "數據暫無"

def fetch_news(symbol):
    """爬取新聞網站摘要"""
    stock_id = symbol.split('.')[0]
    headers = {'User-Agent': 'Mozilla/5.0'}
    # 針對鉅亨網與經濟日報
    urls = ["https://news.cnyes.com/news/cat/tw_stock_news", "https://money.udn.com/money/index"]
    try:
        res = requests.get(urls[0], headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        titles = [t.get_text().strip() for t in soup.find_all(['h3', 'a']) if stock_id in t.get_text()]
        return " | ".join(titles[:5])
    except: return "查無新聞"

# ==================== 3. 主執行流程 ====================

st.title("🏆 GPT-4o 量化混合預測系統")

if st.button("🚀 啟動全方位混合分析任務"):
    client_gs = get_gspread_client()
    if client_gs:
        sh = client_gs.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        
        # --- 自動建立表首 (A-K 欄) ---
        headers = ["日期", "股票代號", "收盤價格", "交易值指標", "預測1", "預測2", "預測3", "預測4", "預測5", "AI評分", "AI總結"]
        if not ws.row_values(1): ws.insert_row(headers, 1)

        # 讀取 Excel 資料
        raw_data = ws.get_all_values()
        df_sheet = pd.DataFrame(raw_data[1:], columns=raw_data[0])
        df_sheet['日期'] = pd.to_datetime(df_sheet['日期'])
        
        # 篩選近 5 日標的
        target_df = df_sheet[df_sheet['日期'] >= (datetime.now() - timedelta(days=5))].head(100)
        tickers = target_df['股票代號'].tolist()
        
        if not tickers:
            st.warning("近 5 日無資料標的，請確認 Excel A-B 欄是否已填入。")
            st.stop()

        p_bar = st.progress(0)
        status = st.empty()
        
        # 批量下載歷史數據
        all_hist = yf.download(tickers, period="3mo", group_by='ticker', threads=True, progress=False)

        for idx, t in enumerate(tickers):
            try:
                status.text(f"GPT-4o 分析中 ({idx+1}/{len(tickers)}): {t}")
                df = all_hist[t].dropna() if isinstance(all_hist.columns, pd.MultiIndex) else all_hist.dropna()
                if df.empty: continue
                
                curr_p = round(float(df['Close'].iloc[-1]), 2)
                tech_info = get_tech_indicators(t, df)
                news_info = fetch_news(t)
                
                # --- GPT-4o 混合分析 ---
                
                response = client_ai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "你是一位結合量化指標與新聞情緒的台股策略專家。"},
                        {"role": "user", "content": f"""
                        分析股票 {t}，目前股價 {curr_p}。
                        量化數據：{tech_info}。
                        新聞背景：{news_info}
                        
                        請根據數據與新聞給出：
                        1. 情緒評分 (-5 到 5)
                        2. 未來 5 日預測價格 (5個數值)
                        3. 30字總結利多利空
                        
                        回答格式要求：評分,價1,價2,價3,價4,價5,總結
                        """}
                    ]
                )
                
                res = response.choices[0].message.content.strip().split(',')
                # 解析並寫入 E-K 欄位
                # E-I: 預測價, J: 分數, K: 總結
                update_row = res[1:6] + [f"GPT分:{res[0]}", res[6] if len(res)>6 else "分析完畢"]
                ws.update(f"E{idx+2}:K{idx+2}", [update_row])
                
                time.sleep(1.0) # 基礎延遲
                    
            except Exception as e:
                st.warning(f"跳過 {t}: {e}")
                
            p_bar.progress((idx + 1) / len(tickers))
            
        status.text("✅ 任務已完成")
        st.success("🎉 全方位混合分析數據已同步至 Excel A-K 欄！")
