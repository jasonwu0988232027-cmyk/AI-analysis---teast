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
st.set_page_config(page_title="GPT-4 量化混合專家 v23.0", layout="wide")

# --- 金鑰設定 ---
SHEET_NAME = "Stock_Predictions_History"
CREDENTIALS_JSON = "eco-precept-485904-j5-7ef3cdda1b03.json"

# 使用您提供的 OpenAI API KEY (sk-svcacct-...)
DEFAULT_GPT_KEY = "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"
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", DEFAULT_GPT_KEY)
client_ai = OpenAI(api_key=OPENAI_API_KEY)

# ==================== 1. 雲端與量化模組 ====================

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

def get_technical_data(ticker, df):
    """計算技術面數據，作為 GPT-4o 的參考 """
    score = 0
    try:
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        crossover = "黃金交叉" if ma5 > ma20 else "死亡交叉"
        if ma5 > ma20: score += 5
        
        info = yf.Ticker(ticker).info
        pe = info.get('forwardPE', "N/A")
        if isinstance(pe, (int, float)) and pe < 18: score += 2
        
        return score, crossover, pe
    except:
        return 0, "無訊號", "N/A"

# ==================== 2. 新聞爬蟲模組 ====================

def fetch_latest_news(symbol):
    stock_id = symbol.split('.')[0]
    headers = {'User-Agent': 'Mozilla/5.0'}
    urls = ["https://news.cnyes.com/news/cat/tw_stock_news", "https://money.udn.com/money/index"]
    news_list = []
    try:
        res = requests.get(urls[0], headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        titles = [t.get_text().strip() for t in soup.find_all(['h3', 'a']) if stock_id in t.get_text()]
        news_list.extend(titles[:5])
    except: pass
    return " | ".join(news_list) if news_list else "暫無新聞"

# ==================== 3. 主執行程序 (混合動力) ====================

st.title("🏆 GPT-4 + 量化因子混合預測系統")

if st.button("啟動 Top 100 混合分析任務"):
    client_gs = get_gspread_client()
    if client_gs:
        sh = client_gs.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        
        # 標題檢查
        headers = ["日期", "股票代號", "收盤價格", "交易值指標", "5日預測-1", "5日預測-2", "5日預測-3", "5日預測-4", "5日預測-5", "誤差%", "AI總結"]
        if not ws.row_values(1): ws.insert_row(headers, 1)

        raw_rows = ws.get_all_values()
        df_sheet = pd.DataFrame(raw_rows[1:], columns=raw_rows[0])
        df_sheet['日期'] = pd.to_datetime(df_sheet['日期'])
        
        # 篩選近 5 日標的
        target_df = df_sheet[df_sheet['日期'] >= (datetime.now() - timedelta(days=5))].head(100)
        tickers = target_df['股票代號'].tolist()
        
        p_bar = st.progress(0)
        all_hist = yf.download(tickers, period="3mo", group_by='ticker', threads=True, progress=False)

        for idx, t in enumerate(tickers):
            try:
                df = all_hist[t].dropna() if isinstance(all_hist.columns, pd.MultiIndex) else all_hist.dropna()
                if df.empty: continue
                
                curr_p = round(float(df['Close'].iloc[-1]), 2)
                
                # 1. 取得數據指標
                tech_score, cross_status, pe_val = get_technical_data(t, df)
                # 2. 爬取新聞
                news_txt = fetch_latest_news(t)
                
                # 3. GPT-4o 綜合判斷 
                response = client_ai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "你是一位結合量化數據與新聞情緒的台股策略分析師。"},
                        {"role": "user", "content": f"""
                        標的：{t}，現價：{curr_p}。
                        量化數據：技術面 {cross_status}，基本面 PE 為 {pe_val}，數據權重分：{tech_score}。
                        新聞內容：{news_txt}
                        
                        請根據數據與新聞給出：
                        1. 綜合情緒評分 (-5 到 5)
                        2. 未來 5 日預測價格 (5個數值)
                        3. 30字總結
                        
                        回答格式：評分,價1,價2,價3,價4,價5,總結
                        """}
                    ]
                )
                
                res = response.choices[0].message.content.strip().split(',')
                score, preds, summary = res[0], res[1:6], res[6]
                
                # 寫入 Excel E-K 欄
                ws.update(f"E{idx+2}:K{idx+2}", [preds + [f"GPT分:{score}", summary]])
                st.write(f"✅ {t}：量化({cross_status}) + AI({score}分) 完成")
                
            except Exception as e:
                st.warning(f"跳過 {t}: {e}")
            p_bar.progress((idx + 1) / len(tickers))
            
        st.success("🎉 全方位混合分析已更新至 Excel！")
