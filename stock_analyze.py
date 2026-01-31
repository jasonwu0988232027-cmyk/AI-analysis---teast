import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from openai import OpenAI
from google.oauth2.service_account import Credentials
from datetime import datetime
import time
import random
import sys

# --- 基礎配置 ---
st.set_page_config(page_title="股市混合分析系統 v31.0", layout="wide")

def get_gspread_client():
    """解決私鑰格式問題"""
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    if "gcp_service_account" in st.secrets:
        try:
            creds_info = dict(st.secrets["gcp_service_account"])
            pk = str(creds_info["private_key"]).strip().replace('"', '').replace("'", "").replace("\\n", "\n")
            creds_info["private_key"] = pk
            return gspread.authorize(Credentials.from_service_account_info(creds_info, scopes=scopes))
        except Exception as e:
            st.error(f"❌ Sheets 授權失敗: {e}")
    return None

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
ai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ==================== 核心分析與降速邏輯 ====================

def get_fallback_data(ticker, df):
    """當 AI 受限時的量化指標備案"""
    # 
    ma5 = df['Close'].rolling(5).mean().iloc[-1]
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    last_p = float(df['Close'].iloc[-1])
    score = 5 if ma5 > ma20 else -2
    preds = [round(last_p * (1 + (score*0.001)*i), 2) for i in range(1, 6)]
    return score, preds, f"量化指標：{('多頭' if score > 0 else '空頭')}趨勢"

st.title("🏆 股市全能分析系統 v31.0")

if st.button("🚀 執行 A-K 欄全方位分析"):
    gc = get_gspread_client()
    if not gc: st.stop()
    
    sh = gc.open("Stock_Predictions_History")
    ws = sh.get_worksheet(0)
    
    # 1. 建立標題列
    headers = ["日期", "股票代號", "收盤價格", "交易值指標", "預測1", "預測2", "預測3", "預測4", "預測5", "AI情緒分", "AI短評摘要"]
    if not ws.row_values(1): ws.insert_row(headers, 1)

    raw = ws.get_all_values()
    df_sheet = pd.DataFrame(raw[1:], columns=raw[0])
    tickers = df_sheet['股票代號'].dropna().tolist()
    
    p_bar = st.progress(0)
    status = st.empty()
    all_hist = yf.download(tickers, period="3mo", group_by='ticker', progress=False)

    for idx, t in enumerate(tickers):
        try:
            status.text(f"正在分析 ({idx+1}/{len(tickers)}): {t}")
            df = all_hist[t].dropna() if isinstance(all_hist.columns, pd.MultiIndex) else all_hist.dropna()
            if df.empty: continue
            
            # --- 核心降速機制：每 3 次請求強制休息 ---
            if idx > 0 and idx % 3 == 0:
                status.text("⏳ 正在等待 OpenAI 頻率限制 (20秒)...")
                time.sleep(21) 

            try:
                # 嘗試 GPT-4o
                response = ai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": f"分析股票 {t}，現價 {df['Close'].iloc[-1]}。請回傳：分數,價1,價2,價3,價4,價5,摘要"}]
                )
                res = response.choices[0].message.content.strip().split(',')
                score, preds, summary = res[0], res[1:6], res[6]
            except Exception as e:
                # 報錯則自動切換量化模型
                score, preds, summary = get_fallback_data(t, df)
                summary = f"[量化備案] {summary}"

            # 更新 E-K 欄
            ws.update(f"E{idx+2}:K{idx+2}", [preds + [f"{score}", summary]])
            
        except Exception as e:
            st.warning(f"跳過 {t}: {e}")
        p_bar.progress((idx + 1) / len(tickers))
        
    st.success("🎉 分析完成！已避開頻率限制並同步 Excel。")
