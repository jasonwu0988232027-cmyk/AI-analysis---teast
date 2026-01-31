import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from openai import OpenAI
from google.oauth2.service_account import Credentials
from datetime import datetime
import time
import sys

# --- 基礎編碼設定 ---
st.set_page_config(page_title="GPT-4o 自我進化分析系統 v32.0", layout="wide")

def get_gspread_client():
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

# ==================== 1. 初始化與頁籤管理 ====================

def init_sheets(sh):
    # 第一頁：預測紀錄
    ws_pred = sh.get_worksheet(0)
    headers_pred = ["日期", "股票代號", "收盤價格", "交易值指標", "預測1", "預測2", "預測3", "預測4", "預測5", "AI情緒分", "AI反思摘要"]
    if not ws_pred.row_values(1):
        ws_pred.insert_row(headers_pred, 1)
    
    # 第二頁：權重占比
    try:
        ws_weight = sh.worksheet("Stock_Weights")
    except gspread.exceptions.WorksheetNotFound:
        ws_weight = sh.add_worksheet(title="Stock_Weights", rows="100", cols="5")
        ws_weight.insert_row(["股票代號", "權重占比", "更新日期", "所屬產業"], 1)
    
    return ws_pred, ws_weight

# ==================== 2. AI 反思訓練與預測邏輯 ====================

def get_ai_prediction_with_reflection(ticker, curr_p, history_preds, tech_info):
    """
    history_preds: 該股票過去的預測與實際結果，提供給 AI 進行反思
    """
    prompt = f"""
    標的: {ticker}, 目前實際收盤價: {curr_p}
    量化指標: {tech_info}
    歷史預測記錄: {history_preds if history_preds else "無歷史資料"}
    
    任務：
    1. 自我檢討：若有歷史資料，比對預測誤差，說明為何看錯或看對。
    2. 重新預測：基於反思，給出未來5日預測與情緒分。
    
    回答格式(嚴格要求): 分數,價1,價2,價3,價4,價5,反思短評(30字)
    """
    try:
        response = ai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "你是一位具有自我修正能力的量化交易員，會根據過去的錯誤調整模型。"},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip().split(',')
    except:
        return ["0", "0", "0", "0", "0", "0", "AI 思考超時"]

# ==================== 3. 主程序 ====================

st.title("🛡️ GPT-4o 自我進化分析系統 v32.0")

if st.button("🚀 啟動進化分析任務"):
    gc = get_gspread_client()
    if not gc: st.stop()
    
    sh = gc.open("Stock_Predictions_History")
    ws_pred, ws_weight = init_sheets(sh)
    
    # 獲取今日日期
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # 讀取現有資料進行比對
    all_data = ws_pred.get_all_values()
    df_history = pd.DataFrame(all_data[1:], columns=all_data[0]) if len(all_data) > 1 else pd.DataFrame(columns=headers_pred)
    
    # 模擬讀取要分析的 Top 100 名單 (從 B 欄或外部 API)
    # 這裡我們假設從 A 欄/B 欄的現有代號出發
    tickers = list(set(df_history['股票代號'].tail(50).tolist())) # 取最近有出現的代碼，可自行替換
    if not tickers: tickers = ["2330.TW", "2317.TW"] # 預設測試
    
    # 計算權重 (以交易值簡單模擬)
    weights_data = []
    total_tickers = len(tickers)
    for t in tickers:
        weights_data.append([t, f"{round(1/total_tickers, 4)*100}%", today_str])
    ws_weight.update("A2:C" + str(len(weights_data)+1), weights_data)

    p_bar = st.progress(0)
    status = st.empty()
    all_hist = yf.download(tickers, period="3mo", group_by='ticker', progress=False)

    for idx, t in enumerate(tickers):
        try:
            # 頻率限制處理
            if idx > 0 and idx % 3 == 0:
                status.text("⏳ 頻率限制冷卻中...")
                time.sleep(21)
                
            df = all_hist[t].dropna() if isinstance(all_hist.columns, pd.MultiIndex) else all_hist.dropna()
            if df.empty: continue
            
            curr_p = round(float(df['Close'].iloc[-1]), 2)
            
            # 撈取歷史紀錄用於反思
            past_record = df_history[df_history['股票代號'] == t].tail(1).to_dict('records')
            
            # AI 分析與反思
            res = get_ai_prediction_with_reflection(t, curr_p, past_record, "MA黃金交叉" if len(df)>20 else "N/A")
            
            # --- 核心：判斷是更新還是追加 ---
            # 檢查是否已存在今日同股票的紀錄
            mask = (df_history['日期'] == today_str) & (df_history['股票代號'] == t)
            
            new_row = [today_str, t, curr_p, "交易值指標", res[1], res[2], res[3], res[4], res[5], res[0], res[6]]
            
            if mask.any():
                row_index = df_history.index[mask][0] + 2 # +2 因為 Excel 從 1 開始且有標題列
                ws_pred.update(f"A{row_index}:K{row_index}", [new_row])
            else:
                ws_pred.append_row(new_row)
            
        except Exception as e:
            st.warning(f"跳過 {t}: {e}")
        p_bar.progress((idx + 1) / len(tickers))

    st.success("🎉 進化分析完成！資料已更新至各分頁。")
