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
st.set_page_config(page_title="GPT-4o 高速進化分析系統 v33.0", layout="wide")

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
client_ai = OpenAI(api_key=OPENAI_API_KEY)

# ==================== 1. 分頁初始化與權重計算 ====================

def init_and_sync_sheets(sh, target_tickers):
    """初始化分頁並記錄權重占比"""
    # 第一頁：預測紀錄
    ws_pred = sh.get_worksheet(0)
    
    # 第二頁：權重占比
    try:
        ws_weight = sh.worksheet("Stock_Weights")
    except gspread.exceptions.WorksheetNotFound:
        ws_weight = sh.add_worksheet(title="Stock_Weights", rows="1000", cols="5")
        ws_weight.insert_row(["股票代號", "權重占比", "更新日期", "說明"], 1)
    
    # 簡單計算等權重（或可根據交易值自定義）
    today_str = datetime.now().strftime("%Y-%m-%d")
    weight_pct = f"{round(1/len(target_tickers), 4)*100}%" if target_tickers else "0%"
    weight_rows = [[t, weight_pct, today_str, "自動分配"] for t in target_tickers]
    
    # 清除舊權重並寫入新權重
    ws_weight.clear()
    ws_weight.insert_row(["股票代號", "權重占比", "更新日期", "說明"], 1)
    ws_weight.append_rows(weight_rows)
    
    return ws_pred

# ==================== 2. 高速 AI 反思分析 ====================

def ai_reflection_analysis(ticker, curr_p, last_record, tech_data):
    """利用 Tier 1 的效能進行高速反思分析"""
    prompt = f"""
    標的: {ticker}, 現價: {curr_p}
    量化指標: {tech_data}
    上次預測記錄: {last_record if last_record else "無"}
    
    請進行：
    1. 自我反思：若有上次記錄，請簡述預測準確度。
    2. 未來預測：給出未來5日價格預測。
    回答格式: 分數,價1,價2,價3,價4,價5,反思短評(30字)
    """
    response = client_ai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "你是具備自我修正能力的交易分析師。"},
                  {"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip().split(',')

# ==================== 3. 主程序 ====================

st.title("⚡ GPT-4o 高速進化分析系統 v33.0 (Tier 1)")

if st.button("🚀 啟動高速全方位分析任務"):
    gc = get_gspread_client()
    if not gc: st.stop()
    
    sh = gc.open("Stock_Predictions_History")
    
    # 讀取現有預測清單 (假設在第一頁的 B 欄)
    ws_temp = sh.get_worksheet(0)
    all_rows = ws_temp.get_all_values()
    df_history = pd.DataFrame(all_rows[1:], columns=all_rows[0]) if len(all_rows) > 1 else pd.DataFrame()
    
    # 取得要分析的代號 (近5日內出現過的標的)
    if not df_history.empty:
        tickers = df_history['股票代號'].unique().tolist()[:100]
    else:
        st.warning("請先在 Excel B 欄填入股票代號。")
        st.stop()

    # 初始化與同步權重
    ws_pred = init_and_sync_sheets(sh, tickers)
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    p_bar = st.progress(0)
    status = st.empty()
    
    # 高速下載數據
    all_hist = yf.download(tickers, period="1mo", group_by='ticker', progress=False)

    for idx, t in enumerate(tickers):
        try:
            status.text(f"⚡ 高速處理中 ({idx+1}/{len(tickers)}): {t}")
            
            df = all_hist[t].dropna() if isinstance(all_hist.columns, pd.MultiIndex) else all_hist.dropna()
            if df.empty: continue
            
            curr_p = round(float(df['Close'].iloc[-1]), 2)
            tech_info = "MA5 > MA20" if (len(df)>1 and df['Close'].rolling(5).mean().iloc[-1] > df['Close'].rolling(20).mean().iloc[-1]) else "MA5 < MA20"
            
            # 獲取該股票的最後一筆歷史預測
            last_rec = df_history[df_history['股票代號'] == t].tail(1).to_dict('records')
            
            # 呼叫 AI (Tier 1 環境，不再需要 sleep 21秒)
            res = ai_reflection_analysis(t, curr_p, last_rec, tech_info)
            
            # 準備寫入資料
            new_row = [today_str, t, curr_p, "交易值指標", res[1], res[2], res[3], res[4], res[5], res[0], res[6]]
            
            # --- 判斷更新或追加 ---
            mask = (df_history['日期'] == today_str) & (df_history['股票代號'] == t)
            if not df_history.empty and mask.any():
                row_idx = df_history.index[mask][0] + 2
                ws_pred.update(f"A{row_idx}:K{row_idx}", [new_row])
            else:
                ws_pred.append_row(new_row)
            
            # 僅保留極短的間隔(0.2秒)確保 API 請求順序
            time.sleep(0.2)
            
        except Exception as e:
            st.warning(f"跳過 {t}: {e}")
            
        p_bar.progress((idx + 1) / len(tickers))

    st.success(f"🎉 高速分析任務完成！共處理 {len(tickers)} 檔標的。資料已同步至 Excel 兩個分頁。")
