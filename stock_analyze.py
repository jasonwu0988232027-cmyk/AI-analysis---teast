import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore') # 忽略一些 pandas 計算產生的警告

def calculate_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_poc(df_slice, step=2.0):
    if df_slice.empty: return np.nan
    top = df_slice['High'].max()
    bottom = df_slice['Low'].min()
    if pd.isna(top) or pd.isna(bottom) or top == bottom:
        return np.nan
        
    dynRows = max(1, int(round((top - bottom) / step)))
    actual_step = (top - bottom) / dynRows
    if actual_step == 0:
        return np.nan
    
    bins = np.zeros(dynRows)
    bin_tops = bottom + actual_step * np.arange(1, dynRows + 1)
    bin_bots = bottom + actual_step * np.arange(0, dynRows)
    
    h = df_slice['High'].values
    l = df_slice['Low'].values
    v = df_slice['Volume'].values
    
    for i in range(len(h)):
        if pd.isna(h[i]) or pd.isna(l[i]) or pd.isna(v[i]) or h[i] == l[i]:
            continue
        intersect_mask = (bin_bots <= h[i]) & (bin_tops >= l[i])
        intersect_count = np.sum(intersect_mask)
        if intersect_count > 0:
            bins[intersect_mask] += v[i] / intersect_count
            
    max_bin_idx = np.argmax(bins)
    poc = (bin_tops[max_bin_idx] + bin_bots[max_bin_idx]) / 2.0
    return poc

def main():
    print("====== 股票策略回測系統 ======")
    ticker = input("請輸入要回測的股票代碼 (例如 2330.TW, 2603.TW): ").strip()
    if not ticker:
        print("未輸入代碼，預設使用 2330.TW")
        ticker = "2330.TW"
        
    print(f"\n[{ticker}] 正在下載歷史資料 (預設使用過去 3 年資料以確保有足夠的指標計算長度)...")
    df = yf.download(ticker, period="3y", progress=False)
    
    # 處理 yfinance 可能返回的 MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, level=1, axis=1)
        except:
            df.columns = df.columns.droplevel(1)
            
    df.dropna(how='all', inplace=True)
    
    # 需要至少大約一年的資料來計算 52週 POC
    poc_window = 252 
    if len(df) <= poc_window:
        print(f"資料筆數 ({len(df)}) 不足一年，無法安全計算 52週 POC，請選擇上市較久的股票。")
        return

    print("資料下載完成，正在計算技術指標與滾動式 POC (Rolling POC 避免未來函數)...")
    df['ATR'] = calculate_atr(df)
    
    # 滾動計算 POC（使用過去 252 天資料作為近 52 週的分布）
    poc_series = np.full(len(df), np.nan)
    for i in range(poc_window, len(df)):
        df_slice = df.iloc[i-poc_window:i]
        poc_series[i] = calculate_poc(df_slice)
    df['POC'] = poc_series
    
    # 計算買入訊號：昨日收盤 <= 昨日POC 且 今日收盤 > 今日POC
    df['Signal_Buy'] = (df['Close'].shift(1) <= df['POC'].shift(1)) & (df['Close'] > df['POC'])
    
    # 計算 Stop Loss, Take Profit
    # 原本邏輯：Trailing Stop Loss 以及 取 Global High 為 Take Profit
    stop_loss = np.full(len(df), np.nan)
    take_profit = np.full(len(df), np.nan)
    
    global_high = -np.inf
    atr_values = df['ATR'].values
    high_values = df['High'].values
    
    for i in range(1, len(df)):
        today_high = high_values[i]
        today_atr = atr_values[i]
        
        # 1. Trailing Stop Loss
        if not np.isnan(today_atr) and not np.isnan(today_high):
            current_sl = today_high - 2 * today_atr
            if np.isnan(stop_loss[i-1]):
                stop_loss[i] = current_sl
            else:
                stop_loss[i] = max(stop_loss[i-1], current_sl)
        else:
            stop_loss[i] = stop_loss[i-1]
            
        # 2. Dynamic Take Profit (All Time High / Local High)
        if not np.isnan(today_high):
            if today_high > global_high:
                global_high = today_high
            if global_high != -np.inf:
                take_profit[i] = global_high
    
    df['Stop_Loss'] = stop_loss
    df['Take_Profit'] = take_profit
    
    # 執行回測引擎
    run_backtest(df, ticker)

def run_backtest(df, ticker, initial_capital=100000.0):
    print("\n========== 開始執行回測模擬 ==========")
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_date = None
    trades = []
    
    dates = df.index
    open_prices = df['Open'].values
    high_prices = df['High'].values
    low_prices = df['Low'].values
    close_prices = df['Close'].values
    buy_signals = df['Signal_Buy'].values
    stop_losses = df['Stop_Loss'].values
    take_profits = df['Take_Profit'].values
    
    equity_curve = []
    
    for i in range(1, len(df)):
        current_date = dates[i]
        O, H, L, C = open_prices[i], high_prices[i], low_prices[i], close_prices[i]
        
        # 判斷是否觸發平倉 (使用昨天的 SL / TP 設定)
        if position > 0:
            sl = stop_losses[i-1]
            tp = take_profits[i-1]
            
            sold = False
            sell_price = 0
            reason = ""
            
            # 若開盤直接跳空跌破停損
            if O <= sl:
                sold = True
                sell_price = O
                reason = "Stop Loss (Gap Down)"
            # 盤中跌破停損
            elif L <= sl:
                sold = True
                sell_price = sl
                reason = "Stop Loss"
            # 若開盤直接跳空突破停利
            elif O >= tp:
                sold = True
                sell_price = O
                reason = "Take Profit (Gap Up)"
            # 盤中突破停利
            elif H >= tp:
                sold = True
                sell_price = tp
                reason = "Take Profit"
                
            if sold:
                capital = position * sell_price
                ret = (sell_price - entry_price) / entry_price
                trades.append({
                    '進場時間': entry_date.date() if hasattr(entry_date, 'date') else entry_date,
                    '出場時間': current_date.date() if hasattr(current_date, 'date') else current_date,
                    '進場價格': round(entry_price, 2),
                    '出場價格': round(sell_price, 2),
                    '報酬率(%)': round(ret * 100, 2),
                    '出場原因': reason
                })
                position = 0
        
        # 判斷是否可以進場 (昨天出現買進訊號，今天開盤買進)
        if position == 0 and buy_signals[i-1]:
            # 用今日開盤價買進 (假設市價買入，資本全下)
            position = capital / O
            entry_price = O
            entry_date = current_date
            capital = 0 
            
        # 紀錄每日收盤淨值
        current_equity = position * C if position > 0 else capital
        equity_curve.append({
            'Date': current_date,
            'Equity': current_equity
        })
        
    # 回測結束，清倉 (按照最後一天收盤價)
    if position > 0:
        current_equity = position * close_prices[-1]
        ret = (close_prices[-1] - entry_price) / entry_price
        trades.append({
            '進場時間': entry_date.date() if hasattr(entry_date, 'date') else entry_date,
            '出場時間': dates[-1].date() if hasattr(dates[-1], 'date') else dates[-1],
            '進場價格': round(entry_price, 2),
            '出場價格': round(close_prices[-1], 2),
            '報酬率(%)': round(ret * 100, 2),
            '出場原因': "End of Backtest"
        })
    else:
        current_equity = capital
        
    equity_df = pd.DataFrame(equity_curve).set_index('Date')
    
    # 列印回測分析
    total_return = ((current_equity - initial_capital) / initial_capital) * 100
    trades_df = pd.DataFrame(trades)
    
    print("\n========== 回測績效報告 ==========")
    print(f"回測標的:\t {ticker}")
    print(f"初始資金:\t ${initial_capital:,.2f}")
    print(f"最終資金:\t ${current_equity:,.2f}")
    print(f"總報酬率:\t {total_return:.2f}%")
    
    if not trades_df.empty:
        win_trades = trades_df[trades_df['報酬率(%)'] > 0]
        loss_trades = trades_df[trades_df['報酬率(%)'] <= 0]
        win_rate = len(win_trades) / len(trades_df) * 100
        
        print(f"總交易次數:\t {len(trades_df)}")
        print(f"勝率:\t\t {win_rate:.2f}% ({len(win_trades)} 勝 / {len(loss_trades)} 敗)")
        print(f"平均每筆報酬:\t {trades_df['報酬率(%)'].mean():.2f}%")
        if len(win_trades) > 0:
            print(f"平均獲利報酬:\t {win_trades['報酬率(%)'].mean():.2f}%")
        if len(loss_trades) > 0:
            print(f"平均虧損報酬:\t {loss_trades['報酬率(%)'].mean():.2f}%")
            
        equity_df['High_Water_Mark'] = equity_df['Equity'].cummax()
        equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['High_Water_Mark']) / equity_df['High_Water_Mark']
        max_drawdown = equity_df['Drawdown'].min() * 100
        print(f"最大回撤 (MDD):\t {max_drawdown:.2f}%")
        
        print("\n--- 交易明細 (前 5 筆) ---")
        print(trades_df.head(5).to_string(index=False))
        print("...\n--- 交易明細 (最後 5 筆) ---")
        print(trades_df.tail(5).to_string(index=False))
        
        # 將所有交易紀錄輸出至 CSV
        csv_filename = f"{ticker}_trades.csv"
        trades_df.to_csv(csv_filename, index=False)
        print(f"\n💡 完整交易明細已匯出至 {csv_filename}")
        
    else:
        print("沒有發生任何交易 (未觸發進場訊號)。")
        
    # 可視化 (如果系統有安裝 matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 設定中文字體 (Windows)
        plt.rcParams['axes.unicode_minus'] = False 
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df.index, equity_df['Equity'], label='策略累積淨值 (Equity)', color='b')
        plt.fill_between(equity_df.index, equity_df['Equity'], initial_capital, where=(equity_df['Equity'] > initial_capital), color='g', alpha=0.3)
        plt.fill_between(equity_df.index, equity_df['Equity'], initial_capital, where=(equity_df['Equity'] <= initial_capital), color='r', alpha=0.3)
        plt.axhline(initial_capital, color='black', linestyle='--', label='初始資金')
        plt.title(f'【{ticker}】策略回測權益曲線', fontsize=16)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('淨值', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        png_filename = f"{ticker}_equity_curve.png"
        plt.savefig(png_filename)
        print(f"💡 權益曲線圖表已儲存為 {png_filename}")
    except ImportError:
        print("\n(溫馨提示：若想生成回測圖表，請安裝 matplotlib 模組: pip install matplotlib)")

if __name__ == "__main__":
    main()
