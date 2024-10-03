import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import requests
import numpy as np

#資料庫處理
import sqlite3

import update_gap



update_gap.run_all()


#建立資料庫連結
connection = sqlite3.connect('選擇權分析資料.sqlite3')

#讀取table
df_options_futures_bs = pd.read_sql("select distinct * from options_futures_bs", connection)
df_options_futures_daygap = pd.read_sql("select distinct * from df_options_futures_daygap", connection)
putcallsum_sep = pd.read_sql("select distinct * from putcallsum_sep", connection)


token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0wNy0zMCAyMzowMTo0MSIsInVzZXJfaWQiOiJqZXlhbmdqYXUiLCJpcCI6IjExNC4zNC4xMjEuMTA0In0.WDAZzKGv4Du5JilaAR7o7M1whpnGaR-vMDuSeTBXhhA", # 參考登入，獲取金鑰

def get_previous_weekday(date):
    """如果日期是周末，返回前一个平日"""
    while date.weekday() > 4:  # 周六和周日的 weekday() 分别为 5 和 6
        date -= timedelta(days=1)
    return date

# 创建 Streamlit 页面
st.set_page_config(page_title="選擇權籌碼", layout="wide")

# 创建选项卡
tab1, tab2 = st.tabs(["選擇權三大法人籌碼", "上下極限"])

with tab1:
    st.header("選擇權三大法人籌碼")
    
    # 获取当前日期
    today = datetime.now().date()
    

    # 设置日期选择器，限制选择的日期不能超过今天
    selected_date = st.date_input("選擇日期", get_previous_weekday(today), min_value=None, max_value=today)


    selected_date_df = df_options_futures_bs[df_options_futures_bs["日期"] == datetime.strftime(selected_date, '%Y-%m-%d')]
    startdate = datetime.strftime(selected_date - timedelta(days=7), '%Y-%m-%d')
    #enddate = datetime.strftime(selected_date, '%Y-%m-%d')
    
    #gap_day_df = df_options_futures_daygap[(df_options_futures_daygap["日期"] >=startdate)&(df_options_futures_daygap["日期"] <=enddate) ]
    
    
    st.write("以下是選擇日期的數據：")
    st.dataframe(selected_date_df)

    st.write("以下是日變動的數據：")
    for i in range(7):
        gap_day_df = df_options_futures_daygap[df_options_futures_daygap["日期"] ==datetime.strftime(selected_date - timedelta(days=6-i), '%Y-%m-%d') ]
        callputtemp = putcallsum_sep[putcallsum_sep["日期"] ==datetime.strftime(selected_date - timedelta(days=6-i), '%Y-%m-%d')]
        
        if len(gap_day_df) !=0:
            call_num = int(callputtemp["價平和買權成交價"])
            put_num = int(callputtemp["價平和賣權成交價"])
            gap_day_df["成交位置"]=""
            for idx in gap_day_df.index:
                if gap_day_df.loc[idx,"種類"]=="買權":
                    if np.abs(gap_day_df.loc[idx,"單價"])>call_num:
                        gap_day_df.loc[idx,"成交位置"] = "價內"
                    else:
                        gap_day_df.loc[idx,"成交位置"] = "價外"

                if gap_day_df.loc[idx,"種類"]=="賣權":
                    if np.abs(gap_day_df.loc[idx,"單價"])>put_num:
                        gap_day_df.loc[idx,"成交位置"] = "價內"
                    else:
                        gap_day_df.loc[idx,"成交位置"] = "價外"
                    
                #gap_day_df.loc[idx,"成交位置"]

            st.text(datetime.strftime(selected_date - timedelta(days=6-i), '%Y-%m-%d') + " 價平和買權成交價: "+str(call_num)+"，價平和賣權成交價: "+str(put_num))
            st.dataframe(gap_day_df)

with tab2:
    st.header("上下極限")
    st.write("這是其他功能的頁面。")

# 运行 Streamlit 应用
# 请在命令行运行 `streamlit run app.py`
