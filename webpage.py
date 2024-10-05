import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import requests
import numpy as np

#資料庫處理
import sqlite3

#import update_gap
#update_gap.run_all()


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

url = "https://api.finmindtrade.com/api/v4/data?"
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0wNy0zMCAyMzowMTo0MSIsInVzZXJfaWQiOiJqZXlhbmdqYXUiLCJpcCI6IjExNC4zNC4xMjEuMTA0In0.WDAZzKGv4Du5JilaAR7o7M1whpnGaR-vMDuSeTBXhhA", # 參考登入，獲取金鑰

parameter = {
"dataset": "TaiwanStockPrice",
"data_id": "TAIEX",
"start_date": "2022-10-02",
"end_date": datetime.strftime(datetime.today(),'%Y-%m-%d'),
"token": token, # 參考登入，獲取金鑰
}
data = requests.get(url, params=parameter)
data = data.json()
WeekTAIEXdata = pd.DataFrame(data['data'])

taiex_fin = pd.DataFrame(data['data'])
taiex_fin.date = pd.to_datetime(taiex_fin.date)
taiex_fin.index = taiex_fin.date
taiex_fin.columns = ['日期', 'stock_id', '成交股數', '成交金額', '開盤指數', '最高指數',
       '最低指數', '收盤指數', '漲跌點數', '成交筆數']

# 创建 Streamlit 页面
st.set_page_config(page_title="選擇權籌碼", layout="wide")

# 创建选项卡
tab1, tab2 = st.tabs(["選擇權三大法人籌碼", "上下極限"])

with tab1:
    st.header("選擇權三大法人籌碼")
    
    # 获取当前日期
    today = datetime.now().date()
    # 创建两列
    col1, col2 = st.columns(2)
    
    with col1:
        # 设置日期选择器，限制选择的日期不能超过今天
        selected_date = st.date_input("選擇日期", get_previous_weekday(today), min_value=None, max_value=today)
    
    with col2:
        # 设置整數輸入器，預設值為 7
        num_days = st.number_input("選擇天數", min_value=1, max_value=30, value=7)

    # 设置日期选择器，限制选择的日期不能超过今天
    #selected_date = st.date_input("選擇日期", get_previous_weekday(today), min_value=None, max_value=today)
    # 设置整數輸入器，預設值為 7
    #num_days = st.number_input("選擇天數", min_value=1, max_value=30, value=7)


    selected_date_df = df_options_futures_bs[df_options_futures_bs["日期"] == datetime.strftime(selected_date, '%Y-%m-%d')]
    startdate = datetime.strftime(selected_date - timedelta(days=7), '%Y-%m-%d')
    #enddate = datetime.strftime(selected_date, '%Y-%m-%d')
    
    #gap_day_df = df_options_futures_daygap[(df_options_futures_daygap["日期"] >=startdate)&(df_options_futures_daygap["日期"] <=enddate) ]
    
    
    st.write("以下是選擇日期的數據：")
    st.dataframe(selected_date_df)

    # 创建两列
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("日變動的數據")
        foreign_df = pd.DataFrame()
        dealer_df = pd.DataFrame()
        retail_df = pd.DataFrame()

        for i in range(num_days):
            gap_day_df = df_options_futures_daygap[df_options_futures_daygap["日期"] ==datetime.strftime(selected_date - timedelta(days=num_days-1-i), '%Y-%m-%d') ]
            callputtemp = putcallsum_sep[putcallsum_sep["日期"] ==datetime.strftime(selected_date - timedelta(days=num_days-1-i), '%Y-%m-%d')]
            
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

                    elif gap_day_df.loc[idx,"種類"]=="賣權":
                        if np.abs(gap_day_df.loc[idx,"單價"])>put_num:
                            gap_day_df.loc[idx,"成交位置"] = "價內"
                        else:
                            gap_day_df.loc[idx,"成交位置"] = "價外"

                    else:
                        gap_day_df.loc[idx,"成交位置"] = "不分"
                        
                    #gap_day_df.loc[idx,"成交位置"]
                foreign_df = foreign_df.append(gap_day_df[gap_day_df["身份"]=="外資"])
                dealer_df = dealer_df.append(gap_day_df[gap_day_df["身份"]=="自營商"])
                retail_df = retail_df.append(gap_day_df[gap_day_df["身份"]=="散戶"])

                #st.text(datetime.strftime(selected_date - timedelta(days=6-i), '%Y-%m-%d') + " 價平和買權成交價: "+str(call_num)+"，價平和賣權成交價: "+str(put_num))
        st.dataframe(foreign_df.sort_values(by="日期",ascending=False))
        st.dataframe(dealer_df.sort_values(by="日期",ascending=False))
        st.dataframe(retail_df.sort_values(by="日期",ascending=False))

    
    with col4:
        st.write("參考數據")
        callputtemp = putcallsum_sep[(putcallsum_sep["日期"] <=datetime.strftime(selected_date, '%Y-%m-%d'))]
        callputtemp = callputtemp[callputtemp["日期"] >=datetime.strftime(selected_date - timedelta(days=num_days), '%Y-%m-%d')]
        callputtemp = callputtemp.sort_values(by="日期",ascending=False)
        callputtemp = callputtemp.reset_index(drop=True)
        callputtemp.columns = ["日期","價平和履約價","價平和買權成交價","價平和賣權成交價"]
        df_cost = pd.read_sql("select distinct * from df_cost", connection)
        df_cost["外資成本"] = df_cost["外資成本"].astype(int)
        df_gap = taiex_fin[['日期','漲跌點數']]
        df_gap["日期"] = df_gap["日期"].astype(str)
        callputtemp = callputtemp.join(df_cost.set_index("日期"),on="日期").join(df_gap.set_index("日期"),on="日期")

        st.dataframe(callputtemp)

        df_option_limit = pd.read_sql("select distinct * from df_option_limit", connection)
        limit_temp = df_option_limit[(df_option_limit["日期"] <=datetime.strftime(selected_date, '%Y-%m-%d'))]
        limit_temp = limit_temp[limit_temp["日期"] >=datetime.strftime(selected_date - timedelta(days=num_days), '%Y-%m-%d')]
        st.dataframe(limit_temp)

        #st.dataframe(taiex_fin)
# 运行 Streamlit 应用
# 请在命令行运行 `streamlit run app.py`