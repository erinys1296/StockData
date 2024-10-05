import requests
import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
from bs4 import BeautifulSoup

#資料庫處理
import sqlite3

import warnings
warnings.filterwarnings('ignore')




def cal_daygap(select_date,previous_date,df_options_futures_bs):
    # 創建一個空的 DataFrame 用於存放計算結果
    results_df = pd.DataFrame()

    # 過濾出選擇日和前一日的數據
    selected_day_df = df_options_futures_bs[df_options_futures_bs['日期'] == select_date].set_index(['身份', '種類'])
    previous_day_df = df_options_futures_bs[df_options_futures_bs['日期'] == previous_date].set_index(['身份', '種類'])

    # 計算異動口數和異動金額
    results_df['異動口數'] = selected_day_df['未平倉口數'] - previous_day_df['未平倉口數']
    results_df['異動金額'] = selected_day_df['未平倉金額'] - previous_day_df['未平倉金額']

    # 根據種類計算單價
    def calculate_unit_price(row):
        if row.name[1] == '期貨':
            return 50000
        elif row.name[1] in ['買權', '賣權']:
            if row['異動口數'] != 0:
                return round((row['異動金額'] * 1000) / row['異動口數'] / 50)
            else:
                return 0
        return None

    results_df['單價'] = results_df.apply(calculate_unit_price, axis=1)

    # 添加其他必要欄位
    results_df['日期'] = select_date
    results_df['身份'] = results_df.index.get_level_values('身份')
    results_df['種類'] = results_df.index.get_level_values('種類')

    # 重新排序列並重置索引
    results_df = results_df.reset_index(drop=True)
    results_df = results_df[['日期', '身份', '種類', '異動口數', '異動金額', '單價']]

    # 顯示計算後的結果
    return results_df

def option_limit(selected_date):
    
        
        
    url = "https://api.finmindtrade.com/api/v4/data"
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0wNy0zMCAyMzowMTo0MSIsInVzZXJfaWQiOiJqZXlhbmdqYXUiLCJpcCI6IjExNC4zNC4xMjEuMTA0In0.WDAZzKGv4Du5JilaAR7o7M1whpnGaR-vMDuSeTBXhhA", # 參考登入，獲取金鑰

    parameter = {
        "dataset": "TaiwanOptionInstitutionalInvestors",
        "data_id": "TXO",
        "start_date": selected_date,
        "end_date": selected_date,
        "token": token, # 參考登入，獲取金鑰
    }
    resp = requests.get(url, params=parameter)
    data = resp.json()
    df = pd.DataFrame(data["data"])
    
    
    selected_df_options = df[((df["institutional_investors"] == "自營商")|(df["institutional_investors"] == "外資"))][list(df.columns)[:4]+["short_open_interest_balance_volume","short_open_interest_balance_amount"]]
    selected_df_options=selected_df_options[selected_df_options.date==selected_date]
    selected_df_options["SC成本"] = (round(selected_df_options["short_open_interest_balance_amount"]*1000/50/selected_df_options["short_open_interest_balance_volume"],0)).astype('int')
    Calltable,Puttable = callputtable(selected_date)

    
    selected_df_options["position"] = 0
    for i in selected_df_options.index:
        if selected_df_options.loc[i,"call_put"] == "買權":
            selected_df_options.loc[i,"position"] = Calltable[(Calltable["最後成交價"]<selected_df_options.loc[i,"SC成本"])&(Calltable["最後成交價"]!=0)]["履約價"].min()
        else: #賣權
            selected_df_options.loc[i,"position"] = Puttable[(Puttable["最後成交價"]<selected_df_options.loc[i,"SC成本"])&(Puttable["最後成交價"]!=0)]["履約價"].max()

    #print(selected_df_options)
    #轉換talbe並計算上下極限
    #上下極限計算
    dfarr = []
    for i in selected_df_options["institutional_investors"].unique():  #自營商 and 外資
        #賣買權 賣賣權 單價
        Callcost = selected_df_options[(selected_df_options["institutional_investors"] == i)&(selected_df_options["call_put"] == '買權')]["SC成本"].values[0]
        Putcost = selected_df_options[(selected_df_options["institutional_investors"] == i)&(selected_df_options["call_put"] == '賣權')]["SC成本"].values[0]

        Callposition = selected_df_options[(selected_df_options["institutional_investors"] == i)&(selected_df_options["call_put"] == '買權')]["position"].values[0]
        Putposition = selected_df_options[(selected_df_options["institutional_investors"] == i)&(selected_df_options["call_put"] == '賣權')]["position"].values[0]
        maxlimit = Callposition + Callcost + Putcost
        minlimit = Putposition - Callcost - Putcost
        dfarr.append(["臺指選擇權",selected_date,i,maxlimit,minlimit])
        
        
    return pd.DataFrame(dfarr,columns = ["商品名稱","日期","身份別","上極限","下極限"])

def callputtable(querydate):

    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0wNy0zMCAyMzowMTo0MSIsInVzZXJfaWQiOiJqZXlhbmdqYXUiLCJpcCI6IjExNC4zNC4xMjEuMTA0In0.WDAZzKGv4Du5JilaAR7o7M1whpnGaR-vMDuSeTBXhhA"
    url = "https://api.finmindtrade.com/api/v4/data?"

    parameter = {
        "dataset": "TaiwanOptionDaily",
        "data_id":"TXO",
        "start_date": querydate.replace('/','-'),
        "end_date": querydate.replace('/','-'),
        "token": token, # 參考登入，獲取金鑰
    }
    data = requests.get(url, params=parameter)
    data = data.json()
    data = pd.DataFrame(data['data'])
    data = data[data["trading_session"] == 'position']
    data.date = pd.to_datetime(data.date)
    data = data[data.date == querydate.replace('/','-')]
    data.columns = ["日期","契約","到期月份(週別)","履約價","買賣權","開盤價","最高價","最低價","最後成交價","成交量","A","b","C"]

    df = data
    #處理欄位空格
    newcol = [stri.replace(' ','') for stri in df.columns]
    df.columns = newcol

    #抓取契約結算日

    # 將結算日的爬蟲寫到 function外 (因為不會隨著時間改變而改變，減少爬蟲次數)
    response = requests.get('https://www.taifex.com.tw/cht/5/optIndxFSP')

    # 解析HTML標記
    soup = BeautifulSoup(response.text, "lxml")

    # 找到表格元素
    table = soup.find("table", {"class": "table_f"}) 

    # 將表格數據轉換成Pandas數據框
    datedf = pd.read_html(str(table))[0]

    #處理欄位空格
    newcol = [stri.replace(' ','') for stri in datedf.columns]
    datedf.columns = newcol

    try:
        enddate = datedf[datedf[datedf.columns[0]]>querydate.replace('-','/')][datedf.columns[0]].min()
        weekfilter = datedf[datedf[datedf.columns[0]] == enddate]["契約月份"].values[0]
        df = df[df["到期月份(週別)"] == weekfilter]

    except:
        if querydate.replace('-','/') in datedf[datedf.columns[0]].values:
            weekfilter = df["到期月份(週別)"].unique()[1]
        else:
            weekfilter = df["到期月份(週別)"].unique()[0]
        df = df[df["到期月份(週別)"] == weekfilter]


    #將 Call 跟 Put 分成兩個 table，並只取 "履約價","最後成交價" 這兩個欄位
    Calltable = df[df["買賣權"] == 'call'][["履約價","最後成交價"]]
    Puttable = df[df["買賣權"] == 'put'][["履約價","最後成交價"]]

    #轉換型態及資料處理
    Calltable["履約價"] = Calltable["履約價"].astype('int')
    Puttable["履約價"] = Puttable["履約價"].astype('int')

    Calltable.loc[Calltable["最後成交價"] == 0,"最後成交價"] = None
    Calltable = Calltable.dropna()
    Puttable.loc[Puttable["最後成交價"] == 0,"最後成交價"] = None
    Puttable = Puttable.dropna()

    Calltable["最後成交價"] = Calltable["最後成交價"].astype('float')
    Puttable["最後成交價"] = Puttable["最後成交價"].astype('float')

    
    return Calltable,Puttable

def catch_cost(querydate):
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0wNy0zMCAyMzowMTo0MSIsInVzZXJfaWQiOiJqZXlhbmdqYXUiLCJpcCI6IjExNC4zNC4xMjEuMTA0In0.WDAZzKGv4Du5JilaAR7o7M1whpnGaR-vMDuSeTBXhhA", # 參考登入，獲取金鑰

    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {
        "dataset": "TaiwanFuturesInstitutionalInvestors",
        "data_id": "TX",# "TXO"
        "start_date": querydate,
        "end_date": querydate,
        "token": "", # 參考登入，獲取金鑰
    }
    resp = requests.get(url, params=parameter)
    data = resp.json()
    selected_df_futures = pd.DataFrame(data["data"])
    selected_df_futures = selected_df_futures[selected_df_futures.date == querydate]
    
    filter_conditions = selected_df_futures['institutional_investors'] == '外資'
    filtered_data = selected_df_futures[filter_conditions]

    # 计算多方和空方的契约口数和契约金额
    多方契約口數 = filtered_data['long_deal_volume'].sum()
    多方契約金額 = filtered_data['long_deal_amount'].sum()
    空方契約口數 = filtered_data['short_deal_volume'].sum()
    空方契約金額 = filtered_data['short_deal_amount'].sum()

    # 计算多方和空方的平均交易价格
    多方平均交易價格 = (多方契約金額 * 1000 / 多方契約口數) / 200 if 多方契約口數 > 0 else 0
    空方平均交易價格 = (空方契約金額 * 1000 / 空方契約口數) / 200 if 空方契約口數 > 0 else 0

    # 确定外资成本价格
    if 多方契約口數 > 空方契約口數:
        外資成本價 = 多方平均交易價格
    else:
        外資成本價 = 空方平均交易價格

    return 外資成本價


def run_all():
    #建立資料庫連結
    connection = sqlite3.connect('選擇權分析資料.sqlite3')
    # 設定 API 金鑰和日期
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0wNy0zMCAyMzowMTo0MSIsInVzZXJfaWQiOiJqZXlhbmdqYXUiLCJpcCI6IjExNC4zNC4xMjEuMTA0In0.WDAZzKGv4Du5JilaAR7o7M1whpnGaR-vMDuSeTBXhhA", # 參考登入，獲取金鑰

    call_put_list = ["買權", "賣權"]
    institutional_investors_list = ["自營商", "外資", "散戶"]

    #讀取table
    df_options_futures_bs = pd.read_sql("select distinct * from options_futures_bs", connection)
    df_options_futures_daygap = pd.read_sql("select distinct * from df_options_futures_daygap", connection)

    putcallsum_sep = pd.read_sql("select distinct * from putcallsum_sep", connection)

    
    #更新 df_options_futures_bs
    maxdate = datetime.strptime(df_options_futures_bs["日期"].max(), '%Y-%m-%d')
    #print(cost_df)
    for delta_day in range((datetime.today() - maxdate).days):
        #select_date = datetime.now()

        # 初始化 DataFrame
        df_options = pd.DataFrame()
        df_futures = pd.DataFrame()


        select_date = (datetime.now() - timedelta(days=delta_day)).strftime('%Y-%m-%d')
        if select_date in df_options_futures_bs["日期"].values:
            continue
        
        try:
            # 下載選擇權select_date數據

            url = "https://api.finmindtrade.com/api/v4/data"
            parameter_options = {
                "dataset": "TaiwanOptionInstitutionalInvestors",
                "data_id": "TXO",
                "start_date": select_date,
                "end_date": select_date,
                "token": token,
            }
            resp_options = requests.get(url, params=parameter_options)
            data_options = resp_options.json()
            df_options = pd.DataFrame(data_options["data"])


            # 下載期貨數據直到滿足數據量條件


            parameter_futures = {
                "dataset": "TaiwanFuturesInstitutionalInvestors",
                "data_id": "TX",
                "start_date": select_date,
                "end_date": select_date,
                "token": token,
            }
            resp_futures = requests.get(url, params=parameter_futures)
            data_futures = resp_futures.json()
            df_futures = pd.DataFrame(data_futures["data"])

            selected_df_options = df_options[df_options['date'] == select_date]
            selected_df_futures = df_futures[df_futures['date'] == select_date]

            selected_df_options.loc[:,"long_short_gap_deal_volume"] = selected_df_options.loc[:,"long_deal_volume"] - selected_df_options.loc[:,"short_deal_volume"]
            selected_df_options.loc[:,"long_short_gap_deal_amount"] = selected_df_options.loc[:,"long_deal_amount"] - selected_df_options.loc[:,"short_deal_amount"]
            selected_df_options.loc[:,"long_short_gap_open_interest_balance_volume"] = selected_df_options.loc[:,"long_open_interest_balance_volume"] - selected_df_options.loc[:,"short_open_interest_balance_volume"]
            selected_df_options.loc[:,"long_short_gap_open_interest_balance_amount"] = selected_df_options.loc[:,"long_open_interest_balance_amount"] - selected_df_options.loc[:,"short_open_interest_balance_amount"]

            selected_df_futures.loc[:,"long_short_gap_deal_volume"] = selected_df_futures.loc[:,"long_deal_volume"] - selected_df_futures.loc[:,"short_deal_volume"]
            selected_df_futures.loc[:,"long_short_gap_deal_amount"] = selected_df_futures.loc[:,"long_deal_amount"] - selected_df_futures.loc[:,"short_deal_amount"]
            selected_df_futures.loc[:,"long_short_gap_open_interest_balance_volume"] = selected_df_futures.loc[:,"long_open_interest_balance_volume"] - selected_df_futures.loc[:,"short_open_interest_balance_volume"]
            selected_df_futures.loc[:,"long_short_gap_open_interest_balance_amount"] = selected_df_futures.loc[:,"long_open_interest_balance_amount"] - selected_df_futures.loc[:,"short_open_interest_balance_amount"]

            call_put_color_list = ["買權", "賣權"]
            institutional_investors_list = ["自營商", "外資", "散戶"]

            # 处理 selected_df_options
            df_options_list = []
            for inv in institutional_investors_list:
                for c_p in call_put_list:
                    temp_list = [inv, c_p]
                    if inv == "散戶":
                        a = selected_df_options[(selected_df_options.call_put == c_p) & (selected_df_options.institutional_investors == institutional_investors_list[0])]["long_short_gap_deal_volume"].values[0]
                        b = selected_df_options[(selected_df_options.call_put == c_p) & (selected_df_options.institutional_investors == institutional_investors_list[1])]["long_short_gap_deal_volume"].values[0]
                        temp_list.append(-1 * (a + b))  # 交易口数
                        a = selected_df_options[(selected_df_options.call_put == c_p) & (selected_df_options.institutional_investors == institutional_investors_list[0])]["long_short_gap_deal_amount"].values[0]
                        b = selected_df_options[(selected_df_options.call_put == c_p) & (selected_df_options.institutional_investors == institutional_investors_list[1])]["long_short_gap_deal_amount"].values[0]
                        temp_list.append(-1 * (a + b))  # 契约金额
                        a = selected_df_options[(selected_df_options.call_put == c_p) & (selected_df_options.institutional_investors == institutional_investors_list[0])]["long_short_gap_open_interest_balance_volume"].values[0]
                        b = selected_df_options[(selected_df_options.call_put == c_p) & (selected_df_options.institutional_investors == institutional_investors_list[1])]["long_short_gap_open_interest_balance_volume"].values[0]
                        temp_list.append(-1 * (a + b))  # 交易口数
                        a = selected_df_options[(selected_df_options.call_put == c_p) & (selected_df_options.institutional_investors == institutional_investors_list[0])]["long_short_gap_open_interest_balance_amount"].values[0]
                        b = selected_df_options[(selected_df_options.call_put == c_p) & (selected_df_options.institutional_investors == institutional_investors_list[1])]["long_short_gap_open_interest_balance_amount"].values[0]
                        temp_list.append(-1 * (a + b))  # 契约金额
                    else:
                        temp_list.append(selected_df_options[(selected_df_options.call_put == c_p) & (selected_df_options.institutional_investors == inv)]["long_short_gap_deal_volume"].values[0])  # 交易口数
                        temp_list.append(selected_df_options[(selected_df_options.call_put == c_p) & (selected_df_options.institutional_investors == inv)]["long_short_gap_deal_amount"].values[0])  # 契约金额
                        temp_list.append(selected_df_options[(selected_df_options.call_put == c_p) & (selected_df_options.institutional_investors == inv)]["long_short_gap_open_interest_balance_volume"].values[0])  # 交易口数
                        temp_list.append(selected_df_options[(selected_df_options.call_put == c_p) & (selected_df_options.institutional_investors == inv)]["long_short_gap_open_interest_balance_amount"].values[0])  # 契约金额
                    df_options_list.append(temp_list)



        except:
            print(select_date,'error')
            continue
            
        selected_result_options = pd.DataFrame(df_options_list, columns=["身份", "種類", "交易口數", "契約金額", "未平倉口數", "未平倉金額"])

        institutional_investors_list = ["自營商", "外資"]

        # 處理 selected_df_futures
        selected_futures_list = []
        for inv in institutional_investors_list:
            temp_list = [inv, "期貨"]  # 新增「種類」欄位，設定值為「期貨」
            temp_list.append(selected_df_futures[selected_df_futures.institutional_investors == inv]["long_short_gap_deal_volume"].values[0])
            temp_list.append(selected_df_futures[selected_df_futures.institutional_investors == inv]["long_short_gap_deal_amount"].values[0])
            temp_list.append(selected_df_futures[selected_df_futures.institutional_investors == inv]["long_short_gap_open_interest_balance_volume"].values[0])
            temp_list.append(selected_df_futures[selected_df_futures.institutional_investors == inv]["long_short_gap_open_interest_balance_amount"].values[0])
            selected_futures_list.append(temp_list)

        selected_result_futures = pd.DataFrame(selected_futures_list, columns=["身份", "種類", "交易口數", "契約金額", "未平倉口數", "未平倉金額"])

        
        # 假设 selected_result_options, previous_result_options, selected_result_futures, previous_result_futures 已经被定义

        # 为每个 DataFrame 添加日期列
        selected_result_options['日期'] = select_date
        selected_result_futures['日期'] = select_date

        # 重新排列每个 DataFrame 的列以确保一致性
        columns = ["日期", "身份", "種類", "交易口數", "契約金額", "未平倉口數", "未平倉金額"]
        selected_result_options = selected_result_options[columns]
        selected_result_futures = selected_result_futures[columns]

        # 合并所有 DataFrame
        combined_df = pd.concat([
            selected_result_options,
            selected_result_futures
        ], ignore_index=True)

        # 设置排序顺序
        identity_order = ['外資', '自營商', '散戶']
        category_order = ['買權', '賣權', '期貨']


        combined_df['身份'] = pd.Categorical(combined_df['身份'], categories=identity_order, ordered=True)
        combined_df['種類'] = pd.Categorical(combined_df['種類'], categories=category_order, ordered=True)


        # 对合并后的 DataFrame 进行排序
        selected_combine_df = combined_df.sort_values(by=["日期", "身份", "種類"])

        # 重置索引
        selected_combine_df = selected_combine_df.reset_index(drop=True)

        # 显示合并后的结果
        df_options_futures_bs = pd.concat([
            df_options_futures_bs,
            selected_combine_df
        ], ignore_index=True)

    # 存到資料庫
    df_options_futures_bs.to_sql('options_futures_bs', connection, if_exists='replace', index=False) 



    datearray = np.sort(df_options_futures_bs["日期"].unique())
    maxdate = df_options_futures_daygap["日期"].max()
    startidx = np.where(datearray ==maxdate)[0][0]+1
    for didx in range(startidx,len(datearray)):
        tempdf = cal_daygap(datearray[didx],datearray[didx-1],df_options_futures_bs)
        df_options_futures_daygap = pd.concat([
            df_options_futures_daygap,
            tempdf
        ], ignore_index=True)

    #存入資料庫
    df_options_futures_daygap.to_sql('df_options_futures_daygap', connection, if_exists='replace', index=False) 

    #更新 df_options_futures_bs
    maxdate = datetime.strptime(putcallsum_sep["日期"].max(), '%Y-%m-%d')
    #print(cost_df)
    for delta_day in range((datetime.today() - maxdate).days):
    #for delta_day in range(46,365):
        querydate = (datetime.now() - timedelta(days=delta_day)).strftime('%Y/%m/%d')

        try:
            
            CT,PT = callputtable(querydate)
            CT.columns = ["履約價","CT成交價"]
            PT.columns = ["履約價","PT成交價"]
            sumdf = CT.join(PT.set_index("履約價"),on=["履約價"],lsuffix='_left', rsuffix='_right')
            sumdf["CTPT差"] = np.abs(sumdf["CT成交價"] - sumdf["PT成交價"])
            sumdf["CTPT和"] = sumdf["CT成交價"] + sumdf["PT成交價"]
            sumdf = sumdf[sumdf["CTPT差"] == sumdf["CTPT差"].min()]
            sumdf = sumdf[sumdf["CTPT和"] == sumdf["CTPT和"].min()]
            putcallsum_sep = pd.concat([putcallsum_sep,pd.DataFrame([[querydate.replace('/','-')] + list(sumdf.values[0])[:3]],columns = ["日期","履約價","價平和買權成交價","價平和賣權成交價"])])
            # 存到資料庫
            # putcallsum_sep.to_sql('putcallsum_sep', connection, if_exists='replace', index=False) 
        except:
            print(delta_day,querydate,'error')
            continue
            
    # 存到資料庫
    putcallsum_sep.to_sql('putcallsum_sep', connection, if_exists='replace', index=False) 


    df_cost = pd.read_sql("select distinct * from df_cost", connection)
    #抓取最新資料的日期
    maxdate = datetime.strptime(df_cost["日期"].max(), '%Y-%m-%d')
    for i in range((datetime.now() - maxdate).days): #只需抓取特定天數
        try:
            selected_date = datetime.strftime(datetime.now() - timedelta(days=i) , '%Y-%m-%d')
            cost = catch_cost(selected_date)
            if cost>0:
                temp_df = pd.DataFrame([[selected_date,cost]],columns = ["日期","外資成本"])
                df_cost = pd.concat([df_cost,temp_df], ignore_index=True)
            df_cost.to_sql('df_cost', connection, if_exists='replace', index=False)
        except:
            pass
    df_cost.sort_values(by='日期', ascending=False, inplace=True)
    df_cost.to_sql('df_cost', connection, if_exists='replace', index=False)


    df_option_limit = pd.read_sql("select distinct * from df_option_limit", connection)
    #抓取最新資料的日期
    maxdate = datetime.strptime(df_option_limit["日期"].max(), '%Y-%m-%d')
    for i in range(320,365*2):#(datetime.now() - maxdate).days
        try:
            selected_date = datetime.strftime(datetime.now() - timedelta(days=i) , '%Y-%m-%d')
            temp_df = option_limit(selected_date)
            df_option_limit = pd.concat([df_option_limit,temp_df], ignore_index=True)
            df_option_limit.to_sql('df_option_limit', connection, if_exists='replace', index=False)
            print(i,selected_date,"done")
        except:
            #print(i,selected_date,"error")
            pass