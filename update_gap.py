import requests
import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np

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