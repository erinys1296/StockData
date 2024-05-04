import time

import streamlit as st

import pandas as pd
import sqlite3
from datetime import datetime, timedelta,time
import numpy as np


import plotly.graph_objects as go
from plotly.subplots import make_subplots

import requests

from PlotFunction import *


import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()





connection = sqlite3.connect('主圖資料.sqlite3')
connectionfuture = sqlite3.connect('FutureData.sqlite3')

#updatecheck = pd.read_sql("select distinct * from updatecheck", connection)

#if datetime.strftime(datetime.today(),'%Y/%m/%d') not in updatecheck.date.values:
    
#    "False"
#    data2 = {
#    "date": [datetime.strftime(datetime.today(),'%Y/%m/%d')],
#    "check": [1]
#    } 

#    updatecheck = updatecheck.append(pd.DataFrame(data2))
#    updatecheck.to_sql('updatecheck', connection, if_exists='replace', index=False) 

#else:
    
#    "True"

#週線
url = "https://api.finmindtrade.com/api/v4/data?"
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0wNy0zMCAyMzowMTo0MSIsInVzZXJfaWQiOiJqZXlhbmdqYXUiLCJpcCI6IjExNC4zNC4xMjEuMTA0In0.WDAZzKGv4Du5JilaAR7o7M1whpnGaR-vMDuSeTBXhhA", # 參考登入，獲取金鑰

parameter = {
"dataset": "TaiwanStockPrice",
"data_id": "TAIEX",
"start_date": "2020-04-02",
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

#taiex = pd.read_sql("select distinct * from taiex", connection, parse_dates=['日期'], index_col=['日期'])
#taiex_vol = pd.read_sql("select distinct * from taiex_vol", connection, parse_dates=['日期'], index_col=['日期'])

#taiex
#taiex_vol
cost_df = pd.read_sql("select distinct Date as [日期], Cost as [外資成本] from cost", connection, parse_dates=['日期'], index_col=['日期']).dropna()
cost_df["外資成本"] = cost_df["外資成本"].astype('int')
limit_df = pd.read_sql("select distinct * from [limit]", connection, parse_dates=['日期'], index_col=['日期'])

inves_limit = limit_df[limit_df["身份別"] == "外資"][['上極限', '下極限']]
dealer_limit = limit_df[limit_df["身份別"] == "自營商"][['上極限', '下極限']]

inves_limit.columns = ["外資上極限","外資下極限"]
dealer_limit.columns = ["自營商上極限","自營商下極限"]

kbars = taiex_fin.join(cost_df).join(dealer_limit).join(inves_limit)

enddate = pd.read_sql("select * from end_date order by 最後結算日 desc", connection, parse_dates=['最後結算日'])


holidf = pd.read_sql("select * from holiday", connection)

holilist = [str(holiday) for holiday in holidf[~(holidf["說明"].str.contains('開始交易') | holidf["說明"].str.contains('最後交易'))]["日期"].values]
#holilist["日期"] = pd.to_datetime(holilist["日期"])

ordervolumn = pd.read_sql("select distinct * from ordervolumn", connection, parse_dates=['日期'], index_col=['日期'])
putcallsum = pd.read_sql("select 日期, max(價平和) as 價平和 from putcallsum group by 日期", connection, parse_dates=['日期'], index_col=['日期'])
putcallsum_month = pd.read_sql("select 日期, max(月選擇權價平和) as 月價平和 from putcallsum_month group by 日期", connection, parse_dates=['日期'], index_col=['日期'])
putcallgap = pd.read_sql("select 日期, max(價外買賣權價差) as 價外買賣權價差 from putcallgap group by 日期", connection, parse_dates=['日期'], index_col=['日期'])


print(putcallsum_month.tail())
kbars = kbars.join(ordervolumn).join(putcallsum).join(putcallsum_month)
kbars = kbars.join(putcallgap)

# 計算布林帶指標
kbars['20MA'] = kbars['收盤指數'].rolling(20).mean()
kbars['std'] = kbars['收盤指數'].rolling(20).std()
kbars['60MA'] = kbars['收盤指數'].rolling(60).mean()
kbars['200MA'] = kbars['收盤指數'].rolling(200).mean()
kbars['upper_band'] = kbars['20MA'] + 2 * kbars['std']
kbars['lower_band'] = kbars['20MA'] - 2 * kbars['std']
kbars['upper_band1'] = kbars['20MA'] + 1 * kbars['std']
kbars['lower_band1'] = kbars['20MA'] - 1 * kbars['std']


kbars['IC'] = kbars['收盤指數'] + 2 * kbars['收盤指數'].shift(1) - kbars['收盤指數'].shift(3) -kbars['收盤指數'].shift(4)

kbars['月價平和日差'] = kbars['月價平和'] - kbars['月價平和'].shift(1)

# 在k线基础上计算KDF，并将结果存储在df上面(k,d,j)
low_list = kbars['最低指數'].rolling(9, min_periods=9).min()
low_list.fillna(value=kbars['最低指數'].expanding().min(), inplace=True)
high_list = kbars['最高指數'].rolling(9, min_periods=9).max()
high_list.fillna(value=kbars['最高指數'].expanding().max(), inplace=True)
rsv = (kbars['收盤指數'] - low_list) / (high_list - low_list) * 100
kbars['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
kbars['D'] = kbars['K'].ewm(com=2).mean()

enddatemonth = enddate[~enddate["契約月份"].str.contains("W")]['最後結算日']
kbars['end_low'] = 0
kbars['end_high'] = 0
#kbars
for datei in kbars.index:
    
    month_low = kbars[(kbars.index >= enddatemonth[enddatemonth<datei].max())&(kbars.index<=datei)]["最低指數"].min()
    month_high = kbars[(kbars.index >= enddatemonth[enddatemonth<datei].max())&(kbars.index<=datei)]['最高指數'].max()
    kbars.loc[datei,'end_low'] =  kbars.loc[datei,'最高指數'] - month_low
    kbars.loc[datei,'end_high'] = kbars.loc[datei,'最低指數'] - month_high
    
kbars["MAX_MA"] = kbars["最高指數"] - kbars["20MA"]
kbars["MIN_MA"] = kbars["最低指數"] - kbars["20MA"]

#詢問
ds = 2
kbars['uline'] = kbars['最高指數'].rolling(ds, min_periods=1).max()
kbars['dline'] = kbars['最低指數'].rolling(ds, min_periods=1).min()

kbars["all_kk"] = 0
barssince5 = 0
barssince6 = 0
kbars['labelb'] = 1
kbars = kbars[~kbars.index.duplicated(keep='first')]
for i in range(2,len(kbars.index)):
    try:
        #(kbars.loc[kbars.index[i],'收盤指數'] > kbars.loc[kbars.index[i-1],"uline"])
        condition51 = (kbars.loc[kbars.index[i-1],"最高指數"] < kbars.loc[kbars.index[i-2],"最低指數"] ) and (kbars.loc[kbars.index[i],"最低指數"] > kbars.loc[kbars.index[i-1],"最高指數"] )
        #condition52 = (kbars.loc[kbars.index[i-1],'收盤指數'] < kbars.loc[kbars.index[i-2],"最低指數"]) and (kbars.loc[kbars.index[i-1],'成交金額'] > kbars.loc[kbars.index[i-2],'成交金額']) and (kbars.loc[kbars.index[i],'收盤指數']>kbars.loc[kbars.index[i-1],"最高指數"] )
        condition53 = (kbars.loc[kbars.index[i],'收盤指數'] > kbars.loc[kbars.index[i-1],"uline"]) and (kbars.loc[kbars.index[i-1],'收盤指數'] <= kbars.loc[kbars.index[i-1],"uline"])

        condition61 = (kbars.loc[kbars.index[i-1],"最低指數"] > kbars.loc[kbars.index[i-2],"最高指數"] ) and (kbars.loc[kbars.index[i],"最高指數"] < kbars.loc[kbars.index[i-1],"最低指數"] )
        #condition62 = (kbars.loc[kbars.index[i-1],'收盤指數'] > kbars.loc[kbars.index[i-2],"最高指數"]) and (kbars.loc[kbars.index[i-1],'成交金額'] > kbars.loc[kbars.index[i-2],'成交金額']) and (kbars.loc[kbars.index[i],'收盤指數']<kbars.loc[kbars.index[i-1],"最低指數"] )
        condition63 = (kbars.loc[kbars.index[i],'收盤指數'] < kbars.loc[kbars.index[i-1],"dline"]) and (kbars.loc[kbars.index[i-1],'收盤指數'] >= kbars.loc[kbars.index[i-1],"dline"])
    except:
        condition51 = True
        condition52 = True
        condition53 = True
        condition61 = True
        condition63 = True
    condition54 = condition51 or condition53 #or condition52
    condition64 = condition61 or condition63 #or condition62 

    #kbars['labelb'] = np.where((kbars['收盤指數']> kbars['upper_band1']) , 1, np.where((kbars['收盤指數']< kbars['lower_band1']),-1,1))

    #print(i)
    if kbars.loc[kbars.index[i],'收盤指數'] > kbars.loc[kbars.index[i],'upper_band1']:
        kbars.loc[kbars.index[i],'labelb'] = 1
    elif kbars.loc[kbars.index[i],'收盤指數'] < kbars.loc[kbars.index[i],'lower_band1']:
        kbars.loc[kbars.index[i],'labelb'] = -1
    else:
        kbars.loc[kbars.index[i],'labelb'] = kbars.loc[kbars.index[i-1],'labelb']

    if condition54 == True:
        barssince5 = 1
    else:
        barssince5 += 1

    if condition64 == True:
        barssince6 = 1
    else:
        barssince6 += 1


    if barssince5 < barssince6:
        kbars.loc[kbars.index[i],"all_kk"] = 1
    else:
        kbars.loc[kbars.index[i],"all_kk"] = -1


max_days20_list =  []
max_days20_x = []
min_days20_list =  []
min_days20_x = []

for dateidx in range(0,len(kbars.index[-59:])):

    try:
        datei = kbars.index[-59:][dateidx]
        days19 = kbars[(kbars.index> kbars.index[-79:][dateidx]) & (kbars.index<datei )]
        max_days19 = days19["九點累積委託賣出數量"].values.max()
        min_days19 = days19["九點累積委託賣出數量"].values.min()
        curday = kbars[kbars.index == datei]["九點累積委託賣出數量"].values[0]
        yesday = kbars[kbars.index == kbars.index[-59:][dateidx-1]]["九點累積委託賣出數量"].values[0]
        tomday = kbars[kbars.index == kbars.index[-59:][dateidx+1]]["九點累積委託賣出數量"].values[0]

        if curday >= max_days19 and curday > yesday and curday > tomday:
            max_days20_list.append(curday)
            max_days20_x.append(datei)

        if curday <= min_days19 and curday < yesday and curday < tomday:
            min_days20_list.append(curday)
            min_days20_x.append(datei)
    except:
        pass

   

#max_days20_list
#max_days20

#max_days20_x
notshowdate = []
for datei in enddate[~enddate["契約月份"].str.contains("W")]['最後結算日']:
    try:
        kbarsdi = np.where(kbars.index == datei)[0]
        notshowdate.append(kbars.index[kbarsdi+1][0])
    except:
        continue
#kbars = kbars.dropna()
kbars = kbars[kbars.index > kbars.index[-60]]

#kbars['labelb'] = np.where(kbars['收盤指數']< kbars['lower_band1'], -1, 1)



def fillcol(label):
    if label >= 1:
        return 'rgba(0,250,0,0.2)'
    else:
        return 'rgba(0,256,256,0.2)'

ICdate = []
datechecki = 1
#(kbars['IC'].index[-1] + timedelta(days = 1)).weekday() == 5
while (kbars['IC'].index[-1] + timedelta(days = datechecki)).weekday() in [5,6] or (kbars['IC'].index[-1] + timedelta(days = datechecki)) in pd.to_datetime(holidf["日期"]).values:
    datechecki +=1
ICdate.append((kbars['IC'].index[-1] + timedelta(days = datechecki)))
datechecki +=1
while (kbars['IC'].index[-1] + timedelta(days = datechecki)).weekday() in [5,6] or (kbars['IC'].index[-1] + timedelta(days = datechecki)) in pd.to_datetime(holidf["日期"]).values:
    datechecki +=1
ICdate.append((kbars['IC'].index[-1] + timedelta(days = datechecki)))



#[kbars['IC'].index[-1] + timedelta(days = 1),kbars['IC'].index[-1] + timedelta(days = 2)]

CPratio = pd.read_sql("select distinct * from putcallratio", connection, parse_dates=['日期'], index_col=['日期'])
CPratio = CPratio[CPratio.index>kbars.index[0]]

bank8 = pd.read_sql("select distinct * from bank", connection, parse_dates=['日期'], index_col=['日期'])
bank8 = bank8[bank8.index>kbars.index[0]]

dfMTX = pd.read_sql("select distinct * from dfMTX", connection, parse_dates=['Date'], index_col=['Date'])
dfMTX = dfMTX[dfMTX.index>kbars.index[0]]

futdf = pd.read_sql("select distinct * from futdf", connection, parse_dates=['日期'], index_col=['日期'])
futdf = futdf[futdf.index>kbars.index[0]]

TXOOIdf = pd.read_sql("select distinct * from TXOOIdf", connection, parse_dates=['日期'], index_col=['日期'])
TXOOIdf = TXOOIdf[TXOOIdf.index>kbars.index[0]]

dfbuysell = pd.read_sql("select distinct * from dfbuysell order by Date", connection, parse_dates=['Date'], index_col=['Date'])
dfbuysell = dfbuysell[dfbuysell.index>kbars.index[0]]

dfMargin = pd.read_sql("select distinct * from dfMargin order by Date", connection, parse_dates=['Date'], index_col=['Date'])
dfMargin = dfMargin[dfMargin.index>kbars.index[0]]
st.set_page_config(layout="wide")




st.title('即時盤')

cost_df = pd.read_sql("select distinct Date as [日期], Cost as [外資成本] from cost", connection, parse_dates=['日期']).dropna()
cost_df["外資成本"] = cost_df["外資成本"].astype('int')
limit_df = pd.read_sql("select distinct * from [limit]", connection, parse_dates=['日期'])

inves_limit = limit_df[limit_df["身份別"] == "外資"][['日期','上極限', '下極限']]
dealer_limit = limit_df[limit_df["身份別"] == "自營商"][['日期','上極限', '下極限']]

inves_limit.columns = ['日期',"外資上極限","外資下極限"]
dealer_limit.columns = ['日期',"自營商上極限","自營商下極限"]

startFuture = datetime.strftime(datetime.today()- timedelta(days=20),'%Y-%m-%d')
endFuture = datetime.strftime(datetime.today(),'%Y-%m-%d')
FutureData = pd.read_sql("select distinct * from futurehourly", connectionfuture, parse_dates=['ts'], index_col=['ts'])#get_future_raw_data(startFuture,endFuture)
#FutureData
df_ts = FutureData.reset_index()
FutureData = FutureData.reset_index()
FutureData.loc[(FutureData.ts.dt.hour<14)&(FutureData.ts.dt.hour>=8),'ts'] = FutureData.loc[(FutureData.ts.dt.hour<14)&(FutureData.ts.dt.hour>=8)].ts - timedelta(minutes=46)
FutureData.index = FutureData.ts

#FutureData.index = FutureData.ts
FutureData.date = pd.to_datetime(FutureData.index)
FutureData["hourdate"] = np.array(FutureData.date.date.astype(str)) +  np.array(FutureData.date.hour.astype(str))
FutureData['date'] = np.array(pd.to_datetime(FutureData.index).values)
#FutureData
FutureData = FutureData.dropna(subset = ['Open'])
FutureData.index = FutureData['date']
#FutureData
tempdf = FutureData[['hourdate','Volume']]
tempdf = tempdf.reset_index()
tempdf = tempdf[['hourdate','Volume']]
#tempdf

Final60Tdata = FutureData.groupby('hourdate').max()[["High"]].join(FutureData.groupby('hourdate').min()[["Low"]]).join(tempdf.groupby('hourdate').sum()[["Volume"]])
#Final60Tdata
#Final60Tdata.index = Final60Tdata['hourdate']
tempopen = FutureData.loc[FutureData.groupby('hourdate').min()['date'].values]
tempopen.index = tempopen.hourdate
tempclose = FutureData.loc[FutureData.groupby('hourdate').max()['date'].values]
tempclose.index = tempclose.hourdate
#tempvolume = FutureData.loc[FutureData.groupby('hourdate').sum()['Volume'].values]
#tempvolume.index = tempvolume.hourdate
Final60Tdata = Final60Tdata.join(tempopen[["Open",'date']]).join(tempclose[["Close"]])
Final60Tdata.index = Final60Tdata.date
Final60Tdata.columns = ['max','min','Volume','open','date','close']


Final60Tdata['dateonly'] = pd.to_datetime((Final60Tdata.date- timedelta(hours=15)).dt.date)
Final60Tdata.loc[(Final60Tdata.date - timedelta(hours=13)).dt.weekday ==6,'dateonly'] = pd.to_datetime((Final60Tdata[(Final60Tdata.date - timedelta(hours=13)).dt.weekday ==6].date- timedelta(hours=63)).dt.date)
Final60Tdata.loc[Final60Tdata.dateonly ==datetime(2023, 9, 29, 0, 0),'dateonly'] = datetime(2023, 9, 28, 0, 0)
Final60Tdata.loc[Final60Tdata.dateonly ==datetime(2023, 10, 10, 0, 0),'dateonly'] = datetime(2023, 10, 6, 0, 0)
Final60Tdata.loc[Final60Tdata.dateonly ==datetime(2024, 1, 1, 0, 0),'dateonly'] = datetime(2023, 12, 29, 0, 0)
Final60Tdata.loc[Final60Tdata.dateonly ==datetime(2024, 2,14, 0, 0),'dateonly'] = datetime(2024, 2, 5, 0, 0)
Final60Tdata.loc[Final60Tdata.dateonly ==datetime(2024, 2,28, 0, 0),'dateonly'] = datetime(2024, 2, 27, 0, 0)
Final60Tdata.loc[Final60Tdata.dateonly ==datetime(2024, 4,5, 0, 0),'dateonly'] = datetime(2024, 4, 3, 0, 0)
Final60Tdata.loc[Final60Tdata.dateonly ==datetime(2024, 5,1, 0, 0),'dateonly'] = datetime(2024, 4, 30, 0, 0)
    
#Final60Tdata.loc[Final60Tdata.dateonly ==datetime(2023, 10, 10, 0, 0),'dateonly'] = datetime(2023, 10, 7, 0, 0)
Final60Tdata = pd.merge(Final60Tdata, cost_df, left_on="dateonly", right_on="日期", how='left')
Final60Tdata = pd.merge(Final60Tdata, inves_limit, on="日期", how='left')
Final60Tdata = pd.merge(Final60Tdata, dealer_limit, on="日期", how='left')
#Final60Tdata.loc[Final60Tdata["外資成本"]==None,['外資成本','外資上極限','外資下極限','自營商上極限','自營商下極限']] = [16347,16673,16227,16645,16155]
Final60Tdata.loc[Final60Tdata.date.dt.minute == 1 ,'date'] = Final60Tdata.loc[Final60Tdata.date.dt.minute == 1 ,'date'] - timedelta(minutes = 1)
Final60Tdata.index = Final60Tdata.date
Final60Tdata = Final60Tdata.sort_index()
#Final60Tdata
Final60Tdata[Final60Tdata.index == Final60Tdata.index[-1]][["日期","外資成本","外資上極限","外資下極限","自營商上極限","自營商下極限"]]





# 計算布林帶指標
# 計算布林帶指標
Final60Tdata['20MA'] = Final60Tdata['close'].rolling(20).mean()
#Final60Tdata['60MA'] = Final60Tdata['close'].rolling(60).mean()
#Final60Tdata['200MA'] = Final60Tdata['close'].rolling(200).mean()
Final60Tdata['std'] = Final60Tdata['close'].rolling(20).std()
Final60Tdata['upper_band'] = Final60Tdata['20MA'] + 2 * Final60Tdata['std']
Final60Tdata['lower_band'] = Final60Tdata['20MA'] - 2 * Final60Tdata['std']
Final60Tdata['upper_band1'] = Final60Tdata['20MA'] + 1 * Final60Tdata['std']
Final60Tdata['lower_band1'] = Final60Tdata['20MA'] - 1 * Final60Tdata['std']

Final60Tdata['IC'] = Final60Tdata['close'] + 2 * Final60Tdata['close'].shift(1) - Final60Tdata['close'].shift(3) -Final60Tdata['close'].shift(4)



# 在k线基础上计算KDF，并将结果存储在df上面(k,d,j)
low_list = Final60Tdata['min'].rolling(9, min_periods=9).min()
low_list.fillna(value=Final60Tdata['min'].expanding().min(), inplace=True)
high_list = Final60Tdata['max'].rolling(9, min_periods=9).max()
high_list.fillna(value=Final60Tdata['max'].expanding().max(), inplace=True)
rsv = (Final60Tdata['close'] - low_list) / (high_list - low_list) * 100
Final60Tdata['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
Final60Tdata['D'] = Final60Tdata['K'].ewm(com=2).mean()

enddatemonth = enddate[~enddate["契約月份"].str.contains("W")]['最後結算日']
Final60Tdata['end_low'] = 0
Final60Tdata['end_high'] = 0

#詢問
ds = 2
Final60Tdata['uline'] = Final60Tdata['max'].rolling(ds, min_periods=1).max()
Final60Tdata['dline'] = Final60Tdata['min'].rolling(ds, min_periods=1).min()

Final60Tdata["all_kk"] = 0
barssince5 = 0
barssince6 = 0
Final60Tdata['labelb'] = 1
Final60Tdata = Final60Tdata[~Final60Tdata.index.duplicated(keep='first')]

for i in range(2,len(Final60Tdata.index)):
    try:
        #(Final60Tdata.loc[Final60Tdata.index[i],'close'] > Final60Tdata.loc[Final60Tdata.index[i-1],"uline"])
        condition51 = (Final60Tdata.loc[Final60Tdata.index[i-1],"max"] < Final60Tdata.loc[Final60Tdata.index[i-2],"min"] ) and (Final60Tdata.loc[Final60Tdata.index[i],"min"] > Final60Tdata.loc[Final60Tdata.index[i-1],"max"] )
        #condition52 = (Final60Tdata.loc[Final60Tdata.index[i-1],'close'] < Final60Tdata.loc[Final60Tdata.index[i-2],"min"]) and (Final60Tdata.loc[Final60Tdata.index[i-1],'成交金額'] > Final60Tdata.loc[Final60Tdata.index[i-2],'成交金額']) and (Final60Tdata.loc[Final60Tdata.index[i],'close']>Final60Tdata.loc[Final60Tdata.index[i-1],"max"] )
        condition53 = (Final60Tdata.loc[Final60Tdata.index[i],'close'] > Final60Tdata.loc[Final60Tdata.index[i-1],"uline"]) and (Final60Tdata.loc[Final60Tdata.index[i-1],'close'] <= Final60Tdata.loc[Final60Tdata.index[i-1],"uline"])

        condition61 = (Final60Tdata.loc[Final60Tdata.index[i-1],"min"] > Final60Tdata.loc[Final60Tdata.index[i-2],"max"] ) and (Final60Tdata.loc[Final60Tdata.index[i],"max"] < Final60Tdata.loc[Final60Tdata.index[i-1],"min"] )
        #condition62 = (Final60Tdata.loc[Final60Tdata.index[i-1],'close'] > Final60Tdata.loc[Final60Tdata.index[i-2],"max"]) and (Final60Tdata.loc[Final60Tdata.index[i-1],'成交金額'] > Final60Tdata.loc[Final60Tdata.index[i-2],'成交金額']) and (Final60Tdata.loc[Final60Tdata.index[i],'close']<Final60Tdata.loc[Final60Tdata.index[i-1],"min"] )
        condition63 = (Final60Tdata.loc[Final60Tdata.index[i],'close'] < Final60Tdata.loc[Final60Tdata.index[i-1],"dline"]) and (Final60Tdata.loc[Final60Tdata.index[i-1],'close'] >= Final60Tdata.loc[Final60Tdata.index[i-1],"dline"])
    except:
        condition51 = True
        condition52 = True
        condition53 = True
        condition61 = True
        condition63 = True
    condition54 = condition51 or condition53 #or condition52
    condition64 = condition61 or condition63 #or condition62 

    #Final60Tdata['labelb'] = np.where((Final60Tdata['close']> Final60Tdata['upper_band1']) , 1, np.where((Final60Tdata['close']< Final60Tdata['lower_band1']),-1,1))

    #print(i)
    if Final60Tdata.loc[Final60Tdata.index[i],'close'] > Final60Tdata.loc[Final60Tdata.index[i],'upper_band1']:
        Final60Tdata.loc[Final60Tdata.index[i],'labelb'] = 1
    elif Final60Tdata.loc[Final60Tdata.index[i],'close'] < Final60Tdata.loc[Final60Tdata.index[i],'lower_band1']:
        Final60Tdata.loc[Final60Tdata.index[i],'labelb'] = -1
    else:
        Final60Tdata.loc[Final60Tdata.index[i],'labelb'] = Final60Tdata.loc[Final60Tdata.index[i-1],'labelb']

    if condition54 == True:
        barssince5 = 1
    else:
        barssince5 += 1

    if condition64 == True:
        barssince6 = 1
    else:
        barssince6 += 1


    if barssince5 < barssince6:
        Final60Tdata.loc[Final60Tdata.index[i],"all_kk"] = 1
    else:
        Final60Tdata.loc[Final60Tdata.index[i],"all_kk"] = -1

Final60Tdata = Final60Tdata[Final60Tdata.index>Final60Tdata.index[-130]]
IChour = []
finalhour = list(Final60Tdata['IC'].index)[-1]
plusi = 1
while (finalhour + timedelta(hours = plusi)).hour in [6,7,13,14] or  (finalhour + timedelta(hours = plusi)-timedelta(hours = 5)).weekday in [5,6]:
    plusi = plusi + 1
IChour.append((finalhour + timedelta(hours = plusi)).strftime("%m-%d-%Y %H:%M"))
IChour.append((finalhour + timedelta(hours = plusi+1)).strftime("%m-%d-%Y %H:%M"))

#Final60Tdata.index = Final60Tdata.index.astype('str')
Final60Tdata.index = Final60Tdata.index.strftime("%m-%d-%Y %H:%M")
#Final60Tdata




#Final60Tdata
fig3_1 = make_subplots(
    rows = 2, 
    cols = 1, 
    horizontal_spacing = 0.2,
    vertical_spacing=0.2,subplot_titles = ["TAIEX FUTURE 60分","TAIEX FUTURE 300分"],
    shared_yaxes=False,
    
    #subplot_titles=subtitle,
    #y_title = "test"# subtitle,
    specs = [[{"secondary_y":True}]]*2
)

### 成交量圖製作 ###
volume_colors1 = [red_color if Final60Tdata['close'][i] > Final60Tdata['close'][i-1] else green_color for i in range(len(Final60Tdata['close']))]
volume_colors1[0] = green_color

#fig.add_trace(go.Bar(x=Final60Tdata.index, y=Final60Tdata['成交金額'], name='成交數量', marker=dict(color=volume_colors),showlegend=False), row=optvrank[0], col=1)
fig3_1.add_trace(go.Bar(x=Final60Tdata.index, y=Final60Tdata['Volume'], name='成交量', marker=dict(color=volume_colors1)), row=1, col=1)
#Final60Tdata.index = Final60Tdata.index - timedelta(hours = 6)

checkb = Final60Tdata["labelb"].values[0]
bandstart = 1
bandidx = 1
checkidx = 0
while bandidx < len(Final60Tdata["labelb"].values):
    #checkidx = bandidx
    bandstart = bandidx-1
    checkidx = bandstart+1
    if checkidx >=len(Final60Tdata["labelb"].values)-1:
        break
    while Final60Tdata["labelb"].values[checkidx] == Final60Tdata["labelb"].values[checkidx+1]:
        checkidx +=1
        if checkidx >=len(Final60Tdata["labelb"].values)-1:
            break
    bandend = checkidx+1
    #print(bandstart,bandend)
    if Final60Tdata["labelb"].values[bandstart+1] == 1:
        fig3_1.add_traces(go.Scatter(x=Final60Tdata.index[bandstart:bandend], y = Final60Tdata['lower_band'].values[bandstart:bandend],
                                    line = dict(color='rgba(0,0,0,0)'),showlegend=False,hoverinfo='none'),rows=[1], cols=[1], secondary_ys= [True])
            
        fig3_1.add_traces(go.Scatter(x=Final60Tdata.index[bandstart:bandend], y = Final60Tdata['upper_band'].values[bandstart:bandend],
                                    line = dict(color='rgba(0,0,0,0)'),
                                    fill='tonexty', 
                                    fillcolor = 'rgba(256,256,0,0.2)',showlegend=False,hoverinfo='none'
                                    ),rows=[1], cols=[1], secondary_ys= [True])
    else:


        fig3_1.add_traces(go.Scatter(x=Final60Tdata.index[bandstart:bandend], y = Final60Tdata['lower_band'].values[bandstart:bandend],
                                    line = dict(color='rgba(0,0,0,0)'),showlegend=False,hoverinfo='none'),rows=[1], cols=[1], secondary_ys= [True])
            
        fig3_1.add_traces(go.Scatter(x=Final60Tdata.index[bandstart:bandend], y = Final60Tdata['upper_band'].values[bandstart:bandend],
                                    line = dict(color='rgba(0,0,0,0)'),
                                    fill='tonexty', 
                                    fillcolor = 'rgba(137, 207, 240,0.2)',showlegend=False,hoverinfo='none'
                                    ),rows=[1], cols=[1], secondary_ys= [True])
    bandidx =checkidx +1
    if bandidx >=len(Final60Tdata["labelb"].values):
        break




fig3_1.add_trace(go.Scatter(x=Final60Tdata.index,
                        y=Final60Tdata['外資成本'],
                        mode='lines',
                        #line=dict(color='green'),
                        name='外資成本'),row=1, col=1, secondary_y= True)

#fig3_1.add_trace(go.Scatter(x=Final60Tdata.index,
#                        y=Final60Tdata['外資下極限'],
#                        mode='lines',
#                        #line=dict(color='green'),
#                        name='外資下極限'),row=1, col=1)



fig3_1.add_traces(go.Scatter(x=Final60Tdata.index, y = Final60Tdata['外資上極限'].values,
                                    line = dict(color='rgba(0,0,0,0)'),showlegend=False,name='外資上極限'),rows=[1], cols=[1], secondary_ys= [True])
            
fig3_1.add_traces(go.Scatter(x=Final60Tdata.index, y = Final60Tdata['自營商上極限'].values,
                            line = dict(color='rgba(0,0,0,0)'),
                            fill='tonexty', 
                            fillcolor = 'rgba(0,0,256,0.2)',showlegend=False,name='自營商上極限'
                            ),rows=[1], cols=[1], secondary_ys= [True])
fig3_1.add_traces(go.Scatter(x=Final60Tdata.index, y = Final60Tdata['外資下極限'].values,
                                    line = dict(color='rgba(0,0,0,0)'),showlegend=False,name='外資下極限'),rows=[1], cols=[1], secondary_ys= [True])
            
fig3_1.add_traces(go.Scatter(x=Final60Tdata.index, y = Final60Tdata['自營商下極限'].values,
                            line = dict(color='rgba(0,0,0,0)'),
                            fill='tonexty', 
                            fillcolor = 'rgba(256,0,0,0.2)',showlegend=False,name='自營商下極限'
                            ),rows=[1], cols=[1], secondary_ys= [True])



fig3_1.add_trace(go.Scatter(x=Final60Tdata.index,
                        y=Final60Tdata['20MA'],
                        mode='lines',
                        line=dict(color='green'),
                        name='MA20'),row=1, col=1, secondary_y= True)
#fig3_1.add_trace(go.Scatter(x=Final60Tdata.index,
#                        y=Final60Tdata['200MA'],
#                        mode='lines',
#                        line=dict(color='blue'),
#                        name='MA200'),row=1, col=1)
#fig3_1.add_trace(go.Scatter(x=Final60Tdata.index,
#                        y=Final60Tdata['60MA'],
#                        mode='lines',
#                        line=dict(color='orange'),
#                        name='MA60'),row=1, col=1)

#fig3_1.add_trace(go.Scatter(x=np.array(list(Final60Tdata['IC'].index)[2:]+IChour).astype('str'),
#                        y=Final60Tdata['IC'].values,
#                        mode='lines',
#                        line=dict(color='orange'),
#                        name='IC操盤線'),row=1, col=1, secondary_y= True)





### K線圖製作 ###
fig3_1.add_trace(
    go.Candlestick(
        x=Final60Tdata[(Final60Tdata['all_kk'] == -1)&(Final60Tdata['close'] >Final60Tdata['open'] )].index,
        open=Final60Tdata[(Final60Tdata['all_kk'] == -1)&(Final60Tdata['close'] >Final60Tdata['open'] )]['open'],
        high=Final60Tdata[(Final60Tdata['all_kk'] == -1)&(Final60Tdata['close'] >Final60Tdata['open'] )]['max'],
        low=Final60Tdata[(Final60Tdata['all_kk'] == -1)&(Final60Tdata['close'] >Final60Tdata['open'] )]['min'],
        close=Final60Tdata[(Final60Tdata['all_kk'] == -1)&(Final60Tdata['close'] >Final60Tdata['open'] )]['close'],
        increasing_line_color=decreasing_color,
        increasing_fillcolor=no_color, #fill_increasing_color(Final60Tdata.index>Final60Tdata.index[50])
        decreasing_line_color=decreasing_color,
        decreasing_fillcolor=no_color,#decreasing_color,
        line=dict(width=2),
        name='OHLC',showlegend=False
    )#,

    ,row=1, col=1, secondary_y= True
)


fig3_1.add_trace(
    go.Candlestick(
        x=Final60Tdata[(Final60Tdata['all_kk'] == 1)&(Final60Tdata['close'] >Final60Tdata['open'] )].index,
        open=Final60Tdata[(Final60Tdata['all_kk'] == 1)&(Final60Tdata['close'] >Final60Tdata['open'] )]['open'],
        high=Final60Tdata[(Final60Tdata['all_kk'] == 1)&(Final60Tdata['close'] >Final60Tdata['open'] )]['max'],
        low=Final60Tdata[(Final60Tdata['all_kk'] == 1)&(Final60Tdata['close'] >Final60Tdata['open'] )]['min'],
        close=Final60Tdata[(Final60Tdata['all_kk'] == 1)&(Final60Tdata['close'] >Final60Tdata['open'] )]['close'],
        increasing_line_color=increasing_color,
        increasing_fillcolor=no_color, #fill_increasing_color(Final60Tdata.index>Final60Tdata.index[50])
        decreasing_line_color=increasing_color,
        decreasing_fillcolor=no_color,#decreasing_color,
        line=dict(width=1),
        name='OHLC',showlegend=False
    )#,

    ,row=1, col=1, secondary_y= True
)

### K線圖製作 ###
fig3_1.add_trace(
    go.Candlestick(
        x=Final60Tdata[(Final60Tdata['all_kk'] == -1)&(Final60Tdata['close'] <Final60Tdata['open'] )].index,
        open=Final60Tdata[(Final60Tdata['all_kk'] == -1)&(Final60Tdata['close'] <Final60Tdata['open'] )]['open'],
        high=Final60Tdata[(Final60Tdata['all_kk'] == -1)&(Final60Tdata['close'] <Final60Tdata['open'] )]['max'],
        low=Final60Tdata[(Final60Tdata['all_kk'] == -1)&(Final60Tdata['close'] <Final60Tdata['open'] )]['min'],
        close=Final60Tdata[(Final60Tdata['all_kk'] == -1)&(Final60Tdata['close'] <Final60Tdata['open'] )]['close'],
        increasing_line_color=decreasing_color,
        increasing_fillcolor=decreasing_color, #fill_increasing_color(Final60Tdata.index>Final60Tdata.index[50])
        decreasing_line_color=decreasing_color,
        decreasing_fillcolor=decreasing_color,#decreasing_color,
        line=dict(width=1),
        name='OHLC',showlegend=False
    )#,

    ,row=1, col=1, secondary_y= True
)


fig3_1.add_trace(
    go.Candlestick(
        x=Final60Tdata[(Final60Tdata['all_kk'] == 1)&(Final60Tdata['close'] <Final60Tdata['open'] )].index,
        open=Final60Tdata[(Final60Tdata['all_kk'] == 1)&(Final60Tdata['close'] <Final60Tdata['open'] )]['open'],
        high=Final60Tdata[(Final60Tdata['all_kk'] == 1)&(Final60Tdata['close'] <Final60Tdata['open'] )]['max'],
        low=Final60Tdata[(Final60Tdata['all_kk'] == 1)&(Final60Tdata['close'] <Final60Tdata['open'] )]['min'],
        close=Final60Tdata[(Final60Tdata['all_kk'] == 1)&(Final60Tdata['close'] <Final60Tdata['open'] )]['close'],
        increasing_line_color=increasing_color,
        increasing_fillcolor=increasing_color, #fill_increasing_color(Final60Tdata.index>FinalＷeekdata.index[50])
        decreasing_line_color=increasing_color,
        decreasing_fillcolor=increasing_color,#decreasing_color,
        line=dict(width=1),
        name='OHLC',showlegend=False
    )#,

    ,row=1, col=1, secondary_y= True
)


start_times = [timedelta(hours=1), timedelta(hours=8, minutes=45),
            timedelta(hours=15), timedelta(hours=20)]
data_300 = []



current_date = datetime.combine(df_ts.iloc[0]['ts'].date(), time(0, 0, 0))

while current_date.date() <= df_ts['ts'].iloc[-1].date():
    for start_time in start_times:
        start = current_date + start_time
        end = start + timedelta(hours=5)

        period = df_ts[(df_ts['ts'] > start) & (df_ts['ts'] < end)].dropna(subset = ['Open'])

        if period.shape[0]:
            data_300.append([start, period.iloc[0]['Open'], period.iloc[-1]['Close'], period['High'].max(),
                            period['Low'].min(),period['Volume'].sum()])
        else:
            data_300.append([start, None, None, None, None,None])

    current_date += timedelta(days=1)

df_300 = pd.DataFrame(data_300, columns=['ts', 'open','close','max','min','Volume'])
df_300['date'] = df_300['ts']

df_300 = df_300.dropna(subset = ['open'])

df_300.set_index('ts', inplace=True)
df_300['dateonly'] = pd.to_datetime((df_300.index- timedelta(hours=15)).date)
df_300.loc[(df_300.date - timedelta(hours=13)).dt.weekday ==6,'dateonly'] = pd.to_datetime((df_300[(df_300.date - timedelta(hours=13)).dt.weekday ==6].date- timedelta(hours=63)).dt.date)
df_300.loc[df_300.dateonly ==datetime(2023, 9, 29, 0, 0),'dateonly'] = datetime(2023, 9, 28, 0, 0)
df_300.loc[df_300.dateonly ==datetime(2023, 10, 10, 0, 0),'dateonly'] = datetime(2023, 10, 6, 0, 0)
df_300.loc[df_300.dateonly ==datetime(2024, 1, 1, 0, 0),'dateonly'] = datetime(2023, 12, 29, 0, 0)
df_300.loc[df_300.dateonly ==datetime(2024, 2, 14, 0, 0),'dateonly'] = datetime(2024, 2, 5, 0, 0)
df_300 = pd.merge(df_300, cost_df, left_on="dateonly", right_on="日期",how='left')
df_300 = pd.merge(df_300, inves_limit, on="日期",how='left')
df_300 = pd.merge(df_300, dealer_limit, on="日期",how='left')
df_300.loc[df_300.dateonly ==datetime(2024, 1, 1, 0, 0),'dateonly'] = datetime(2023, 12, 29, 0, 0)
df_300.loc[df_300.dateonly ==datetime(2024, 2, 14, 0, 0),'dateonly'] = datetime(2024, 2, 5, 0, 0)
df_300.loc[df_300.dateonly ==datetime(2024, 2, 28, 0, 0),'dateonly'] = datetime(2024, 2, 27, 0, 0)
df_300.loc[df_300.dateonly ==datetime(2024, 4, 5, 0, 0),'dateonly'] = datetime(2024, 4, 3, 0, 0)
df_300.loc[df_300.dateonly ==datetime(2024, 5, 1, 0, 0),'dateonly'] = datetime(2024, 4, 30, 0, 0)
#df_300.loc[Final60Tdata["外資成本"]==None,['外資成本','外資上極限','外資下極限','自營商上極限','自營商下極限']] = [16347,16673,16227,16645,16155]
df_300.index = df_300.date

df_300 = df_300.sort_index()

# 計算布林帶指標
df_300['20MA'] = df_300['close'].rolling(20).mean()
#df_300['60MA'] = df_300['close'].rolling(60).mean()
#df_300['200MA'] = df_300['close'].rolling(200).mean()
df_300['std'] = df_300['close'].rolling(20).std()
df_300['upper_band'] = df_300['20MA'] + 2 * df_300['std']
df_300['lower_band'] = df_300['20MA'] - 2 * df_300['std']
df_300['upper_band1'] = df_300['20MA'] + 1 * df_300['std']
df_300['lower_band1'] = df_300['20MA'] - 1 * df_300['std']






# 在k线基础上计算KDF，并将结果存储在df上面(k,d,j)
low_list = df_300['min'].rolling(9, min_periods=9).min()
low_list.fillna(value=df_300['min'].expanding().min(), inplace=True)
high_list = df_300['max'].rolling(9, min_periods=9).max()
high_list.fillna(value=df_300['max'].expanding().max(), inplace=True)
rsv = (df_300['close'] - low_list) / (high_list - low_list) * 100
df_300['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
df_300['D'] = df_300['K'].ewm(com=2).mean()

enddatemonth = enddate[~enddate["契約月份"].str.contains("W")]['最後結算日']
df_300['end_low'] = 0
df_300['end_high'] = 0

#詢問
ds = 2
df_300['uline'] = df_300['max'].rolling(ds, min_periods=1).max()
df_300['dline'] = df_300['min'].rolling(ds, min_periods=1).min()

df_300["all_kk"] = 0
barssince5 = 0
barssince6 = 0
df_300['labelb'] = 1
df_300 = df_300[~df_300.index.duplicated(keep='first')]

#df_300 = df_300.dropna(subset = ['open','close','max','min'])

df_300['IC'] = df_300['close'] + 2 * df_300['close'].shift(1) - df_300['close'].shift(3) -df_300['close'].shift(4)
#df_300 = df_300[df_300.index>df_300.index[-80]]

for i in range(2,len(df_300.index)):
    try:
        #(df_300.loc[df_300.index[i],'close'] > df_300.loc[df_300.index[i-1],"uline"])
        condition51 = (df_300.loc[df_300.index[i-1],"max"] < df_300.loc[df_300.index[i-2],"min"] ) and (df_300.loc[df_300.index[i],"min"] > df_300.loc[df_300.index[i-1],"max"] )
        #condition52 = (df_300.loc[df_300.index[i-1],'close'] < df_300.loc[df_300.index[i-2],"min"]) and (df_300.loc[df_300.index[i-1],'成交金額'] > df_300.loc[df_300.index[i-2],'成交金額']) and (df_300.loc[df_300.index[i],'close']>df_300.loc[df_300.index[i-1],"max"] )
        condition53 = (df_300.loc[df_300.index[i],'close'] > df_300.loc[df_300.index[i-1],"uline"]) and (df_300.loc[df_300.index[i-1],'close'] <= df_300.loc[df_300.index[i-1],"uline"])

        condition61 = (df_300.loc[df_300.index[i-1],"min"] > df_300.loc[df_300.index[i-2],"max"] ) and (df_300.loc[df_300.index[i],"max"] < df_300.loc[df_300.index[i-1],"min"] )
        #condition62 = (df_300.loc[df_300.index[i-1],'close'] > df_300.loc[df_300.index[i-2],"max"]) and (df_300.loc[df_300.index[i-1],'成交金額'] > df_300.loc[df_300.index[i-2],'成交金額']) and (df_300.loc[df_300.index[i],'close']<df_300.loc[df_300.index[i-1],"min"] )
        condition63 = (df_300.loc[df_300.index[i],'close'] < df_300.loc[df_300.index[i-1],"dline"]) and (df_300.loc[df_300.index[i-1],'close'] >= df_300.loc[df_300.index[i-1],"dline"])
    except:
        condition51 = True
        condition52 = True
        condition53 = True
        condition61 = True
        condition63 = True
    condition54 = condition51 or condition53 #or condition52
    condition64 = condition61 or condition63 #or condition62 

    #df_300['labelb'] = np.where((df_300['close']> df_300['upper_band1']) , 1, np.where((df_300['close']< df_300['lower_band1']),-1,1))

    #print(i)
    if df_300.loc[df_300.index[i],'close'] > df_300.loc[df_300.index[i],'upper_band1']:
        df_300.loc[df_300.index[i],'labelb'] = 1
    elif df_300.loc[df_300.index[i],'close'] < df_300.loc[df_300.index[i],'lower_band1']:
        df_300.loc[df_300.index[i],'labelb'] = -1
    else:
        df_300.loc[df_300.index[i],'labelb'] = df_300.loc[df_300.index[i-1],'labelb']

    if condition54 == True:
        barssince5 = 1
    else:
        barssince5 += 1

    if condition64 == True:
        barssince6 = 1
    else:
        barssince6 += 1


    if barssince5 < barssince6:
        df_300.loc[df_300.index[i],"all_kk"] = 1
    else:
        df_300.loc[df_300.index[i],"all_kk"] = -1


df_300 = df_300[df_300.index>df_300.index[-80]]






IChour2 = []
finalhour = list(df_300['IC'].index)[-1]
plusi = 1
while (finalhour + timedelta(hours = plusi)).hour in [6,7,13,14] or  (finalhour + timedelta(hours = plusi)-timedelta(hours = 5)).weekday in [5,6]:
    plusi = plusi + 1
IChour2.append((finalhour + timedelta(hours = plusi)).strftime("%m-%d-%Y %H:%M"))
IChour2.append((finalhour + timedelta(hours = plusi+5)).strftime("%m-%d-%Y %H:%M"))

df_300.index = df_300.index.strftime(("%m-%d-%Y %H:%M"))
#df_300

checkb = df_300["labelb"].values[0]
bandstart = 1
bandidx = 1
checkidx = 0
while bandidx < len(df_300["labelb"].values):
    #checkidx = bandidx
    bandstart = bandidx-1
    checkidx = bandstart+1
    if checkidx >=len(df_300["labelb"].values)-1:
        break
    while df_300["labelb"].values[checkidx] == df_300["labelb"].values[checkidx+1]:
        checkidx +=1
        if checkidx >=len(df_300["labelb"].values)-1:
            break
    bandend = checkidx+1
    #print(bandstart,bandend)
    if df_300["labelb"].values[bandstart+1] == 1:
        fig3_1.add_traces(go.Scatter(x=df_300.index[bandstart:bandend], y = df_300['lower_band'].values[bandstart:bandend],
                                    line = dict(color='rgba(0,0,0,0)'),showlegend=False),rows=[2], cols=[1], secondary_ys= [True])
            
        fig3_1.add_traces(go.Scatter(x=df_300.index[bandstart:bandend], y = df_300['upper_band'].values[bandstart:bandend],
                                    line = dict(color='rgba(0,0,0,0)'),
                                    fill='tonexty', 
                                    fillcolor = 'rgba(256,256,0,0.2)',showlegend=False
                                    ),rows=[2], cols=[1], secondary_ys= [True])
    else:


        fig3_1.add_traces(go.Scatter(x=df_300.index[bandstart:bandend], y = df_300['lower_band'].values[bandstart:bandend],
                                    line = dict(color='rgba(0,0,0,0)'),showlegend=False),rows=[2], cols=[1], secondary_ys= [True])
            
        fig3_1.add_traces(go.Scatter(x=df_300.index[bandstart:bandend], y = df_300['upper_band'].values[bandstart:bandend],
                                    line = dict(color='rgba(0,0,0,0)'),
                                    fill='tonexty', 
                                    fillcolor = 'rgba(137, 207, 240,0.2)',showlegend=False
                                    ),rows=[2], cols=[1], secondary_ys= [True])
    bandidx =checkidx +1
    if bandidx >=len(df_300["labelb"].values):
        break
### 成交量圖製作 ###
volume_colors1 = [red_color if df_300['close'][i] > df_300['close'][i-1] else green_color for i in range(len(df_300['close']))]
volume_colors1[0] = green_color

#fig.add_trace(go.Bar(x=df_300.index, y=df_300['成交金額'], name='成交量', marker=dict(color=volume_colors),showlegend=False), row=optvrank[0], col=1)
fig3_1.add_trace(go.Bar(x=df_300.index, y=df_300['Volume'], name='成交量', marker=dict(color=volume_colors1)), row=2, col=1)

#df_300.index = df_300.index - timedelta(hours = 6)

fig3_1.add_trace(go.Scatter(x=df_300.index,
                        y=df_300['外資成本'],
                        mode='lines',
                        #line=dict(color='green'),
                        name='外資成本'),row=2, col=1, secondary_y= True)
#fig3_1.add_trace(go.Scatter(x=df_300.index,
#                        y=df_300['外資上極限'],
#                        mode='lines',
#                        #line=dict(color='green'),
#                        name='外資上極限'),row=2, col=1)
#fig3_1.add_trace(go.Scatter(x=df_300.index,
#                        y=df_300['外資下極限'],
#                        mode='lines',
#                        #line=dict(color='green'),
#                        name='外資下極限'),row=2, col=1)

fig3_1.add_traces(go.Scatter(x=df_300.index, y = df_300['外資上極限'].values,
                                    line = dict(color='rgba(0,0,0,0)'),showlegend=False),rows=[2], cols=[1], secondary_ys= [True])
            
fig3_1.add_traces(go.Scatter(x=df_300.index, y = df_300['自營商上極限'].values,
                            line = dict(color='rgba(0,0,0,0)'),
                            fill='tonexty', 
                            fillcolor = 'rgba(0,0,256,0.2)',showlegend=False
                            ),rows=[2], cols=[1], secondary_ys= [True])
fig3_1.add_traces(go.Scatter(x=df_300.index, y = df_300['外資下極限'].values,
                                    line = dict(color='rgba(0,0,0,0)'),showlegend=False),rows=[2], cols=[1], secondary_ys= [True])
            
fig3_1.add_traces(go.Scatter(x=df_300.index, y = df_300['自營商下極限'].values,
                            line = dict(color='rgba(0,0,0,0)'),
                            fill='tonexty', 
                            fillcolor = 'rgba(256,0,0,0.2)',showlegend=False
                            ),rows=[2], cols=[1], secondary_ys= [True])



fig3_1.add_trace(go.Scatter(x=df_300.index,
                        y=df_300['20MA'],
                        mode='lines',
                        line=dict(color='green'),
                        name='MA20'),row=2, col=1, secondary_y= True)
#fig3_1.add_trace(go.Scatter(x=df_300.index,
#                        y=df_300['200MA'],
#                        mode='lines',
#                        line=dict(color='blue'),
#                        name='MA200'),row=1, col=1)
#fig3_1.add_trace(go.Scatter(x=df_300.index,
#                        y=df_300['60MA'],
#                        mode='lines',
#                        line=dict(color='orange'),
#                        name='MA60'),row=1, col=1)

#fig3_1.add_trace(go.Scatter(x=list(df_300['IC'].index)[2:]+IChour2,
#                        y=df_300['IC'].values,
#                        mode='lines',
#                        line=dict(color='orange'),
#                        name='IC操盤線'),row=2, col=1, secondary_y= True)



### K線圖製作 ###
fig3_1.add_trace(
    go.Candlestick(
        x=df_300[(df_300['all_kk'] == -1)&(df_300['close'] >df_300['open'] )].index,
        open=df_300[(df_300['all_kk'] == -1)&(df_300['close'] >df_300['open'] )]['open'],
        high=df_300[(df_300['all_kk'] == -1)&(df_300['close'] >df_300['open'] )]['max'],
        low=df_300[(df_300['all_kk'] == -1)&(df_300['close'] >df_300['open'] )]['min'],
        close=df_300[(df_300['all_kk'] == -1)&(df_300['close'] >df_300['open'] )]['close'],
        increasing_line_color=decreasing_color,
        increasing_fillcolor=no_color, #fill_increasing_color(df_300.index>df_300.index[50])
        decreasing_line_color=decreasing_color,
        decreasing_fillcolor=no_color,#decreasing_color,
        line=dict(width=2),
        name='OHLC',showlegend=False
    )#,

    ,row=2, col=1, secondary_y= True
)


fig3_1.add_trace(
    go.Candlestick(
        x=df_300[(df_300['all_kk'] == 1)&(df_300['close'] >df_300['open'] )].index,
        open=df_300[(df_300['all_kk'] == 1)&(df_300['close'] >df_300['open'] )]['open'],
        high=df_300[(df_300['all_kk'] == 1)&(df_300['close'] >df_300['open'] )]['max'],
        low=df_300[(df_300['all_kk'] == 1)&(df_300['close'] >df_300['open'] )]['min'],
        close=df_300[(df_300['all_kk'] == 1)&(df_300['close'] >df_300['open'] )]['close'],
        increasing_line_color=increasing_color,
        increasing_fillcolor=no_color, #fill_increasing_color(df_300.index>df_300.index[50])
        decreasing_line_color=increasing_color,
        decreasing_fillcolor=no_color,#decreasing_color,
        line=dict(width=1),
        name='OHLC',showlegend=False
    )#,

    ,row=2, col=1, secondary_y= True
)

### K線圖製作 ###
fig3_1.add_trace(
    go.Candlestick(
        x=df_300[(df_300['all_kk'] == -1)&(df_300['close'] <df_300['open'] )].index,
        open=df_300[(df_300['all_kk'] == -1)&(df_300['close'] <df_300['open'] )]['open'],
        high=df_300[(df_300['all_kk'] == -1)&(df_300['close'] <df_300['open'] )]['max'],
        low=df_300[(df_300['all_kk'] == -1)&(df_300['close'] <df_300['open'] )]['min'],
        close=df_300[(df_300['all_kk'] == -1)&(df_300['close'] <df_300['open'] )]['close'],
        increasing_line_color=decreasing_color,
        increasing_fillcolor=decreasing_color, #fill_increasing_color(df_300.index>df_300.index[50])
        decreasing_line_color=decreasing_color,
        decreasing_fillcolor=decreasing_color,#decreasing_color,
        line=dict(width=1),
        name='OHLC',showlegend=False
    )#,

    ,row=2, col=1, secondary_y= True
)


fig3_1.add_trace(
    go.Candlestick(
        x=df_300[(df_300['all_kk'] == 1)&(df_300['close'] <df_300['open'] )].index,
        open=df_300[(df_300['all_kk'] == 1)&(df_300['close'] <df_300['open'] )]['open'],
        high=df_300[(df_300['all_kk'] == 1)&(df_300['close'] <df_300['open'] )]['max'],
        low=df_300[(df_300['all_kk'] == 1)&(df_300['close'] <df_300['open'] )]['min'],
        close=df_300[(df_300['all_kk'] == 1)&(df_300['close'] <df_300['open'] )]['close'],
        increasing_line_color=increasing_color,
        increasing_fillcolor=increasing_color, #fill_increasing_color(Final60Tdata.index>FinalＷeekdata.index[50])
        decreasing_line_color=increasing_color,
        decreasing_fillcolor=increasing_color,#decreasing_color,
        line=dict(width=1),
        name='OHLC',showlegend=False
    )#,

    ,row=2, col=1, secondary_y= True
)

#T60noshow = []
#curdatetime = Final60Tdata.index[0]
#while curdatetime < datetime.today():
#    if curdatetime not in Final60Tdata.index:
#        T60noshow.append(datetime.strftime(curdatetime,'%Y-%m-%d %H:%M:%S'))
#    curdatetime = curdatetime + timedelta(hours = 1)

#T60noshow




fig3_1.update_xaxes(
    rangeslider= {'visible':False},
    rangebreaks=[
        dict(bounds=[6, 8], pattern="hour"),
        #dict(bounds=[6,24],pattern = "sat"),
        dict(bounds=[13, 15], pattern="hour"),
        #dict(bounds=[6,24], pattern="hour"),#,bounds=['sat','sun']),# hide weekends, eg. hide sat to before mon
        dict(bounds=['sun','mon']),
        #dict(values=[str(holiday) for holiday in holidf[~(holidf["說明"].str.contains('開始交易') | holidf["說明"].str.contains('最後交易'))]["日期"].values]+['2023-08-03'])
        #dict(dvalues=T60noshow[10:], pattern="hour")
    ],
                row = 1, 
                col = 1
)


fig3_1.update_xaxes(
    rangeslider= {'visible':False},
    rangebreaks=[
        #dict(bounds=[6, 8], pattern="hour"),

        dict(bounds=['sat', 'mon']),# hide weekends, eg. hide sat to before mon
        #dict(values=T300noshow)
    ],
                row = 2, 
                col = 1
)

fig3_1.update_yaxes(
    range=[0, Final60Tdata['Volume'].max()+100],showgrid=False,
    secondary_y=False,
                row = 1, 
                col = 1
)
fig3_1.update_yaxes(
    range=[Final60Tdata['外資下極限'].min() - 30, Final60Tdata['自營商上極限'].max() + 30],showgrid=False,
    secondary_y=True,
                row = 1, 
                col = 1
)
fig3_1.update_yaxes(
    range=[df_300['min'].min() - 200, df_300['max'].max() + 200],showgrid=False,
        secondary_y=True,
                row = 2, 
                col = 1
)
fig3_1.update_yaxes(
    range=[0, df_300['Volume'].max()+100],showgrid=False,
    secondary_y=False,
                row = 2, 
                col = 1,showticklabels=False
)




# 設定圖的標題跟長寬
fig3_1.update_annotations(font_size=12)
fig3_1.update_layout(title_text = "", hovermode='x unified',
                yaxis = dict(showgrid=False,showticklabels=False),#,tickformat = ",.0f",range=[Final60Tdata['min'].min() - 50, Final60Tdata['max'].max() + 50]),
                yaxis2 = dict(showgrid=False),#showticklabels=False,range=[0, Final60Tdata['Volume'].max()+100]),
                #yaxis = dict(showgrid=False,showticklabels=False),

                width = 1200, 
                height = 1200,
                hoverlabel_namelength=-1,
                
                hoverlabel=dict(align='left',bgcolor='rgba(255,255,255,0.5)',font=dict(color='black')),
                legend_traceorder="reversed",
                )
#fig3_1.update_traces(xaxis='x1',hoverlabel=dict(align='left'))


st.plotly_chart(fig3_1)
