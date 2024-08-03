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
from os import getcwd

#import update
#from tkinter import messagebox

#def fxn():
#    warnings.warn("deprecated", DeprecationWarning)

#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#    fxn()


#if datetime.now().hour == 15 or datetime.now().hour == 19:
#    if datetime.now().weekday()<4:
#        update.run_all()
#    else:
#        pass
#else:
#    pass
##msgresult = messagebox.askyesno('askyesno', 'askyesno')

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
"start_date": "2023-03-02",
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

putcallgap_month = pd.read_sql("select 日期, max(價外買賣權價差) as 月價外買賣權價差 from putcallgap_month group by 日期", connection, parse_dates=['日期'], index_col=['日期'])


print(putcallsum_month.tail())
kbars = kbars.join(ordervolumn).join(putcallsum).join(putcallsum_month)
kbars = kbars.join(putcallgap).join(putcallgap_month)

# 計算布林帶指標
kbars['20MA'] = kbars['收盤指數'].rolling(20).mean()
kbars['std'] = kbars['收盤指數'].rolling(20).std()
kbars['60MA'] = kbars['收盤指數'].rolling(60).mean()
kbars['200MA'] = kbars['收盤指數'].rolling(200).mean()

kbars = kbars[kbars.index > kbars.index[-100]]

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
tab1, tab2 = st.tabs(["主圖", "支撐壓力"])

with tab1:

    #st.sidebar.write('結算日顯示')
    #option_month = st.sidebar.checkbox('月結算日', value = True)
    #option_week = st.sidebar.checkbox('週結算日', value = False)


    #st.sidebar.write('附圖選擇')


    #option_2c = st.sidebar.checkbox('開盤賣張張數', value = True)
    #option_2d = st.sidebar.checkbox('價平和', value = True)
    #option_2e = st.sidebar.checkbox('月價平和日差', value = True)
    #option_2f = st.sidebar.checkbox('月結趨勢', value = True)
    #kbars
    options_vice = [ True ]*4

    optvn = 0
    optvrank = []
    for opv in options_vice:
        if opv == True:
            optvn += 1
            optvrank.append(optvn+1)
        else:
            optvrank.append(0)
    subtitle_all = ['OHLC',   '開盤賣張','價平和','月價平和日差','月結趨勢']
    subtitle =['OHLC']
    for i in range(1,5):
        if optvrank[i-1] != 0:
            subtitle.append(subtitle_all[i])    



    #subtitle
    enddate = pd.read_sql("select * from end_date", connection, parse_dates=['最後結算日'])


    st.title('選擇權')
    rowcount = optvn + 1 + 8 + 2
    rowh = [0.2] + [ 0.6/(rowcount - 3)] * (rowcount - 3)+[0.1,0.1]
    fig = make_subplots(
        rows=rowcount, cols=1,
        shared_xaxes=True, 
        vertical_spacing=0.02,
        row_heights= rowh[:rowcount],
        shared_yaxes=False,
        #subplot_titles=subtitle,
        #y_title = "test"# subtitle,
        specs = [[{"secondary_y":True}]]*rowcount
    )

    increasing_color = 'rgb(255, 0, 0)'
    decreasing_color = 'rgb(0, 0, 245)'

    red_color = 'rgba(255, 0, 0, 0.1)'
    green_color = 'rgba(30, 144, 255,0.1)'

    no_color = 'rgba(256, 256, 256,0)'

    blue_color = 'rgb(30, 144, 255)'
    red_color_full = 'rgb(255, 0, 0)'

    orange_color = 'rgb(245, 152, 59)'
    green_color_full = 'rgb(52, 186, 7)'

    gray_color = 'rgb(188, 194, 192)'
    black_color = 'rgb(0, 0, 0)'


    ### 成本價及上下極限 ###
    #fig.add_trace(go.Scatter(x=list(kbars['IC'].index)[1:]+[ICdate[0]],
    #                y=kbars['外資成本'].shift(1).values,
    #                mode='lines',
    #                line=dict(color='yellow'),
    #                name='外資成本'),row=1, col=1, secondary_y= True)


    #自營商外資上極限
    #fig.add_scatter(x=np.concatenate([kbars.index,kbars.index[::-1]]), y=np.concatenate([kbars['外資上極限'], kbars['自營商上極限'][::-1]]), 
    #                fill='toself',fillcolor= 'rgba(0,0,256,0.1)', line_width=0,name='上極限',row=1, col=1 )



    #自營商外資下極限
    #fig.add_scatter(x=np.concatenate([kbars.index,kbars.index[::-1]]), y=np.concatenate([kbars['外資下極限'], kbars['自營商下極限'][::-1]]), 
    #                fill='toself',fillcolor= 'rgba(256,0,0,0.1)', line_width=0,name='下極限',row=1, col=1)

    #上下極限
    #kbars[kbars['收盤指數']> kbars['upper_band1']]

    #np.concatenate([kbars[kbars['收盤指數'] > kbars['upper_band1']].index,kbars[kbars['收盤指數']> kbars['upper_band1']].index[::-1]])
    #buling_colors = ['rgba(0,256,0,0.1)' if kbars['收盤指數'][i] > kbars['upper_band1'][i] else 'rgba(0,256,256,0.1)' for i in range(len(kbars['lower_band']))]
    #fillcol(kbars['labelb'].iloc[0])
    #fig.add_scatter(x=np.concatenate([kbars.index,kbars.index[::-1]]), y=np.concatenate([kbars['lower_band'], kbars['upper_band'][::-1]]), 
    #               fill='toself',fillcolor= kbars['labelb'].iloc[0], line_width=0,name='布林上下極限',row=1, col=1)


    checkb = kbars["labelb"].values[0]
    bandstart = 1
    bandidx = 1
    checkidx = 0
    while bandidx < len(kbars["labelb"].values):
        #checkidx = bandidx
        bandstart = bandidx-1
        checkidx = bandstart+1
        if checkidx >=len(kbars["labelb"].values)-1:
            break
        while kbars["labelb"].values[checkidx] == kbars["labelb"].values[checkidx+1]:
            checkidx +=1
            if checkidx >=len(kbars["labelb"].values)-1:
                break
        bandend = checkidx+1
        #print(bandstart,bandend)
        if kbars["labelb"].values[bandstart+1] == 1:
            fig.add_traces(go.Scatter(x=kbars.index[bandstart:bandend], y = kbars['lower_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),showlegend=False,hoverinfo='none'),secondary_ys= [True,True])
                
            fig.add_traces(go.Scatter(x=kbars.index[bandstart:bandend], y = kbars['upper_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),
                                        fill='tonexty', 
                                        fillcolor = 'rgba(256,256,0,0.2)',showlegend=False,hoverinfo='none'
                                        ),secondary_ys= [True,True])
        else:


            fig.add_traces(go.Scatter(x=kbars.index[bandstart:bandend], y = kbars['lower_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),showlegend=False,hoverinfo='none'), secondary_ys= [True,True])
                
            fig.add_traces(go.Scatter(x=kbars.index[bandstart:bandend], y = kbars['upper_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),
                                        fill='tonexty', 
                                        fillcolor = 'rgba(137, 207, 240,0.2)',showlegend=False,hoverinfo='none'
                                        ),secondary_ys= [True,True])
        bandidx =checkidx +1
        if bandidx >=len(kbars["labelb"].values):
            break

    #fig.add_scatter(x=np.concatenate([kbars[kbars['收盤指數'] <= kbars['upper_band1']].index,kbars[kbars['收盤指數']<= kbars['upper_band1']].index[::-1]]), y=np.concatenate([kbars[kbars['收盤指數']<= kbars['upper_band1']]['lower_band'], kbars[kbars['收盤指數'] <= kbars['upper_band1']]['upper_band'][::-1]]), 
    #                fill='toself',fillcolor= 'rgba(0,256,256,0.1)', line_width=0,name='布林上下極限',row=1, col=1)

    #fig.add_trace(go.Scatter(x=kbars.index,
    #                 y=kbars['外資上極限'],
    #                 mode='lines',
    #                 line=dict(color='#9467bd'),
    #                 name='外資上極限'))

    #fig.add_trace(go.Scatter(x=kbars.index,
    #                 y=kbars['外資下極限'],
    #                 mode='lines',
    #                 line=dict(color='#17becf'),
    #                 name='外資下極限'))

    ### 成交量圖製作 ###
    volume_colors = [red_color if kbars['收盤指數'][i] > kbars['收盤指數'][i-1] else green_color for i in range(len(kbars['收盤指數']))]
    volume_colors[0] = green_color

    #fig.add_trace(go.Bar(x=kbars.index, y=kbars['成交金額'], name='Volume', marker=dict(color=volume_colors),showlegend=False), row=optvrank[0], col=1)
    fig.add_trace(go.Bar(x=kbars.index, y=kbars['成交金額'], name='成交金額', marker=dict(color=volume_colors)), row=1, col=1)


    fig.add_trace(go.Scatter(x=kbars.index,
                            y=kbars['20MA'],
                            mode='lines',
                            line=dict(color='green'),
                            name='20MA'),row=1, col=1, secondary_y= True)

    fig.add_trace(go.Scatter(x=list(kbars['IC'].index)[2:]+ICdate,
                            y=kbars['IC'].values,
                            mode='lines',
                            line=dict(color='orange'),
                            name='IC操盤線'),row=1, col=1, secondary_y= True)
    
    fig.add_trace(go.Scatter(x=kbars.index,
                            y=kbars['200MA'],
                            mode='lines',
                            line=dict(color='blue'),
                            name='MA200'),row=1, col=1, secondary_y= True)
    fig.add_trace(go.Scatter(x=kbars.index,
                            y=kbars['60MA'],
                            mode='lines',
                            line=dict(color='orange'),
                            name='MA60'),row=1, col=1, secondary_y= True)

   
         
    # fig.add_trace(go.Scatter(x=[kbars.index[0],kbars.index[0]],y=[15500,17500], line_width=0.1, line_color="green",name='月結算日',showlegend=False),row=1, col=1)
    # #if option_month == True:
    # for i in enddate[~enddate["契約月份"].str.contains("W")]['最後結算日']:
    #     if i > kbars.index[0] :#and i!=enddate[~enddate["契約月份"].str.contains("W")]['最後結算日'].values[6]:
    #         fig.add_vline(x=i, line_width=1, line_color="green",name='月結算日',row=1, col=1)

    # #enddate['最後結算日'].values
    # #enddate.groupby(enddate['最後結算日'].dt.month)['最後結算日'].max()
    # #list(enddate['最後結算日'].values)[:3]
    # #if option_week == True:
    # for i in enddate['最後結算日']:
    #     if i > kbars.index[0] :# and i!=enddate.groupby(enddate['最後結算日'].dt.month)['最後結算日'].max()[6] and i not in enddate.groupby(enddate['最後結算日'].dt.month)['最後結算日'].max():
    #         fig.add_vline(x=i, line_width=1,line_dash="dash", line_color="blue",name='週結算日')#, line_dash="dash"
    #     #fig.add_hrect(y0=0.9, y1=2.6, line_width=0, fillcolor="red", opacity=0.2)
    


    ### K線圖製作 ###
    fig.add_trace(
        go.Candlestick(
            x=kbars[(kbars['all_kk'] == -1)&(kbars['收盤指數'] >kbars['開盤指數'] )].index,
            open=kbars[(kbars['all_kk'] == -1)&(kbars['收盤指數'] >kbars['開盤指數'] )]['開盤指數'],
            high=kbars[(kbars['all_kk'] == -1)&(kbars['收盤指數'] >kbars['開盤指數'] )]['最高指數'],
            low=kbars[(kbars['all_kk'] == -1)&(kbars['收盤指數'] >kbars['開盤指數'] )]['最低指數'],
            close=kbars[(kbars['all_kk'] == -1)&(kbars['收盤指數'] >kbars['開盤指數'] )]['收盤指數'],
            increasing_line_color=decreasing_color,
            increasing_fillcolor=no_color, #fill_increasing_color(kbars.index>kbars.index[50])
            decreasing_line_color=decreasing_color,
            decreasing_fillcolor=no_color,#decreasing_color,
            line=dict(width=2),
            name='OHLC',showlegend=False
        )#,
        
        ,row=1, col=1, secondary_y= True
    )


    fig.add_trace(
        go.Candlestick(
            x=kbars[(kbars['all_kk'] == 1)&(kbars['收盤指數'] >kbars['開盤指數'] )].index,
            open=kbars[(kbars['all_kk'] == 1)&(kbars['收盤指數'] >kbars['開盤指數'] )]['開盤指數'],
            high=kbars[(kbars['all_kk'] == 1)&(kbars['收盤指數'] >kbars['開盤指數'] )]['最高指數'],
            low=kbars[(kbars['all_kk'] == 1)&(kbars['收盤指數'] >kbars['開盤指數'] )]['最低指數'],
            close=kbars[(kbars['all_kk'] == 1)&(kbars['收盤指數'] >kbars['開盤指數'] )]['收盤指數'],
            increasing_line_color=increasing_color,
            increasing_fillcolor=no_color, #fill_increasing_color(kbars.index>kbars.index[50])
            decreasing_line_color=increasing_color,
            decreasing_fillcolor=no_color,#decreasing_color,
            line=dict(width=1),
            name='OHLC',showlegend=False
        )#,
        
        ,row=1, col=1, secondary_y= True
    )

    ### K線圖製作 ###
    fig.add_trace(
        go.Candlestick(
            x=kbars[(kbars['all_kk'] == -1)&(kbars['收盤指數'] <kbars['開盤指數'] )].index,
            open=kbars[(kbars['all_kk'] == -1)&(kbars['收盤指數'] <kbars['開盤指數'] )]['開盤指數'],
            high=kbars[(kbars['all_kk'] == -1)&(kbars['收盤指數'] <kbars['開盤指數'] )]['最高指數'],
            low=kbars[(kbars['all_kk'] == -1)&(kbars['收盤指數'] <kbars['開盤指數'] )]['最低指數'],
            close=kbars[(kbars['all_kk'] == -1)&(kbars['收盤指數'] <kbars['開盤指數'] )]['收盤指數'],
            increasing_line_color=decreasing_color,
            increasing_fillcolor=decreasing_color, #fill_increasing_color(kbars.index>kbars.index[50])
            decreasing_line_color=decreasing_color,
            decreasing_fillcolor=decreasing_color,#decreasing_color,
            line=dict(width=1),
            name='OHLC',showlegend=False
        )#,
        
        ,row=1, col=1, secondary_y= True
    )


    fig.add_trace(
        go.Candlestick(
            x=kbars[(kbars['all_kk'] == 1)&(kbars['收盤指數'] <kbars['開盤指數'] )].index,
            open=kbars[(kbars['all_kk'] == 1)&(kbars['收盤指數'] <kbars['開盤指數'] )]['開盤指數'],
            high=kbars[(kbars['all_kk'] == 1)&(kbars['收盤指數'] <kbars['開盤指數'] )]['最高指數'],
            low=kbars[(kbars['all_kk'] == 1)&(kbars['收盤指數'] <kbars['開盤指數'] )]['最低指數'],
            close=kbars[(kbars['all_kk'] == 1)&(kbars['收盤指數'] <kbars['開盤指數'] )]['收盤指數'],
            increasing_line_color=increasing_color,
            increasing_fillcolor=increasing_color, #fill_increasing_color(kbars.index>kbars.index[50])
            decreasing_line_color=increasing_color,
            decreasing_fillcolor=increasing_color,#decreasing_color,
            line=dict(width=1),
            name='OHLC',showlegend=False
        )#,
        
        ,row=1, col=1, secondary_y= True
    )



    
    ### KD線 ###
    #if optvrank[0] != 0:
    #    fig.add_trace(go.Scatter(x=kbars.index, y=kbars['K'], name='K', line=dict(width=1, color='rgb(41, 98, 255)'),showlegend=False), row=optvrank[0], col=1)
    #    fig.add_trace(go.Scatter(x=kbars.index, y=kbars['D'], name='D', line=dict(width=1, color='rgb(255, 109, 0)'),showlegend=False), row=optvrank[0], col=1)

    ## 委賣數量 ##
    if optvrank[0] != 0:
        days20 = kbars[(kbars.index> (kbars.index[-1] + timedelta(days = -20)))]
        max_days20 = days20["九點累積委託賣出數量"].values.max()
        
        min_days20 = days20["九點累積委託賣出數量"].values.min()
        #volume_colors = [increasing_color if kbars['九點累積委託賣出數量	'][i] > kbars['收盤指數'][i-1] else decreasing_color for i in range(len(kbars['收盤指數']))]
        fig.add_trace(go.Scatter(x=kbars.index, y=kbars['九點累積委託賣出數量'], name='成交數量',showlegend=False), row=optvrank[0], col=1)
        fig.add_scatter(x=np.array(max_days20_x), y=np.array(max_days20_list),marker=dict(color = blue_color,size=5),showlegend=False,mode = 'markers', row=optvrank[0], col=1)
        fig.add_scatter(x=np.array(min_days20_x), y=np.array(min_days20_list),marker=dict(color = orange_color,size=5),showlegend=False,mode = 'markers', row=optvrank[0], col=1)
        fig.update_yaxes(title_text="開盤賣張", row=optvrank[0], col=1)
    
    charti = 3
    ## 價平和

    PCsum_colors = [increasing_color if kbars['價平和'][i] > kbars['價平和'][i-1] else decreasing_color for i in range(len(kbars['價平和']))]
    PCsum_colors[0] = decreasing_color
    fig.add_trace(go.Bar(x=kbars.index, y=kbars['價平和'], name='PCsum', marker=dict(color=PCsum_colors),showlegend=False), row=charti, col=1)
    #fig.add_hline(y = 50, line_width=0.2,line_dash="dash", line_color="blue", row=charti, col=1)
    for i in range(1,int(max(kbars['價平和'].values)//50)+1):
        fig.add_trace(go.Scatter(x=kbars.index,y=[i*50]*len(kbars.index),showlegend=False,hoverinfo='none', line_width=0.5,line_dash="dash", line_color="black"), row=charti, col=1)
    fig.update_yaxes(title_text="價平和", row=charti, col=1)

    charti = charti +1
    ## 價外買賣權價差

    fig.add_trace(go.Bar(x=kbars[(kbars['價外買賣權價差']>0)].index, y=(kbars[(kbars['價外買賣權價差']>0)]['價外買賣權價差']), name='價外買賣權價差',marker=dict(color = red_color_full),showlegend=False), row=charti, col=1)
    fig.add_trace(go.Bar(x=kbars[(kbars['價外買賣權價差']<=0)].index, y=(kbars[(kbars['價外買賣權價差']<=0)]['價外買賣權價差']), name='價外買賣權價差',marker=dict(color = blue_color),showlegend=False), row=charti, col=1)
    #fig.add_hline(y = 50, line_width=0.2,line_dash="dash", line_color="blue", row=charti, col=1)
    fig.update_yaxes(title_text="價外買賣權價差", row=charti, col=1)
        

    charti = charti +1
    

    fig.add_trace(go.Bar(x=kbars[(kbars['月價平和日差']>0)&(~kbars.index.isin(notshowdate))].index, y=(kbars[(kbars['月價平和日差']>0)&(~kbars.index.isin(notshowdate))]['月價平和日差']), name='月價平和日差',marker=dict(color = red_color_full),showlegend=False), row=charti, col=1)
    fig.add_trace(go.Bar(x=kbars[(kbars['月價平和日差']<=0)&(~kbars.index.isin(notshowdate))].index, y=(kbars[(kbars['月價平和日差']<=0)&(~kbars.index.isin(notshowdate))]['月價平和日差']), name='月價平和日差',marker=dict(color = blue_color),showlegend=False), row=charti, col=1)
    fig.update_yaxes(title_text="月價平和日差", row=charti, col=1)

    charti = charti +1
    ## 月價外買賣權價差

    fig.add_trace(go.Bar(x=kbars[(kbars['月價外買賣權價差']>0)].index, y=(kbars[(kbars['月價外買賣權價差']>0)]['月價外買賣權價差']), name='月價外買賣權價差',marker=dict(color = red_color_full),showlegend=False), row=charti, col=1)
    fig.add_trace(go.Bar(x=kbars[(kbars['月價外買賣權價差']<=0)].index, y=(kbars[(kbars['月價外買賣權價差']<=0)]['月價外買賣權價差']), name='月價外買賣權價差',marker=dict(color = blue_color),showlegend=False), row=charti, col=1)
    #fig.add_hline(y = 50, line_width=0.2,line_dash="dash", line_color="blue", row=charti, col=1)
    fig.update_yaxes(title_text="月價外買賣權價差", row=charti, col=1)
    
    
    charti = charti +1
    ## 月結趨勢

    fig.add_trace(go.Bar(x=kbars.index, y=kbars['end_high'], name='MAX_END',marker=dict(color = black_color),showlegend=False), row=charti, col=1)
    fig.add_trace(go.Bar(x=kbars.index, y=kbars['end_low'], name='MIN_END',marker=dict(color = gray_color),showlegend=False), row=charti, col=1)
    fig.update_yaxes(title_text="月結趨勢", row=charti, col=1, tickfont=dict(size=8))

    
    ##外資買賣超
    fig.add_trace(go.Bar(x=dfbuysell[dfbuysell['ForeBuySell']>0].index, y=(dfbuysell[dfbuysell['ForeBuySell']>0]["ForeBuySell"]).round(2), name='外資買賣超',marker=dict(color = red_color_full),showlegend=False), row=charti+1, col=1)
    fig.add_trace(go.Bar(x=dfbuysell[dfbuysell['ForeBuySell']<=0].index, y=(dfbuysell[dfbuysell['ForeBuySell']<=0]["ForeBuySell"]).round(2), name='外資買賣超',marker=dict(color = blue_color),showlegend=False), row=charti+1, col=1)
    #fig.add_trace(go.Bar(x=bank8.index, y=bank8["八大行庫買賣超金額"]/10000, name='eightbank',showlegend=False), row=charti+2, col=1)
    fig.update_yaxes(title_text="外資買賣超(億元)", row=charti+1, col=1)


    
    ## 外資臺股期貨未平倉淨口數
    #fut_colors = [red_color_full if kbars['收盤指數'][i] > kbars['收盤指數'][i-1] else blue_color for i in range(len(kbars['收盤指數']))]
    #fut_colors[0] = blue_color
    fut_colors = [increasing_color if futdf['多空未平倉口數淨額'][i] > futdf['多空未平倉口數淨額'][i-1] else decreasing_color for i in range(len(futdf['多空未平倉口數淨額']))]
    fut_colors[0] = decreasing_color
    #fig.add_trace(go.Bar(x=kbars.index, y=kbars['成交金額'], name='成交金額', marker=dict(color=volume_colors)), row=1, col=1, secondary_y= True)
    fig.add_trace(go.Bar(x=futdf.index, y=futdf['多空未平倉口數淨額'], name='fut', marker=dict(color=fut_colors),showlegend=False), row=charti+2, col=1)
    #fig.add_trace(go.Bar(x=bank8.index, y=bank8["八大行庫買賣超金額"]/10000, name='eightbank',showlegend=False), row=charti+2, col=1)
    fig.update_yaxes(title_text="外資未平倉淨口數", row=charti+2, col=1)

    
    

    #put call ratio
    #fig.add_trace(go.Scatter(x=kbars.index,y=kbars['收盤指數'],
    #                mode='lines',
    #                line=dict(color='black'),
    #                name='收盤指數',showlegend=False),row=charti+1, col=1)
    #fig.add_trace(go.Bar(x=CPratio.index, y=CPratio['買賣權未平倉量比率%']-100, name='PC_Ratio',showlegend=False), row=charti+4, col=1)
    #fig.update_yaxes(title_text="PutCallRatio", row=charti+4, col=1)
    

    #選擇權外資OI
    fig.add_trace(go.Bar(x=TXOOIdf.index, y=(TXOOIdf["買買賣賣"]), name='買買權+賣賣權',marker=dict(color = red_color_full),showlegend=False), row=charti+3, col=1)
    fig.add_trace(go.Bar(x=TXOOIdf.index, y=(TXOOIdf["買賣賣買"]), name='買賣權+賣買權',marker=dict(color = blue_color),showlegend=False), row=charti+3, col=1)
    #fig.add_trace(go.Bar(x=bank8.index, y=bank8["八大行庫買賣超金額"]/10000, name='eightbank',showlegend=False), row=charti+2, col=1)3
    fig.update_yaxes(title_text="選擇權外資OI", row=charti+3, col=1)

    #心態
    fin = []
    find = []
    for idx in range(1,len(dfbuysell.index)) :
        try:
            datei = dfbuysell.index[idx]
            one = dfbuysell.loc[datei,'ForeBuySell']

            
            two = (int(futdf.loc[datei,'多空未平倉口數淨額']) - int(futdf.loc[dfbuysell.index[idx-1],'多空未平倉口數淨額']))*kbars.loc[datei,'收盤指數']*200
            three = TXOOIdf.loc[datei,'買買賣賣'] - TXOOIdf.loc[datei,'買賣賣買']
            find.append(datei)
            fin.append(one+two/100000000+three/100000000)
            #print(datei,one,two/100000000,three/100000000)
        except:
            continue
    fin = np.array(fin)
    find = np.array(find)
    fig.add_trace(go.Bar(x=find[fin>0], y=fin[fin>0], name='外資期現選心態',marker=dict(color = red_color_full),showlegend=False), row=charti+4, col=1)
    fig.add_trace(go.Bar(x=find[fin<=0], y=fin[fin<=0], name='外資期現選心態',marker=dict(color = blue_color),showlegend=False), row=charti+4, col=1)
    fig.update_yaxes(title_text="外資期現選心態", row=charti+4, col=1)


    ## 小台散戶多空比
    
    fig.add_trace(go.Bar(x=dfMTX[dfMTX['MTXRatio']>0].index, y=(dfMTX[dfMTX['MTXRatio']>0]['MTXRatio']*100).round(2), name='小台散戶多空比',marker=dict(color = orange_color),showlegend=False), row=charti+8, col=1)
    fig.add_trace(go.Bar(x=dfMTX[dfMTX['MTXRatio']<=0].index, y=(dfMTX[dfMTX['MTXRatio']<=0]['MTXRatio']*100).round(2), name='小台散戶多空比',marker=dict(color = green_color_full),showlegend=False), row=charti+8, col=1)
    #fig.add_trace(go.Bar(x=bank8.index, y=bank8["八大行庫買賣超金額"]/10000, name='eightbank',showlegend=False), row=charti+2, col=1)
    fig.update_yaxes(title_text="小台散戶多空比", row=charti+8, col=1)

    

    #八大行庫買賣超
    fig.add_trace(go.Bar(x=bank8[bank8["八大行庫買賣超金額"]>0].index, y=(bank8[bank8["八大行庫買賣超金額"]>0]["八大行庫買賣超金額"]/100000).round(2), name='八大行庫買賣超',marker=dict(color = orange_color),showlegend=False), row=charti+6, col=1)
    fig.add_trace(go.Bar(x=bank8[bank8["八大行庫買賣超金額"]<=0].index, y=(bank8[bank8["八大行庫買賣超金額"]<=0]["八大行庫買賣超金額"]/100000).round(2), name='八大行庫買賣超',marker=dict(color = green_color_full),showlegend=False), row=charti+6, col=1)
    #fig.add_trace(go.Bar(x=bank8.index, y=bank8["八大行庫買賣超金額"]/10000, name='eightbank',showlegend=False), row=charti+2, col=1)
    fig.update_yaxes(title_text="八大行庫", row=charti+6, col=1)


    
    fig.add_trace(go.Scatter(x=dfMargin.index, y=dfMargin['MarginRate'],marker=dict(color = gray_color),line_width=3, name='MarginRate',showlegend=False), row=charti+7, col=1)
    fig.update_yaxes(title_text="大盤融資資維持率", row=charti+7, col=1)    



    #美元匯率
    url = "https://api.finmindtrade.com/api/v4/data?"
    parameter = {
    "dataset": "TaiwanExchangeRate",
    "data_id":'USD',
    "start_date": '2023-01-02',
    "end_date": datetime.strftime(datetime.today(),'%Y-%m-%d'),
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0wNy0zMCAyMzowMTo0MSIsInVzZXJfaWQiOiJqZXlhbmdqYXUiLCJpcCI6IjExNC4zNC4xMjEuMTA0In0.WDAZzKGv4Du5JilaAR7o7M1whpnGaR-vMDuSeTBXhhA", # 參考登入，獲取金鑰
    }
    data = requests.get(url, params=parameter)
    data = data.json()
    TaiwanExchangeRate = pd.DataFrame(data['data'])
    TaiwanExchangeRate.date = pd.to_datetime(TaiwanExchangeRate.date)
    TaiwanExchangeRate = TaiwanExchangeRate[~(TaiwanExchangeRate['spot_buy']==-1)]

    fig.add_trace(go.Scatter(x=TaiwanExchangeRate[(TaiwanExchangeRate.date>kbars.index[0])&(TaiwanExchangeRate.date!=datetime.strptime('2023-08-03', '%Y-%m-%d'))].date, y=TaiwanExchangeRate[(TaiwanExchangeRate.date>kbars.index[0])&(TaiwanExchangeRate.date!=datetime.strptime('2023-08-03', '%Y-%m-%d'))]['spot_buy'],marker=dict(color = gray_color), name='ExchangeRate',line_width=3,showlegend=False), row=charti+5, col=1)
    fig.update_yaxes(title_text="美元匯率", row=charti+5, col=1)  

    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0wNy0zMCAyMzowMTo0MSIsInVzZXJfaWQiOiJqZXlhbmdqYXUiLCJpcCI6IjExNC4zNC4xMjEuMTA0In0.WDAZzKGv4Du5JilaAR7o7M1whpnGaR-vMDuSeTBXhhA"
    url = "https://api.finmindtrade.com/api/v4/data?"

    

    

    

    


        
    ### 圖表設定 ###
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_annotations(font_size=12)

    fig.update_layout(
        title=u'大盤指數技術分析圖',
        #title_x=0.5,
        #title_y=0.93,
        hovermode='x unified',
        height=350 + 150* rowcount,
        width = 1200,
        hoverlabel_namelength=-1,
        hoverlabel_align = "left",
        xaxis2=dict(showgrid=False),
        yaxis2=dict(showgrid=False,tickformat = ",.0f",range=[kbars['最低指數'].min() - 200, kbars['最高指數'].max() + 200]),
        yaxis = dict(showgrid=False,showticklabels=False,range=[0, 90*10**10]),
        #yaxis = dict(range=[kbars['min'].min() - 2000, kbars['最高指數'].max() + 500]),
        dragmode = 'drawline',
        hoverlabel=dict(align='left',bgcolor='rgba(255,255,255,0.5)',font=dict(color='black')),
        legend_traceorder="reversed",
        
    )

    fig.update_traces(xaxis='x1',hoverlabel=dict(align='left'))

    # 隱藏周末與市場休市日期 ### 導入台灣的休市資料

    noshowdate = []
    for delta_day in range((datetime.now() - datetime.strptime(kbars.index.strftime('%Y-%m-%d')[0], '%Y-%m-%d')).days):
        if (datetime.now() - timedelta(days=delta_day)).strftime('%Y-%m-%d') not in kbars.index.strftime('%Y-%m-%d').values:
            noshowdate.append((datetime.now() - timedelta(days=delta_day)).strftime('%Y-%m-%d'))
        
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=['sat', 'mon']), # hide weekends, eg. hide sat to before mon
            dict(values= noshowdate)
        ]
    )


    #fig.update_traces(hoverlabel=dict(align='left'))

    st.plotly_chart(fig)
    

with tab2:

    
    WeekTAIEXdata.date = pd.to_datetime(WeekTAIEXdata.date)
    WeekTAIEXdata["yearww"] = WeekTAIEXdata.date.dt.year.astype(str) + 'W' + WeekTAIEXdata.date.dt.isocalendar().week.astype('str').str.zfill(2)
    FinalＷeekdata = WeekTAIEXdata.groupby('yearww').max()[["max"]].join(WeekTAIEXdata.groupby('yearww').min()[["min"]]).join(WeekTAIEXdata[['yearww',"Trading_Volume"]].groupby('yearww').sum()[["Trading_Volume"]]).join(WeekTAIEXdata[['yearww',"Trading_money"]].groupby('yearww').sum()[["Trading_money"]])
    WeekTAIEXdata.index = WeekTAIEXdata.date
    tempopen = WeekTAIEXdata.loc[WeekTAIEXdata.groupby('yearww').min()['date'].values]
    tempopen.index = tempopen.yearww
    tempclose = WeekTAIEXdata.loc[WeekTAIEXdata.groupby('yearww').max()['date'].values]
    tempclose.index = tempclose.yearww
    FinalＷeekdata = FinalＷeekdata.join(tempopen[["open"]]).join(tempclose[["close"]])
    # 計算布林帶指標
    FinalＷeekdata['20MA'] = FinalＷeekdata['close'].rolling(20).mean()
    FinalＷeekdata['60MA'] = FinalＷeekdata['close'].rolling(60).mean()
    FinalＷeekdata['200MA'] = FinalＷeekdata['close'].rolling(200).mean()
    FinalＷeekdata['std'] = FinalＷeekdata['close'].rolling(20).std()
    FinalＷeekdata['upper_band'] = FinalＷeekdata['20MA'] + 2 * FinalＷeekdata['std']
    FinalＷeekdata['lower_band'] = FinalＷeekdata['20MA'] - 2 * FinalＷeekdata['std']
    FinalＷeekdata['upper_band1'] = FinalＷeekdata['20MA'] + 1 * FinalＷeekdata['std']
    FinalＷeekdata['lower_band1'] = FinalＷeekdata['20MA'] - 1 * FinalＷeekdata['std']

    FinalＷeekdata['IC'] = FinalＷeekdata['close'] + 2 * FinalＷeekdata['close'].shift(1) - FinalＷeekdata['close'].shift(3) -FinalＷeekdata['close'].shift(4)

    # 在k线基础上计算KDF，并将结果存储在df上面(k,d,j)
    low_list = FinalＷeekdata['min'].rolling(9, min_periods=9).min()
    low_list.fillna(value=FinalＷeekdata['min'].expanding().min(), inplace=True)
    high_list = FinalＷeekdata['max'].rolling(9, min_periods=9).max()
    high_list.fillna(value=FinalＷeekdata['max'].expanding().max(), inplace=True)
    rsv = (FinalＷeekdata['close'] - low_list) / (high_list - low_list) * 100
    FinalＷeekdata['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
    FinalＷeekdata['D'] = FinalＷeekdata['K'].ewm(com=2).mean()

    enddatemonth = enddate[~enddate["契約月份"].str.contains("W")]['最後結算日']
    FinalＷeekdata['end_low'] = 0
    FinalＷeekdata['end_high'] = 0
    
    #詢問
    ds = 2
    FinalＷeekdata['uline'] = FinalＷeekdata['max'].rolling(ds, min_periods=1).max()
    FinalＷeekdata['dline'] = FinalＷeekdata['min'].rolling(ds, min_periods=1).min()

    FinalＷeekdata["all_kk"] = 0
    barssince5 = 0
    barssince6 = 0
    FinalＷeekdata['labelb'] = 1
    FinalＷeekdata = FinalＷeekdata[~FinalＷeekdata.index.duplicated(keep='first')]
    for i in range(2,len(FinalＷeekdata.index)):
        try:
            #(FinalＷeekdata.loc[FinalＷeekdata.index[i],'close'] > FinalＷeekdata.loc[FinalＷeekdata.index[i-1],"uline"])
            condition51 = (FinalＷeekdata.loc[FinalＷeekdata.index[i-1],"max"] < FinalＷeekdata.loc[FinalＷeekdata.index[i-2],"min"] ) and (FinalＷeekdata.loc[FinalＷeekdata.index[i],"min"] > FinalＷeekdata.loc[FinalＷeekdata.index[i-1],"max"] )
            #condition52 = (FinalＷeekdata.loc[FinalＷeekdata.index[i-1],'close'] < FinalＷeekdata.loc[FinalＷeekdata.index[i-2],"min"]) and (FinalＷeekdata.loc[FinalＷeekdata.index[i-1],'成交金額'] > FinalＷeekdata.loc[FinalＷeekdata.index[i-2],'成交金額']) and (FinalＷeekdata.loc[FinalＷeekdata.index[i],'close']>FinalＷeekdata.loc[FinalＷeekdata.index[i-1],"max"] )
            condition53 = (FinalＷeekdata.loc[FinalＷeekdata.index[i],'close'] > FinalＷeekdata.loc[FinalＷeekdata.index[i-1],"uline"]) and (FinalＷeekdata.loc[FinalＷeekdata.index[i-1],'close'] <= FinalＷeekdata.loc[FinalＷeekdata.index[i-1],"uline"])

            condition61 = (FinalＷeekdata.loc[FinalＷeekdata.index[i-1],"min"] > FinalＷeekdata.loc[FinalＷeekdata.index[i-2],"max"] ) and (FinalＷeekdata.loc[FinalＷeekdata.index[i],"max"] < FinalＷeekdata.loc[FinalＷeekdata.index[i-1],"min"] )
            #condition62 = (FinalＷeekdata.loc[FinalＷeekdata.index[i-1],'close'] > FinalＷeekdata.loc[FinalＷeekdata.index[i-2],"max"]) and (FinalＷeekdata.loc[FinalＷeekdata.index[i-1],'成交金額'] > FinalＷeekdata.loc[FinalＷeekdata.index[i-2],'成交金額']) and (FinalＷeekdata.loc[FinalＷeekdata.index[i],'close']<FinalＷeekdata.loc[FinalＷeekdata.index[i-1],"min"] )
            condition63 = (FinalＷeekdata.loc[FinalＷeekdata.index[i],'close'] < FinalＷeekdata.loc[FinalＷeekdata.index[i-1],"dline"]) and (FinalＷeekdata.loc[FinalＷeekdata.index[i-1],'close'] >= FinalＷeekdata.loc[FinalＷeekdata.index[i-1],"dline"])
        except:
            condition51 = True
            condition52 = True
            condition53 = True
            condition61 = True
            condition63 = True
        condition54 = condition51 or condition53 #or condition52
        condition64 = condition61 or condition63 #or condition62 

        #FinalＷeekdata['labelb'] = np.where((FinalＷeekdata['close']> FinalＷeekdata['upper_band1']) , 1, np.where((FinalＷeekdata['close']< FinalＷeekdata['lower_band1']),-1,1))

        #print(i)
        if FinalＷeekdata.loc[FinalＷeekdata.index[i],'close'] > FinalＷeekdata.loc[FinalＷeekdata.index[i],'upper_band1']:
            FinalＷeekdata.loc[FinalＷeekdata.index[i],'labelb'] = 1
        elif FinalＷeekdata.loc[FinalＷeekdata.index[i],'close'] < FinalＷeekdata.loc[FinalＷeekdata.index[i],'lower_band1']:
            FinalＷeekdata.loc[FinalＷeekdata.index[i],'labelb'] = -1
        else:
            FinalＷeekdata.loc[FinalＷeekdata.index[i],'labelb'] = FinalＷeekdata.loc[FinalＷeekdata.index[i-1],'labelb']

        if condition54 == True:
            barssince5 = 1
        else:
            barssince5 += 1

        if condition64 == True:
            barssince6 = 1
        else:
            barssince6 += 1


        if barssince5 < barssince6:
            FinalＷeekdata.loc[FinalＷeekdata.index[i],"all_kk"] = 1
        else:
            FinalＷeekdata.loc[FinalＷeekdata.index[i],"all_kk"] = -1

    FinalＷeekdata = FinalＷeekdata[FinalＷeekdata.index>FinalＷeekdata.index[-61]]

    # 設定左右子圖
    fig1_1 = make_subplots(
        rows = 1, 
        cols = 2, 
        horizontal_spacing = 0.1,
        vertical_spacing=0.02,subplot_titles = ["加權週線","加權日線"],
        specs = [[{"secondary_y":True}]*2]
        
    )
    

    ICweek = []
    finalweek = list(FinalＷeekdata['IC'].index)[-1]
    ICweek.append(finalweek[:5] + str(int(finalweek[-2:])+1))
    ICweek.append(finalweek[:5] + str(int(finalweek[-2:])+2))

    dataTPEx,ICdate1 = get_stock_data("TPEx")
    dataTPEx = dataTPEx[dataTPEx.index>kbars.index[0]]
    data50,ICdate1 = get_stock_data("00675L")
    data50 = data50[data50.index>kbars.index[0]]

    oldcol = list(kbars.columns)
    refcol = list(kbars.columns)
    newcol = list(dataTPEx.columns)
    refcol[1:10] = newcol[1:10]
    kbars.columns = refcol

    plot_stockdata(fig1_1,holidf,FinalＷeekdata,1,1,ICweek)
    plot_stockdata(fig1_1,holidf,kbars,1,2,ICdate)

    kbars.columns = oldcol
    

    # 設定左右子圖
    fig1_2 = make_subplots(
        rows = 1, 
        cols = 2, 
        horizontal_spacing = 0.1,
        vertical_spacing=0.02,subplot_titles = ["櫃買指數","富邦台灣50正2"],
        specs = [[{"secondary_y":True}]*2]
        
    )
    plot_stockdata(fig1_2,holidf,dataTPEx,1,1,ICdate)
    plot_stockdata(fig1_2,holidf,data50,1,2,ICdate)


    #fig1_2.update_traces(xaxis='x1',hoverlabel=dict(align='left'))

    
    # 隱藏周末與市場休市日期 ### 導入台灣的休市資料
    fig1_1.update_xaxes(
        rangeslider= {'visible':False},
        
                    row = 1, 
                    col = 1
    )

    # 隱藏周末與市場休市日期 ### 導入台灣的休市資料
    fig1_1.update_xaxes(
        rangeslider= {'visible':False},
        rangebreaks=[
            dict(bounds=['sat', 'mon']), # hide weekends, eg. hide sat to before mon
            dict(values=[str(holiday) for holiday in holidf[~(holidf["說明"].str.contains('開始交易') | holidf["說明"].str.contains('最後交易'))]["日期"].values]+['2023-08-03'])
        ],
                    row = 1, 
                    col = 2
    )


   # 設定圖的標題跟長寬
    fig1_1.update_annotations(font_size=12)
    fig1_1.update_layout(title_text = "", hovermode='x unified',#,hoverlabel=list(bgcolor='rgba(255,255,255,0.5),font=list(color='black')'),
                    width = 1200, 
                    height = 400,
               
                    yaxis2=dict(showgrid=False),
                    yaxis=dict(showgrid=False))
    
    st.plotly_chart(fig1_1)

    

    # 設定圖的標題跟長寬
    fig1_2.update_annotations(font_size=12)
    fig1_2.update_layout(title_text = "", 
                    width = 1200, 
                    height = 400,
              
                    yaxis2=dict(showgrid=False),
                    yaxis=dict(showgrid=False))

    st.plotly_chart(fig1_2)
    
    


    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0wNy0zMCAyMzowMTo0MSIsInVzZXJfaWQiOiJqZXlhbmdqYXUiLCJpcCI6IjExNC4zNC4xMjEuMTA0In0.WDAZzKGv4Du5JilaAR7o7M1whpnGaR-vMDuSeTBXhhA"
    url = "https://api.finmindtrade.com/api/v4/data?"

    parameter = {
        "dataset": "TaiwanOptionDaily",
        "data_id":"TXO",
        "start_date": datetime.strftime(datetime.today()- timedelta(days=7),'%Y-%m-%d'),
        "token": token, # 參考登入，獲取金鑰
    }
    data = requests.get(url, params=parameter)
    data = data.json()
    data = pd.DataFrame(data['data'])
    data = data[data["trading_session"] == 'position']
    data.date = pd.to_datetime(data.date)
    
    contract1 = enddate['契約月份'].values[0]
    try:
        contract2 = data[(data.date == enddate['最後結算日'].values[0]+1000000000*60*60*24)].contract_date.unique()[0]
    except:
        contract2 = ""

    data1 = data[(data.date < enddate['最後結算日'].values[0]) & (data.contract_date == contract1)]
    data2 = data[(data.date >= enddate['最後結算日'].values[0]) & (data.contract_date == contract2)]
    df = pd.concat([data1,data2])

    df = df[df['strike_price']>kbars['收盤指數'].values[-1]-500]
    df = df[df['strike_price']<kbars['收盤指數'].values[-1]+500]

    call_df = df.loc[(df['call_put'] == 'call')]
    index = call_df['strike_price'].unique()
    idx = np.sort(index)
    dates = call_df['date'].unique()
    call_t_df = pd.DataFrame(index = idx, columns = list(dates))
    # 取出各日期的未沖銷契約數 依序放入 dataframe 中
    for col in call_t_df.columns:
        call_t_df[col] = call_df.loc[call_df['date'] == col].set_index('strike_price')['open_interest']
        
    #空值填入 0
    call_t_df = call_t_df.fillna(0)


    put_df = df.loc[(df['call_put'] == 'put')]
    index = put_df['strike_price'].unique()
    idx = np.sort(index)
    dates = put_df['date'].unique()
    put_t_df = pd.DataFrame(index = idx, columns = list(dates))
    for col in put_t_df.columns:
        put_t_df[col] = put_df.loc[put_df['date'] == col].set_index('strike_price')['open_interest']

    #空值填入 0
    put_t_df = put_t_df.fillna(0)

    datecol = []
    titlelist = []
    for di in call_t_df.columns:
        datecol.append(str(di)[:10])
        titlelist.append(str(di)[:10] + " Call")
        titlelist.append(str(di)[:10] + " Put")
        
    # 設定左右子圖
    fig1 = make_subplots(
        rows = 1, 
        cols = 10, 
        horizontal_spacing = 0.02, 
        subplot_titles = titlelist
    )

    try:

        ## 圖一
        # 畫買權長條圖
        fig1.add_trace(go.Bar(y = call_t_df.index,
                            x = -call_t_df[datecol[0]],
                            orientation = 'h',
                            name = datecol[0] + ' Call',
                            #text = ("(" + (call_t_df[datecol[0]] - call_t_df['2021-11-3']).astype('int').astype('str') + ") " + call_t_df['2021-11-4'].astype('int').astype('str')),
                            marker = dict(color = 'red'),showlegend=False), 
                    row = 1, 
                    col = 1 )

        # 畫賣權長條圖
        fig1.add_trace(go.Bar(y = put_t_df.index,
                            x = put_t_df[datecol[0]],
                            orientation = 'h',
                            name = datecol[0] + ' Put',
                            #text = (put_t_df['2021-11-4'].astype('int').astype('str') + " (" + (put_t_df['2021-11-4'] - put_t_df['2021-11-3']).astype('int').astype('str') + ")"),
                            marker = dict(color = 'green'),showlegend=False), 
                    row = 1, 
                    col = 2 )


        # 設定圖的x跟y軸標題
        fig1.update_xaxes(tickvals = [-15000,  -10000,  -5000,  0],
                        ticktext = ['15k',  '10k',  '5k',  '0'], 
                        title_text = "未沖銷契約數",
                        row = 1, 
                        col = 1)

        fig1.update_xaxes(tickvals = [0, 5000, 10000, 15000],
                        ticktext = ['0',  '5k',  '10k',  '15k'], 
                        title_text = "未沖銷契約數",
                        row = 1, 
                        col = 2)

        fig1.update_yaxes(autorange = "reversed", 
                        showticklabels = False, 
                        title_text = "履約價",
                        row = 1, 
                        col = 1)

        fig1.update_yaxes(autorange = "reversed", 
                        row = 1, 
                        col = 2)


        ## 圖二
        # 畫買權長條圖
        fig1.add_trace(go.Bar(y = call_t_df.index,
                            x = -call_t_df[datecol[1]],
                            orientation = 'h',
                            name = datecol[1] + ' Call',
                            #text = ("(" + (call_t_df['2021-11-5'] - call_t_df['2021-11-4']).astype('int').astype('str') + ") " + call_t_df['2021-11-5'].astype('int').astype('str')),
                            marker = dict(color = 'red'),showlegend=False), 
                    row = 1, 
                    col = 3 )

        # 畫賣權長條圖
        fig1.add_trace(go.Bar(y = put_t_df.index,
                            x = put_t_df[datecol[1]],
                            orientation = 'h',
                            name = datecol[1]+' Put',
                            #text = (put_t_df['2021-11-5'].astype('int').astype('str') + " (" + (put_t_df['2021-11-5'] - put_t_df['2021-11-4']).astype('int').astype('str') + ")"),
                            marker = dict(color = 'green'),showlegend=False), 
                    row = 1, 
                    col = 4 )


        # 設定圖的x跟y軸標題
        fig1.update_xaxes(tickvals = [-15000,  -10000,  -5000,  0],
                        ticktext = ['15k',  '10k',  '5k',  '0'], 
                        title_text = "未沖銷契約數",
                        row = 1, 
                        col = 3)

        fig1.update_xaxes(tickvals = [0, 5000, 10000, 15000],
                        ticktext = ['0',  '5k',  '10k',  '15k'], 
                        title_text = "未沖銷契約數",
                        row = 1, 
                        col = 4)

        fig1.update_yaxes(autorange = "reversed", 
                        showticklabels = False, 
                        row = 1, 
                        col = 3)

        fig1.update_yaxes(autorange = "reversed", 
                        row = 1, 
                        col = 4)


        ## 圖三
        # 畫買權長條圖
        fig1.add_trace(go.Bar(y = call_t_df.index,
                            x = -call_t_df[datecol[2]],
                            orientation = 'h',
                            name = datecol[2] + ' Call',
                            #text = ("(" + (call_t_df['2021-11-8'] - call_t_df['2021-11-5']).astype('int').astype('str') + ") " + call_t_df['2021-11-8'].astype('int').astype('str')),
                            marker = dict(color = 'red'),showlegend=False), 
                    row = 1, 
                    col = 5 )

        # 畫賣權長條圖
        fig1.add_trace(go.Bar(y = put_t_df.index,
                            x = put_t_df[datecol[2]],
                            orientation = 'h',
                            name = datecol[2] + ' Put',
                            #text = (put_t_df['2021-11-8'].astype('int').astype('str') + " (" + (put_t_df['2021-11-8'] - put_t_df['2021-11-5']).astype('int').astype('str') + ")"),
                            marker = dict(color = 'green'),showlegend=False), 
                    row = 1, 
                    col = 6 )


        # 設定圖的x跟y軸標題
        fig1.update_xaxes(tickvals = [-15000,  -10000,  -5000,  0],
                        ticktext = ['15k',  '10k',  '5k',  '0'], 
                        title_text = "未沖銷契約數",
                        row = 1, 
                        col = 5)

        fig1.update_xaxes(tickvals = [0, 5000, 10000, 15000],
                        ticktext = ['0',  '5k',  '10k',  '15k'], 
                        title_text = "未沖銷契約數",
                        row = 1, 
                        col = 6)

        fig1.update_yaxes(autorange = "reversed", 
                        showticklabels = False, 
                        row = 1, 
                        col = 5)

        fig1.update_yaxes(autorange = "reversed", 
                        row = 1, 
                        col = 6)

        ## 圖四
        # 畫買權長條圖
        fig1.add_trace(go.Bar(y = call_t_df.index,
                            x = -call_t_df[datecol[3]],
                            orientation = 'h',
                            name = datecol[3]+' Call',
                            #text = ("(" + (call_t_df['2021-11-9'] - call_t_df['2021-11-8']).astype('int').astype('str') + ") " + call_t_df['2021-11-9'].astype('int').astype('str')),
                            marker = dict(color = 'red'),showlegend=False), 
                    row = 1, 
                    col = 7 )

        # 畫賣權長條圖
        fig1.add_trace(go.Bar(y = put_t_df.index,
                            x = put_t_df[datecol[3]],
                            orientation = 'h',
                            name = datecol[3]+' Put',
                            #text = (put_t_df['2021-11-9'].astype('int').astype('str') + " (" + (put_t_df['2021-11-9'] - put_t_df['2021-11-8']).astype('int').astype('str') + ")"),
                            marker = dict(color = 'green'),showlegend=False), 
                    row = 1, 
                    col = 8 )


        # 設定圖的x跟y軸標題
        fig1.update_xaxes(tickvals = [-15000,  -10000,  -5000,  0],
                        ticktext = ['15k',  '10k',  '5k',  '0'], 
                        title_text = "未沖銷契約數",
                        row = 1, 
                        col = 7)

        fig1.update_xaxes(tickvals = [0, 5000, 10000, 15000],
                        ticktext = ['0',  '5k',  '10k',  '15k'], 
                        title_text = "未沖銷契約數",
                        row = 1, 
                        col = 8)

        fig1.update_yaxes(autorange = "reversed", 
                        showticklabels = False, 
                        row = 1, 
                        col = 7)

        fig1.update_yaxes(autorange = "reversed", 
                        row = 1, 
                        col = 8)

        ## 圖五
        try:
            # 畫買權長條圖
            fig1.add_trace(go.Bar(y = call_t_df.index,
                                x = -call_t_df[datecol[4]],
                                orientation = 'h',
                                name = datecol[4]+' Call',
                                #text = ("(" + (call_t_df['2021-11-10'] - call_t_df['2021-11-9']).astype('int').astype('str') + ") " + call_t_df['2021-11-10'].astype('int').astype('str')),
                                marker = dict(color = 'red'),showlegend=False), 
                        row = 1, 
                        col = 9 )

            # 畫賣權長條圖
            fig1.add_trace(go.Bar(y = put_t_df.index,
                                x = put_t_df[datecol[4]],
                                orientation = 'h',
                                name = datecol[4]+' Put',
                                #text = (put_t_df['2021-11-10'].astype('int').astype('str') + " (" + (put_t_df['2021-11-10'] - put_t_df['2021-11-9']).astype('int').astype('str') + ")"),
                                marker = dict(color = 'green'),showlegend=False), 
                        row = 1, 
                        col = 10 )


            # 設定圖的x跟y軸標題
            fig1.update_xaxes(tickvals = [-15000,  -10000,  -5000,  0],
                            ticktext = ['15k',  '10k',  '5k',  '0'], 
                            title_text = "未沖銷契約數",
                            row = 1, 
                            col = 9)

            fig1.update_xaxes(tickvals = [0, 5000, 10000, 15000],
                            ticktext = ['0',  '5k',  '10k',  '15k'], 
                            title_text = "未沖銷契約數",
                            row = 1, 
                            col = 10)

            fig1.update_yaxes(autorange = "reversed", 
                            showticklabels = False, 
                            row = 1, 
                            col = 9)

            fig1.update_yaxes(autorange = "reversed", 
                            
                            row = 1, 
                            col = 10)
        except:
            pass
        for hi in range(10):
            try:
                fig1.add_hline(y=kbars['收盤指數'].values[-1], line_width=1,line_dash="dash", line_color="black",name='當天收盤',row=1,col=hi)#, line_dash="dash"
            except:
                pass

        # 設定圖的標題跟長寬
        fig1.update_annotations(font_size=12)
        fig1.update_layout(title_text = "臺指選擇權 未沖銷契約數 支撐壓力圖 (週)", 
                        width = 1200, 
                        height = 650)

        #fig.show()
        st.plotly_chart(fig1)


        contract1 = data[(data.date == data.date.max())&(~data.contract_date.str.contains("W"))].contract_date.min()
        try:
            contract2 = data[(data.date == data.date.min())&(~data.contract_date.str.contains("W"))].contract_date.min()
        except:
            contract2 = ""
        #contract1 
        #contract2
        #data
        #enddate[~enddate["契約月份"].str.contains("W")]['最後結算日']
        #enddatemonth = enddate[~enddate["契約月份"].str.contains("W")]['最後結算日'].values[0]

        data1 = data[(data.date < enddate[~enddate["契約月份"].str.contains("W")]['最後結算日'].values[0]) & (data.contract_date == contract2)]
        data2 = data[(data.date >= enddate[~enddate["契約月份"].str.contains("W")]['最後結算日'].values[0]) & (data.contract_date == contract1)]
        df = pd.concat([data1,data2])
        #data2

        df = df[df['strike_price']>kbars['收盤指數'].values[-1]-1000]
        dfmonth = df[df['strike_price']<kbars['收盤指數'].values[-1]+1000]

        contract_month = data[(data.date == data.date.max())&(~data.contract_date.str.contains("W"))].contract_date.min()
        #dfmonth = data[(data.date == data.date.max())&(data.contract_date==contract_month)]

        call_df = dfmonth.loc[(dfmonth['call_put'] == 'call')]
        index = call_df['strike_price'].unique()
        idx = np.sort(index)
        dates = call_df['date'].unique()
        call_t_df = pd.DataFrame(index = idx, columns = list(dates))
        # 取出各日期的未沖銷契約數 依序放入 dataframe 中
        for col in call_t_df.columns:
            call_t_df[col] = call_df.loc[call_df['date'] == col].set_index('strike_price')['open_interest']
            
        #空值填入 0
        call_t_df = call_t_df.fillna(0)


        put_df = dfmonth.loc[(dfmonth['call_put'] == 'put')]
        index = put_df['strike_price'].unique()
        idx = np.sort(index)
        dates = put_df['date'].unique()
        put_t_df = pd.DataFrame(index = idx, columns = list(dates))
        for col in put_t_df.columns:
            put_t_df[col] = put_df.loc[put_df['date'] == col].set_index('strike_price')['open_interest']

        #空值填入 0
        put_t_df = put_t_df.fillna(0)
    except:
        pass
    #call_t_df 
    #  設定左右子圖
    fig2 = make_subplots(
        rows = 1, 
        cols = 10, 
        horizontal_spacing = 0.02, 
        subplot_titles = titlelist
    )
    try:
        ## 圖一
        # 畫買權長條圖
        fig2.add_trace(go.Bar(y = call_t_df.index,
                            x = -call_t_df[datecol[0]],
                            orientation = 'h',
                            name = datecol[0] + ' Call',
                            #text = ("(" + (call_t_df[datecol[0]] - call_t_df['2021-11-3']).astype('int').astype('str') + ") " + call_t_df['2021-11-4'].astype('int').astype('str')),
                            marker = dict(color = 'red'),showlegend=False), 
                    row = 1, 
                    col = 1 )

        # 畫賣權長條圖
        fig2.add_trace(go.Bar(y = put_t_df.index,
                            x = put_t_df[datecol[0]],
                            orientation = 'h',
                            name = datecol[0] + ' Put',
                            #text = (put_t_df['2021-11-4'].astype('int').astype('str') + " (" + (put_t_df['2021-11-4'] - put_t_df['2021-11-3']).astype('int').astype('str') + ")"),
                            marker = dict(color = 'green'),showlegend=False), 
                    row = 1, 
                    col = 2 )


        # 設定圖的x跟y軸標題
        fig2.update_xaxes(tickvals = [-15000,  -10000,  -5000,  0],
                        ticktext = ['15k',  '10k',  '5k',  '0'], 
                        title_text = "未沖銷契約數",
                        row = 1, 
                        col = 1)

        fig2.update_xaxes(tickvals = [0, 5000, 10000, 15000],
                        ticktext = ['0',  '5k',  '10k',  '15k'], 
                        title_text = "未沖銷契約數",
                        row = 1, 
                        col = 2)

        fig2.update_yaxes(autorange = "reversed", 
                        showticklabels = False, 
                        title_text = "履約價",
                        row = 1, 
                        col = 1)

        fig2.update_yaxes(autorange = "reversed", 
                        row = 1, 
                        col = 2)


        ## 圖二
        # 畫買權長條圖
        fig2.add_trace(go.Bar(y = call_t_df.index,
                            x = -call_t_df[datecol[1]],
                            orientation = 'h',
                            name = datecol[1] + ' Call',
                            #text = ("(" + (call_t_df['2021-11-5'] - call_t_df['2021-11-4']).astype('int').astype('str') + ") " + call_t_df['2021-11-5'].astype('int').astype('str')),
                            marker = dict(color = 'red'),showlegend=False), 
                    row = 1, 
                    col = 3 )

        # 畫賣權長條圖
        fig2.add_trace(go.Bar(y = put_t_df.index,
                            x = put_t_df[datecol[1]],
                            orientation = 'h',
                            name = datecol[1]+' Put',
                            #text = (put_t_df['2021-11-5'].astype('int').astype('str') + " (" + (put_t_df['2021-11-5'] - put_t_df['2021-11-4']).astype('int').astype('str') + ")"),
                            marker = dict(color = 'green'),showlegend=False), 
                    row = 1, 
                    col = 4 )


        # 設定圖的x跟y軸標題
        fig2.update_xaxes(tickvals = [-15000,  -10000,  -5000,  0],
                        ticktext = ['15k',  '10k',  '5k',  '0'], 
                        title_text = "未沖銷契約數",
                        row = 1, 
                        col = 3)

        fig2.update_xaxes(tickvals = [0, 5000, 10000, 15000],
                        ticktext = ['0',  '5k',  '10k',  '15k'], 
                        title_text = "未沖銷契約數",
                        row = 1, 
                        col = 4)

        fig2.update_yaxes(autorange = "reversed", 
                        showticklabels = False, 
                        row = 1, 
                        col = 3)

        fig2.update_yaxes(autorange = "reversed", 
                        row = 1, 
                        col = 4)


        ## 圖三
        # 畫買權長條圖
        fig2.add_trace(go.Bar(y = call_t_df.index,
                            x = -call_t_df[datecol[2]],
                            orientation = 'h',
                            name = datecol[2] + ' Call',
                            #text = ("(" + (call_t_df['2021-11-8'] - call_t_df['2021-11-5']).astype('int').astype('str') + ") " + call_t_df['2021-11-8'].astype('int').astype('str')),
                            marker = dict(color = 'red'),showlegend=False), 
                    row = 1, 
                    col = 5 )

        # 畫賣權長條圖
        fig2.add_trace(go.Bar(y = put_t_df.index,
                            x = put_t_df[datecol[2]],
                            orientation = 'h',
                            name = datecol[2] + ' Put',
                            #text = (put_t_df['2021-11-8'].astype('int').astype('str') + " (" + (put_t_df['2021-11-8'] - put_t_df['2021-11-5']).astype('int').astype('str') + ")"),
                            marker = dict(color = 'green'),showlegend=False), 
                    row = 1, 
                    col = 6 )


        # 設定圖的x跟y軸標題
        fig2.update_xaxes(tickvals = [-15000,  -10000,  -5000,  0],
                        ticktext = ['15k',  '10k',  '5k',  '0'], 
                        title_text = "未沖銷契約數",
                        row = 1, 
                        col = 5)

        fig2.update_xaxes(tickvals = [0, 5000, 10000, 15000],
                        ticktext = ['0',  '5k',  '10k',  '15k'], 
                        title_text = "未沖銷契約數",
                        row = 1, 
                        col = 6)

        fig2.update_yaxes(autorange = "reversed", 
                        showticklabels = False, 
                        row = 1, 
                        col = 5)

        fig2.update_yaxes(autorange = "reversed", 
                        row = 1, 
                        col = 6)

        ## 圖四
        # 畫買權長條圖
        fig2.add_trace(go.Bar(y = call_t_df.index,
                            x = -call_t_df[datecol[3]],
                            orientation = 'h',
                            name = datecol[3]+' Call',
                            #text = ("(" + (call_t_df['2021-11-9'] - call_t_df['2021-11-8']).astype('int').astype('str') + ") " + call_t_df['2021-11-9'].astype('int').astype('str')),
                            marker = dict(color = 'red'),showlegend=False), 
                    row = 1, 
                    col = 7 )

        # 畫賣權長條圖
        fig2.add_trace(go.Bar(y = put_t_df.index,
                            x = put_t_df[datecol[3]],
                            orientation = 'h',
                            name = datecol[3]+' Put',
                            #text = (put_t_df['2021-11-9'].astype('int').astype('str') + " (" + (put_t_df['2021-11-9'] - put_t_df['2021-11-8']).astype('int').astype('str') + ")"),
                            marker = dict(color = 'green'),showlegend=False), 
                    row = 1, 
                    col = 8 )


        # 設定圖的x跟y軸標題
        fig2.update_xaxes(tickvals = [-15000,  -10000,  -5000,  0],
                        ticktext = ['15k',  '10k',  '5k',  '0'], 
                        title_text = "未沖銷契約數",
                        row = 1, 
                        col = 7)

        fig2.update_xaxes(tickvals = [0, 5000, 10000, 15000],
                        ticktext = ['0',  '5k',  '10k',  '15k'], 
                        title_text = "未沖銷契約數",
                        row = 1, 
                        col = 8)

        fig2.update_yaxes(autorange = "reversed", 
                        showticklabels = False, 
                        row = 1, 
                        col = 7)

        fig2.update_yaxes(autorange = "reversed", 
                        row = 1, 
                        col = 8)
    except:
        pass

    ## 圖五
    try:
        # 畫買權長條圖
        fig2.add_trace(go.Bar(y = call_t_df.index,
                            x = -call_t_df[datecol[4]],
                            orientation = 'h',
                            name = datecol[4]+' Call',
                            #text = ("(" + (call_t_df['2021-11-10'] - call_t_df['2021-11-9']).astype('int').astype('str') + ") " + call_t_df['2021-11-10'].astype('int').astype('str')),
                            marker = dict(color = 'red'),showlegend=False), 
                    row = 1, 
                    col = 9 )

        # 畫賣權長條圖
        fig2.add_trace(go.Bar(y = put_t_df.index,
                            x = put_t_df[datecol[4]],
                            orientation = 'h',
                            name = datecol[4]+' Put',
                            #text = (put_t_df['2021-11-10'].astype('int').astype('str') + " (" + (put_t_df['2021-11-10'] - put_t_df['2021-11-9']).astype('int').astype('str') + ")"),
                            marker = dict(color = 'green'),showlegend=False), 
                    row = 1, 
                    col = 10 )


        # 設定圖的x跟y軸標題
        fig2.update_xaxes(tickvals = [-15000,  -10000,  -5000,  0],
                        ticktext = ['15k',  '10k',  '5k',  '0'], 
                        title_text = "未沖銷契約數",
                        row = 1, 
                        col = 9)

        fig2.update_xaxes(tickvals = [0, 5000, 10000, 15000],
                        ticktext = ['0',  '5k',  '10k',  '15k'], 
                        title_text = "未沖銷契約數",
                        row = 1, 
                        col = 10)

        fig2.update_yaxes(autorange = "reversed", 
                        showticklabels = False, 
                        row = 1, 
                        col = 9)

        fig2.update_yaxes(autorange = "reversed", 
                        
                        row = 1, 
                        col = 10)
    except:
        pass

    for hi in range(10):
        try:
            fig2.add_hline(y=kbars['收盤指數'].values[-1], line_width=1,line_dash="dash", line_color="black",name='當天收盤',row=1,col=hi)#, line_dash="dash"
        except:
            pass

    # 設定圖的標題跟長寬
    fig2.update_annotations(font_size=12)
    fig2.update_layout(title_text = "臺指選擇權 未沖銷契約數 支撐壓力圖（月）", 
                    width = 1200, 
                    height = 650)

    #fig.show()
    st.plotly_chart(fig2)


