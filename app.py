import time

import streamlit as st

import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import requests


connection = sqlite3.connect('主圖資料.sqlite3')


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


taiex = pd.read_sql("select distinct * from taiex", connection, parse_dates=['日期'], index_col=['日期'])
taiex_vol = pd.read_sql("select distinct * from taiex_vol", connection, parse_dates=['日期'], index_col=['日期'])
cost_df = pd.read_sql("select distinct Date as [日期], Cost as [外資成本] from cost", connection, parse_dates=['日期'], index_col=['日期']).dropna()
cost_df["外資成本"] = cost_df["外資成本"].astype('int')
limit_df = pd.read_sql("select distinct * from [limit]", connection, parse_dates=['日期'], index_col=['日期'])

inves_limit = limit_df[limit_df["身份別"] == "外資"][['上極限', '下極限']]
dealer_limit = limit_df[limit_df["身份別"] == "自營商"][['上極限', '下極限']]

inves_limit.columns = ["外資上極限","外資下極限"]
dealer_limit.columns = ["自營商上極限","自營商下極限"]

kbars = taiex.join(taiex_vol).join(cost_df).join(dealer_limit).join(inves_limit)

enddate = pd.read_sql("select * from end_date order by 最後結算日 desc", connection, parse_dates=['最後結算日'])


holidf = pd.read_sql("select * from holiday", connection)

holilist = [str(holiday) for holiday in holidf[~(holidf["說明"].str.contains('開始交易') | holidf["說明"].str.contains('最後交易'))]["日期"].values]
#holilist["日期"] = pd.to_datetime(holilist["日期"])

ordervolumn = pd.read_sql("select distinct * from ordervolumn", connection, parse_dates=['日期'], index_col=['日期'])
putcallsum = pd.read_sql("select 日期, max(價平和) as 價平和 from putcallsum group by 日期", connection, parse_dates=['日期'], index_col=['日期'])
putcallsum_month = pd.read_sql("select 日期, max(月選擇權價平和) as 月價平和 from putcallsum_month group by 日期", connection, parse_dates=['日期'], index_col=['日期'])
kbars = kbars.join(ordervolumn).join(putcallsum).join(putcallsum_month)

# 計算布林帶指標
kbars['MA'] = kbars['收盤指數'].rolling(20).mean()
kbars['std'] = kbars['收盤指數'].rolling(20).std()
kbars['upper_band'] = kbars['MA'] + 2 * kbars['std']
kbars['lower_band'] = kbars['MA'] - 2 * kbars['std']
kbars['upper_band1'] = kbars['MA'] + 1 * kbars['std']
kbars['lower_band1'] = kbars['MA'] - 1 * kbars['std']

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
    
kbars["MAX_MA"] = kbars["最高指數"] - kbars["MA"]
kbars["MIN_MA"] = kbars["最低指數"] - kbars["MA"]

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
        condition52 = (kbars.loc[kbars.index[i-1],'收盤指數'] < kbars.loc[kbars.index[i-2],"最低指數"]) and (kbars.loc[kbars.index[i-1],'成交金額'] > kbars.loc[kbars.index[i-2],'成交金額']) and (kbars.loc[kbars.index[i],'收盤指數']>kbars.loc[kbars.index[i-1],"最高指數"] )
        condition53 = (kbars.loc[kbars.index[i],'收盤指數'] > kbars.loc[kbars.index[i-1],"uline"]) and (kbars.loc[kbars.index[i-1],'收盤指數'] <= kbars.loc[kbars.index[i-1],"uline"])

        condition61 = (kbars.loc[kbars.index[i-1],"最低指數"] > kbars.loc[kbars.index[i-2],"最高指數"] ) and (kbars.loc[kbars.index[i],"最高指數"] < kbars.loc[kbars.index[i-1],"最低指數"] )
        condition62 = (kbars.loc[kbars.index[i-1],'收盤指數'] > kbars.loc[kbars.index[i-2],"最高指數"]) and (kbars.loc[kbars.index[i-1],'成交金額'] > kbars.loc[kbars.index[i-2],'成交金額']) and (kbars.loc[kbars.index[i],'收盤指數']<kbars.loc[kbars.index[i-1],"最低指數"] )
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

    print(i)
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

for dateidx in range(0,len(kbars.index[-60:]),5):

    try:
        datei = kbars.index[-60:][dateidx+10]
        days20 = kbars[(kbars.index> kbars.index[-80:][dateidx+10]) & (kbars.index<datei )]
        max_days20 = days20["九點累積委託賣出數量"].values.max()
        min_days20 = days20["九點累積委託賣出數量"].values.min()
    except:
        pass


    try:
        max_days_didx = np.where(kbars["九點累積委託賣出數量"].values[-60:]==max_days20)[0][0]
        min_days_didx = np.where(kbars["九點累積委託賣出數量"].values[-60:]==min_days20)[0][0]
        #max_days_didx
        try :
            if max_days20 not in max_days20_list and max_days20 > kbars[(kbars.index == kbars.index[-60:][max_days_didx+1])]["九點累積委託賣出數量"].values[0] and max_days20 > kbars[(kbars.index == kbars.index[-60:][max_days_didx-1])]["九點累積委託賣出數量"].values[0]:
                max_days20_list.append(max_days20)
                max_days20_x.append(days20[days20["九點累積委託賣出數量"]==max_days20].index.values[0])
        except:
            pass
        try:
            if min_days20 not in min_days20_list and min_days20 < kbars[(kbars.index == kbars.index[-60:][min_days_didx+1])]["九點累積委託賣出數量"].values[0] and min_days20 < kbars[(kbars.index == kbars.index[-60:][min_days_didx-1])]["九點累積委託賣出數量"].values[0]:
                min_days20_list.append(min_days20)
                min_days20_x.append(days20[days20["九點累積委託賣出數量"]==min_days20].index.values[0])
        except:
            continue
    except:
        continue

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
kbars = kbars.dropna()
kbars = kbars[kbars.index > kbars.index[-60]]

#kbars['labelb'] = np.where(kbars['收盤指數']< kbars['lower_band1'], -1, 1)



def fillcol(label):
    if label >= 1:
        return 'rgba(0,250,0,0.3)'
    else:
        return 'rgba(0,256,256,0.3)'

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

tab1, tab2, tab3 = st.tabs(["主圖", "支撐壓力","Raw Data"])

with tab1:

    st.sidebar.write('結算日顯示')
    option_month = st.sidebar.checkbox('月結算日', value = True)
    option_week = st.sidebar.checkbox('週結算日', value = False)


    st.sidebar.write('附圖選擇')


    option_2c = st.sidebar.checkbox('開盤賣張張數', value = True)
    option_2d = st.sidebar.checkbox('價平和', value = True)
    option_2e = st.sidebar.checkbox('月價平和日差', value = True)
    option_2f = st.sidebar.checkbox('月結趨勢', value = True)
    options_vice = [ option_2c , option_2d, option_2e , option_2f]

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
    decreasing_color = 'rgb(30, 144, 255)'

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
    fig.add_trace(go.Scatter(x=list(kbars['IC'].index)[1:]+[ICdate[0]],
                    y=kbars['外資成本'].shift(1).values,
                    mode='lines',
                    line=dict(color='yellow'),
                    name='外資成本'),row=1, col=1, secondary_y= True)


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
        print(bandstart,bandend)
        if kbars["labelb"].values[bandstart+1] == 1:
            fig.add_traces(go.Scatter(x=kbars.index[bandstart:bandend], y = kbars['lower_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),showlegend=False),secondary_ys= [True,True])
                
            fig.add_traces(go.Scatter(x=kbars.index[bandstart:bandend], y = kbars['upper_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),
                                        fill='tonexty', 
                                        fillcolor = 'rgba(256,256,0,0.4)',showlegend=False
                                        ),secondary_ys= [True,True])
        else:


            fig.add_traces(go.Scatter(x=kbars.index[bandstart:bandend], y = kbars['lower_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),showlegend=False), secondary_ys= [True,True])
                
            fig.add_traces(go.Scatter(x=kbars.index[bandstart:bandend], y = kbars['upper_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),
                                        fill='tonexty', 
                                        fillcolor = 'rgba(137, 207, 240,0.4)',showlegend=False
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

    fig.add_trace(go.Scatter(x=kbars.index,
                            y=kbars['MA'],
                            mode='lines',
                            line=dict(color='green'),
                            name='MA'),row=1, col=1, secondary_y= True)

    fig.add_trace(go.Scatter(x=list(kbars['IC'].index)[2:]+ICdate,
                            y=kbars['IC'].values,
                            mode='lines',
                            line=dict(color='orange'),
                            name='IC操盤線'),row=1, col=1, secondary_y= True)
    
    fig.add_trace(go.Scatter(x=[kbars.index[0],kbars.index[0]],y=[15500,17500], line_width=0.1, line_color="green",name='月結算日',showlegend=False),row=1, col=1)
    if option_month == True:
        for i in enddate[~enddate["契約月份"].str.contains("W")]['最後結算日']:
            if i > kbars.index[0] :#and i!=enddate[~enddate["契約月份"].str.contains("W")]['最後結算日'].values[6]:
                fig.add_vline(x=i, line_width=1, line_color="green",name='月結算日',row=1, col=1)

    #enddate['最後結算日'].values
    #enddate.groupby(enddate['最後結算日'].dt.month)['最後結算日'].max()
    #list(enddate['最後結算日'].values)[:3]
    if option_week == True:
        for i in enddate['最後結算日']:
            if i > kbars.index[0] :# and i!=enddate.groupby(enddate['最後結算日'].dt.month)['最後結算日'].max()[6] and i not in enddate.groupby(enddate['最後結算日'].dt.month)['最後結算日'].max():
                fig.add_vline(x=i, line_width=1,line_dash="dash", line_color="blue",name='週結算日')#, line_dash="dash"
        #fig.add_hrect(y0=0.9, y1=2.6, line_width=0, fillcolor="red", opacity=0.2)



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



    ### 成交量圖製作 ###
    volume_colors = [red_color if kbars['收盤指數'][i] > kbars['收盤指數'][i-1] else green_color for i in range(len(kbars['收盤指數']))]
    volume_colors[0] = green_color

    #fig.add_trace(go.Bar(x=kbars.index, y=kbars['成交金額'], name='Volume', marker=dict(color=volume_colors),showlegend=False), row=optvrank[0], col=1)
    fig.add_trace(go.Bar(x=kbars.index, y=kbars['成交金額'], name='Volume', marker=dict(color=volume_colors)), row=1, col=1)

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
        fig.add_trace(go.Scatter(x=kbars.index, y=kbars['九點累積委託賣出數量'], name='Volume',showlegend=False), row=optvrank[0], col=1)
        fig.add_scatter(x=np.array(max_days20_x), y=np.array(max_days20_list),marker=dict(color = blue_color,size=5),showlegend=False,mode = 'markers', row=optvrank[0], col=1)
        fig.add_scatter(x=np.array(min_days20_x), y=np.array(min_days20_list),marker=dict(color = orange_color,size=5),showlegend=False,mode = 'markers', row=optvrank[0], col=1)
        fig.update_yaxes(title_text="開盤賣張", row=optvrank[0], col=1)
    ## 價平和
    if optvrank[1] != 0:
        PCsum_colors = [increasing_color if kbars['價平和'][i] > kbars['價平和'][i-1] else decreasing_color for i in range(len(kbars['價平和']))]
        PCsum_colors[0] = decreasing_color
        fig.add_trace(go.Bar(x=kbars.index, y=kbars['價平和'], name='PCsum', marker=dict(color=PCsum_colors),showlegend=False), row=optvrank[1], col=1)
        #fig.add_hline(y = 50, line_width=0.2,line_dash="dash", line_color="blue", row=optvrank[1], col=1)
        for i in range(1,int(max(kbars['價平和'].values)//50)+1):
            fig.add_trace(go.Scatter(x=kbars.index,y=[i*50]*len(kbars.index),showlegend=False, line_width=0.5,line_dash="dash", line_color="black"), row=optvrank[1], col=1)
        fig.update_yaxes(title_text="價平和", row=optvrank[1], col=1)

        

    ## MA差
    
    #notshowdate
    if optvrank[2] != 0:
        fig.add_trace(go.Bar(x=kbars[(kbars['月價平和日差']>0)&(~kbars.index.isin(notshowdate))].index, y=(kbars[(kbars['月價平和日差']>0)&(~kbars.index.isin(notshowdate))]['月價平和日差']), name='月價平和日差',marker=dict(color = red_color_full),showlegend=False), row=optvrank[2], col=1)
        fig.add_trace(go.Bar(x=kbars[(kbars['月價平和日差']<=0)&(~kbars.index.isin(notshowdate))].index, y=(kbars[(kbars['月價平和日差']<=0)&(~kbars.index.isin(notshowdate))]['月價平和日差']), name='月價平和日差',marker=dict(color = blue_color),showlegend=False), row=optvrank[2], col=1)
        fig.update_yaxes(title_text="月價平和日差", row=optvrank[2], col=1)
    ## 月結趨勢
    if optvrank[3] != 0:
        fig.add_trace(go.Bar(x=kbars.index, y=kbars['end_high'], name='MAX_END',marker=dict(color = black_color),showlegend=False), row=optvrank[3], col=1)
        fig.add_trace(go.Bar(x=kbars.index, y=kbars['end_low'], name='MIN_END',marker=dict(color = gray_color),showlegend=False), row=optvrank[3], col=1)
        fig.update_yaxes(title_text="月結趨勢", row=optvrank[3], col=1, tickfont=dict(size=8))
    
    
    ##外資買賣超
    fig.add_trace(go.Bar(x=dfbuysell[dfbuysell['ForeBuySell']>0].index, y=(dfbuysell[dfbuysell['ForeBuySell']>0]["ForeBuySell"]).round(2), name='外資買賣超',marker=dict(color = red_color_full),showlegend=False), row=optvrank[3]+1, col=1)
    fig.add_trace(go.Bar(x=dfbuysell[dfbuysell['ForeBuySell']<=0].index, y=(dfbuysell[dfbuysell['ForeBuySell']<=0]["ForeBuySell"]).round(2), name='外資買賣超',marker=dict(color = blue_color),showlegend=False), row=optvrank[3]+1, col=1)
    #fig.add_trace(go.Bar(x=bank8.index, y=bank8["八大行庫買賣超金額"]/10000, name='eightbank',showlegend=False), row=optvrank[3]+2, col=1)
    fig.update_yaxes(title_text="外資買賣超(億元)", row=optvrank[3]+1, col=1)


    
    ## 外資臺股期貨未平倉淨口數
    #fut_colors = [red_color_full if kbars['收盤指數'][i] > kbars['收盤指數'][i-1] else blue_color for i in range(len(kbars['收盤指數']))]
    #fut_colors[0] = blue_color
    fut_colors = [increasing_color if futdf['多空未平倉口數淨額'][i] > futdf['多空未平倉口數淨額'][i-1] else decreasing_color for i in range(len(futdf['多空未平倉口數淨額']))]
    fut_colors[0] = decreasing_color
    #fig.add_trace(go.Bar(x=kbars.index, y=kbars['成交金額'], name='Volume', marker=dict(color=volume_colors)), row=1, col=1, secondary_y= True)
    fig.add_trace(go.Bar(x=futdf.index, y=futdf['多空未平倉口數淨額'], name='fut', marker=dict(color=fut_colors),showlegend=False), row=optvrank[3]+2, col=1)
    #fig.add_trace(go.Bar(x=bank8.index, y=bank8["八大行庫買賣超金額"]/10000, name='eightbank',showlegend=False), row=optvrank[3]+2, col=1)
    fig.update_yaxes(title_text="外資未平倉淨口數", row=optvrank[3]+2, col=1)

    
    

    #put call ratio
    #fig.add_trace(go.Scatter(x=kbars.index,y=kbars['收盤指數'],
    #                mode='lines',
    #                line=dict(color='black'),
    #                name='收盤指數',showlegend=False),row=optvrank[3]+1, col=1)
    #fig.add_trace(go.Bar(x=CPratio.index, y=CPratio['買賣權未平倉量比率%']-100, name='PC_Ratio',showlegend=False), row=optvrank[3]+4, col=1)
    #fig.update_yaxes(title_text="PutCallRatio", row=optvrank[3]+4, col=1)
    

    #選擇權外資OI
    fig.add_trace(go.Bar(x=TXOOIdf.index, y=(TXOOIdf["買買賣賣"]), name='買買權+賣賣權',marker=dict(color = red_color_full),showlegend=False), row=optvrank[3]+3, col=1)
    fig.add_trace(go.Bar(x=TXOOIdf.index, y=(TXOOIdf["買賣賣買"]), name='買賣權+賣買權',marker=dict(color = blue_color),showlegend=False), row=optvrank[3]+3, col=1)
    #fig.add_trace(go.Bar(x=bank8.index, y=bank8["八大行庫買賣超金額"]/10000, name='eightbank',showlegend=False), row=optvrank[3]+2, col=1)3
    fig.update_yaxes(title_text="選擇權外資OI", row=optvrank[3]+3, col=1)

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
            print(datei,one,two/100000000,three/100000000)
        except:
            continue
    fin = np.array(fin)
    find = np.array(find)
    fig.add_trace(go.Bar(x=find[fin>0], y=fin[fin>0], name='外資期現選心態',marker=dict(color = red_color_full),showlegend=False), row=optvrank[3]+4, col=1)
    fig.add_trace(go.Bar(x=find[fin<=0], y=fin[fin<=0], name='外資期現選心態',marker=dict(color = blue_color),showlegend=False), row=optvrank[3]+4, col=1)
    fig.update_yaxes(title_text="外資期現選心態", row=optvrank[3]+4, col=1)


    ## 小台散戶多空比
    
    fig.add_trace(go.Bar(x=dfMTX[dfMTX['MTXRatio']>0].index, y=(dfMTX[dfMTX['MTXRatio']>0]['MTXRatio']*100).round(2), name='小台散戶多空比',marker=dict(color = orange_color),showlegend=False), row=optvrank[3]+5, col=1)
    fig.add_trace(go.Bar(x=dfMTX[dfMTX['MTXRatio']<=0].index, y=(dfMTX[dfMTX['MTXRatio']<=0]['MTXRatio']*100).round(2), name='小台散戶多空比',marker=dict(color = green_color_full),showlegend=False), row=optvrank[3]+5, col=1)
    #fig.add_trace(go.Bar(x=bank8.index, y=bank8["八大行庫買賣超金額"]/10000, name='eightbank',showlegend=False), row=optvrank[3]+2, col=1)
    fig.update_yaxes(title_text="小台散戶多空比", row=optvrank[3]+5, col=1)

    

    #八大行庫買賣超
    fig.add_trace(go.Bar(x=bank8[bank8["八大行庫買賣超金額"]>0].index, y=(bank8[bank8["八大行庫買賣超金額"]>0]["八大行庫買賣超金額"]/100000).round(2), name='八大行庫買賣超',marker=dict(color = orange_color),showlegend=False), row=optvrank[3]+6, col=1)
    fig.add_trace(go.Bar(x=bank8[bank8["八大行庫買賣超金額"]<=0].index, y=(bank8[bank8["八大行庫買賣超金額"]<=0]["八大行庫買賣超金額"]/100000).round(2), name='八大行庫買賣超',marker=dict(color = green_color_full),showlegend=False), row=optvrank[3]+6, col=1)
    #fig.add_trace(go.Bar(x=bank8.index, y=bank8["八大行庫買賣超金額"]/10000, name='eightbank',showlegend=False), row=optvrank[3]+2, col=1)
    fig.update_yaxes(title_text="八大行庫", row=optvrank[3]+6, col=1)


    
    fig.add_trace(go.Scatter(x=dfMargin.index, y=dfMargin['MarginRate'],marker=dict(color = gray_color),line_width=3, name='MarginRate',showlegend=False), row=optvrank[3]+7, col=1)
    fig.update_yaxes(title_text="大盤融資資維持率", row=optvrank[3]+7, col=1)    



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

    fig.add_trace(go.Scatter(x=TaiwanExchangeRate[(TaiwanExchangeRate.date>kbars.index[0])&(TaiwanExchangeRate.date!=datetime.strptime('2023-08-03', '%Y-%m-%d'))].date, y=TaiwanExchangeRate[(TaiwanExchangeRate.date>kbars.index[0])&(TaiwanExchangeRate.date!=datetime.strptime('2023-08-03', '%Y-%m-%d'))]['spot_buy'],marker=dict(color = gray_color), name='ExchangeRate',line_width=3,showlegend=False), row=optvrank[3]+8, col=1)
    fig.update_yaxes(title_text="美元匯率", row=optvrank[3]+8, col=1)  

    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0wNy0zMCAyMzowMTo0MSIsInVzZXJfaWQiOiJqZXlhbmdqYXUiLCJpcCI6IjExNC4zNC4xMjEuMTA0In0.WDAZzKGv4Du5JilaAR7o7M1whpnGaR-vMDuSeTBXhhA"
    url = "https://api.finmindtrade.com/api/v4/data?"


    #櫃買指數
    fig.update_yaxes(title_text="櫃買指數", row=optvrank[3]+9, col=1)  
    parameter = {
    "dataset": "TaiwanStockPrice",
    "data_id": "TPEx",
    "start_date": "2022-04-02",
    "end_date": datetime.strftime(datetime.today(),'%Y-%m-%d'),
    "token": token, # 參考登入，獲取金鑰
    }
    resp = requests.get(url, params=parameter)
    dataTPEx = resp.json()
    dataTPEx = pd.DataFrame(dataTPEx["data"])

    dataTPEx.date = pd.to_datetime(dataTPEx.date)
    dataTPEx.index = dataTPEx.date


    # 計算布林帶指標
    dataTPEx['20MA'] = dataTPEx['close'].rolling(20).mean()
    dataTPEx['60MA'] = dataTPEx['close'].rolling(60).mean()
    dataTPEx['200MA'] = dataTPEx['close'].rolling(200).mean()
    dataTPEx['std'] = dataTPEx['close'].rolling(20).std()
    dataTPEx['upper_band'] = dataTPEx['20MA'] + 2 * dataTPEx['std']
    dataTPEx['lower_band'] = dataTPEx['20MA'] - 2 * dataTPEx['std']
    dataTPEx['upper_band1'] = dataTPEx['20MA'] + 1 * dataTPEx['std']
    dataTPEx['lower_band1'] = dataTPEx['20MA'] - 1 * dataTPEx['std']

    dataTPEx['IC'] = dataTPEx['close'] + 2 * dataTPEx['close'].shift(1) - dataTPEx['close'].shift(3) -dataTPEx['close'].shift(4)

    # 在k线基础上计算KDF，并将结果存储在df上面(k,d,j)
    low_list = dataTPEx['min'].rolling(9, min_periods=9).min()
    low_list.fillna(value=dataTPEx['min'].expanding().min(), inplace=True)
    high_list = dataTPEx['max'].rolling(9, min_periods=9).max()
    high_list.fillna(value=dataTPEx['max'].expanding().max(), inplace=True)
    rsv = (dataTPEx['close'] - low_list) / (high_list - low_list) * 100
    dataTPEx['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
    dataTPEx['D'] = dataTPEx['K'].ewm(com=2).mean()

    enddatemonth = enddate[~enddate["契約月份"].str.contains("W")]['最後結算日']
    dataTPEx['end_low'] = 0
    dataTPEx['end_high'] = 0
    #dataTPEx
    for datei in dataTPEx.index:
        
        month_low = dataTPEx[(dataTPEx.index >= enddatemonth[enddatemonth<datei].max())&(dataTPEx.index<=datei)]["min"].min()
        month_high = dataTPEx[(dataTPEx.index >= enddatemonth[enddatemonth<datei].max())&(dataTPEx.index<=datei)]['max'].max()
        dataTPEx.loc[datei,'end_low'] =  dataTPEx.loc[datei,'max'] - month_low
        dataTPEx.loc[datei,'end_high'] = dataTPEx.loc[datei,'min'] - month_high
        
    dataTPEx["MAX_MA"] = dataTPEx["max"] - dataTPEx["20MA"]
    dataTPEx["MIN_MA"] = dataTPEx["min"] - dataTPEx["20MA"]

    #詢問
    ds = 2
    dataTPEx['uline'] = dataTPEx['max'].rolling(ds, min_periods=1).max()
    dataTPEx['dline'] = dataTPEx['min'].rolling(ds, min_periods=1).min()

    dataTPEx["all_kk"] = 0
    barssince5 = 0
    barssince6 = 0
    dataTPEx['labelb'] = 1
    dataTPEx = dataTPEx[~dataTPEx.index.duplicated(keep='first')]
    for i in range(2,len(dataTPEx.index)):
        try:
            #(dataTPEx.loc[dataTPEx.index[i],'close'] > dataTPEx.loc[dataTPEx.index[i-1],"uline"])
            condition51 = (dataTPEx.loc[dataTPEx.index[i-1],"max"] < dataTPEx.loc[dataTPEx.index[i-2],"min"] ) and (dataTPEx.loc[dataTPEx.index[i],"min"] > dataTPEx.loc[dataTPEx.index[i-1],"max"] )
            condition52 = (dataTPEx.loc[dataTPEx.index[i-1],'close'] < dataTPEx.loc[dataTPEx.index[i-2],"min"]) and (dataTPEx.loc[dataTPEx.index[i-1],'成交金額'] > dataTPEx.loc[dataTPEx.index[i-2],'成交金額']) and (dataTPEx.loc[dataTPEx.index[i],'close']>dataTPEx.loc[dataTPEx.index[i-1],"max"] )
            condition53 = (dataTPEx.loc[dataTPEx.index[i],'close'] > dataTPEx.loc[dataTPEx.index[i-1],"uline"]) and (dataTPEx.loc[dataTPEx.index[i-1],'close'] <= dataTPEx.loc[dataTPEx.index[i-1],"uline"])

            condition61 = (dataTPEx.loc[dataTPEx.index[i-1],"min"] > dataTPEx.loc[dataTPEx.index[i-2],"max"] ) and (dataTPEx.loc[dataTPEx.index[i],"max"] < dataTPEx.loc[dataTPEx.index[i-1],"min"] )
            condition62 = (dataTPEx.loc[dataTPEx.index[i-1],'close'] > dataTPEx.loc[dataTPEx.index[i-2],"max"]) and (dataTPEx.loc[dataTPEx.index[i-1],'成交金額'] > dataTPEx.loc[dataTPEx.index[i-2],'成交金額']) and (dataTPEx.loc[dataTPEx.index[i],'close']<dataTPEx.loc[dataTPEx.index[i-1],"min"] )
            condition63 = (dataTPEx.loc[dataTPEx.index[i],'close'] < dataTPEx.loc[dataTPEx.index[i-1],"dline"]) and (dataTPEx.loc[dataTPEx.index[i-1],'close'] >= dataTPEx.loc[dataTPEx.index[i-1],"dline"])
        except:
            condition51 = True
            condition52 = True
            condition53 = True
            condition61 = True
            condition63 = True
        condition54 = condition51 or condition53 #or condition52
        condition64 = condition61 or condition63 #or condition62 

        #dataTPEx['labelb'] = np.where((dataTPEx['close']> dataTPEx['upper_band1']) , 1, np.where((dataTPEx['close']< dataTPEx['lower_band1']),-1,1))

        print(i)
        if dataTPEx.loc[dataTPEx.index[i],'close'] > dataTPEx.loc[dataTPEx.index[i],'upper_band1']:
            dataTPEx.loc[dataTPEx.index[i],'labelb'] = 1
        elif dataTPEx.loc[dataTPEx.index[i],'close'] < dataTPEx.loc[dataTPEx.index[i],'lower_band1']:
            dataTPEx.loc[dataTPEx.index[i],'labelb'] = -1
        else:
            dataTPEx.loc[dataTPEx.index[i],'labelb'] = dataTPEx.loc[dataTPEx.index[i-1],'labelb']

        if condition54 == True:
            barssince5 = 1
        else:
            barssince5 += 1

        if condition64 == True:
            barssince6 = 1
        else:
            barssince6 += 1


        if barssince5 < barssince6:
            dataTPEx.loc[dataTPEx.index[i],"all_kk"] = 1
        else:
            dataTPEx.loc[dataTPEx.index[i],"all_kk"] = -1

    dataTPEx = dataTPEx[dataTPEx.index>kbars.index[0]]

    ### 成本價及上下極限 ###

    checkb = dataTPEx["labelb"].values[0]
    bandstart = 1
    bandidx = 1
    checkidx = 0
    while bandidx < len(dataTPEx["labelb"].values):
        #checkidx = bandidx
        bandstart = bandidx-1
        checkidx = bandstart+1
        if checkidx >=len(dataTPEx["labelb"].values)-1:
            break
        while dataTPEx["labelb"].values[checkidx] == dataTPEx["labelb"].values[checkidx+1]:
            checkidx +=1
            if checkidx >=len(dataTPEx["labelb"].values)-1:
                break
        bandend = checkidx+1
        print(bandstart,bandend)
        if dataTPEx["labelb"].values[bandstart+1] == 1:
            fig.add_traces(go.Scatter(x=dataTPEx.index[bandstart:bandend], y = dataTPEx['lower_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),showlegend=False),rows=[optvrank[3]+9], cols=[1])
                
            fig.add_traces(go.Scatter(x=dataTPEx.index[bandstart:bandend], y = dataTPEx['upper_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),
                                        fill='tonexty', 
                                        fillcolor = 'rgba(256,256,0,0.4)',showlegend=False
                                        ),rows=[optvrank[3]+9], cols=[1])
        else:


            fig.add_traces(go.Scatter(x=dataTPEx.index[bandstart:bandend], y = dataTPEx['lower_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),showlegend=False),rows=[optvrank[3]+9], cols=[1])
                
            fig.add_traces(go.Scatter(x=dataTPEx.index[bandstart:bandend], y = dataTPEx['upper_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),
                                        fill='tonexty', 
                                        fillcolor = 'rgba(137, 207, 240,0.4)',showlegend=False
                                        ),rows=[optvrank[3]+9], cols=[1])
        bandidx =checkidx +1
        if bandidx >=len(dataTPEx["labelb"].values):
            break

    

    fig.add_trace(go.Scatter(x=dataTPEx.index,
                            y=dataTPEx['20MA'],
                            mode='lines',
                            line=dict(color='green'),
                            name='MA20'),row=optvrank[3]+9, col=1)
    fig.add_trace(go.Scatter(x=dataTPEx.index,
                            y=dataTPEx['200MA'],
                            mode='lines',
                            line=dict(color='blue'),
                            name='MA60'),row=optvrank[3]+9, col=1)
    fig.add_trace(go.Scatter(x=dataTPEx.index,
                            y=dataTPEx['60MA'],
                            mode='lines',
                            line=dict(color='orange'),
                            name='MA200'),row=optvrank[3]+9, col=1)

    fig.add_trace(go.Scatter(x=list(dataTPEx['IC'].index)[2:]+ICdate,
                            y=dataTPEx['IC'].values,
                            mode='lines',
                            line=dict(color='orange'),
                            name='IC操盤線'),row=optvrank[3]+9, col=1)





    ### K線圖製作 ###
    fig.add_trace(
        go.Candlestick(
            x=dataTPEx[(dataTPEx['all_kk'] == -1)&(dataTPEx['close'] >dataTPEx['open'] )].index,
            open=dataTPEx[(dataTPEx['all_kk'] == -1)&(dataTPEx['close'] >dataTPEx['open'] )]['open'],
            high=dataTPEx[(dataTPEx['all_kk'] == -1)&(dataTPEx['close'] >dataTPEx['open'] )]['max'],
            low=dataTPEx[(dataTPEx['all_kk'] == -1)&(dataTPEx['close'] >dataTPEx['open'] )]['min'],
            close=dataTPEx[(dataTPEx['all_kk'] == -1)&(dataTPEx['close'] >dataTPEx['open'] )]['close'],
            increasing_line_color=decreasing_color,
            increasing_fillcolor=no_color, #fill_increasing_color(dataTPEx.index>dataTPEx.index[50])
            decreasing_line_color=decreasing_color,
            decreasing_fillcolor=no_color,#decreasing_color,
            line=dict(width=2),
            name='OHLC',showlegend=False
        )#,
        
        ,row=optvrank[3]+9, col=1
    )


    fig.add_trace(
        go.Candlestick(
            x=dataTPEx[(dataTPEx['all_kk'] == 1)&(dataTPEx['close'] >dataTPEx['open'] )].index,
            open=dataTPEx[(dataTPEx['all_kk'] == 1)&(dataTPEx['close'] >dataTPEx['open'] )]['open'],
            high=dataTPEx[(dataTPEx['all_kk'] == 1)&(dataTPEx['close'] >dataTPEx['open'] )]['max'],
            low=dataTPEx[(dataTPEx['all_kk'] == 1)&(dataTPEx['close'] >dataTPEx['open'] )]['min'],
            close=dataTPEx[(dataTPEx['all_kk'] == 1)&(dataTPEx['close'] >dataTPEx['open'] )]['close'],
            increasing_line_color=increasing_color,
            increasing_fillcolor=no_color, #fill_increasing_color(dataTPEx.index>dataTPEx.index[50])
            decreasing_line_color=increasing_color,
            decreasing_fillcolor=no_color,#decreasing_color,
            line=dict(width=1),
            name='OHLC',showlegend=False
        )#,
        
        ,row=optvrank[3]+9, col=1
    )

    ### K線圖製作 ###
    fig.add_trace(
        go.Candlestick(
            x=dataTPEx[(dataTPEx['all_kk'] == -1)&(dataTPEx['close'] <dataTPEx['open'] )].index,
            open=dataTPEx[(dataTPEx['all_kk'] == -1)&(dataTPEx['close'] <dataTPEx['open'] )]['open'],
            high=dataTPEx[(dataTPEx['all_kk'] == -1)&(dataTPEx['close'] <dataTPEx['open'] )]['max'],
            low=dataTPEx[(dataTPEx['all_kk'] == -1)&(dataTPEx['close'] <dataTPEx['open'] )]['min'],
            close=dataTPEx[(dataTPEx['all_kk'] == -1)&(dataTPEx['close'] <dataTPEx['open'] )]['close'],
            increasing_line_color=decreasing_color,
            increasing_fillcolor=decreasing_color, #fill_increasing_color(dataTPEx.index>dataTPEx.index[50])
            decreasing_line_color=decreasing_color,
            decreasing_fillcolor=decreasing_color,#decreasing_color,
            line=dict(width=1),
            name='OHLC',showlegend=False
        )#,
        
        ,row=optvrank[3]+9, col=1
    )


    fig.add_trace(
        go.Candlestick(
            x=dataTPEx[(dataTPEx['all_kk'] == 1)&(dataTPEx['close'] <dataTPEx['open'] )].index,
            open=dataTPEx[(dataTPEx['all_kk'] == 1)&(dataTPEx['close'] <dataTPEx['open'] )]['open'],
            high=dataTPEx[(dataTPEx['all_kk'] == 1)&(dataTPEx['close'] <dataTPEx['open'] )]['max'],
            low=dataTPEx[(dataTPEx['all_kk'] == 1)&(dataTPEx['close'] <dataTPEx['open'] )]['min'],
            close=dataTPEx[(dataTPEx['all_kk'] == 1)&(dataTPEx['close'] <dataTPEx['open'] )]['close'],
            increasing_line_color=increasing_color,
            increasing_fillcolor=increasing_color, #fill_increasing_color(dataTPEx.index>dataTPEx.index[50])
            decreasing_line_color=increasing_color,
            decreasing_fillcolor=increasing_color,#decreasing_color,
            line=dict(width=1),
            name='OHLC',showlegend=False
        )#,
        
        ,row=optvrank[3]+9, col=1
    )



    # 50正2
    fig.update_yaxes(title_text="富邦台灣50正2", row=optvrank[3]+10, col=1)  
    parameter = {
        "dataset": "TaiwanStockPrice",
        "data_id": "00675L",
        "start_date": "2022-04-02",
        "end_date": datetime.strftime(datetime.today(),'%Y-%m-%d'),
        "token": token, # 參考登入，獲取金鑰
    }
    resp = requests.get(url, params=parameter)
    data50 = resp.json()
    data50 = pd.DataFrame(data50["data"]) 
    data50.date = pd.to_datetime(data50.date)
    data50.index = data50.date 

    # 計算布林帶指標
    data50['20MA'] = data50['close'].rolling(20).mean()
    data50['60MA'] = data50['close'].rolling(60).mean()
    data50['200MA'] = data50['close'].rolling(200).mean()
    data50['std'] = data50['close'].rolling(20).std()
    data50['upper_band'] = data50['20MA'] + 2 * data50['std']
    data50['lower_band'] = data50['20MA'] - 2 * data50['std']
    data50['upper_band1'] = data50['20MA'] + 1 * data50['std']
    data50['lower_band1'] = data50['20MA'] - 1 * data50['std']

    data50['IC'] = data50['close'] + 2 * data50['close'].shift(1) - data50['close'].shift(3) -data50['close'].shift(4)

    # 在k线基础上计算KDF，并将结果存储在df上面(k,d,j)
    low_list = data50['min'].rolling(9, min_periods=9).min()
    low_list.fillna(value=data50['min'].expanding().min(), inplace=True)
    high_list = data50['max'].rolling(9, min_periods=9).max()
    high_list.fillna(value=data50['max'].expanding().max(), inplace=True)
    rsv = (data50['close'] - low_list) / (high_list - low_list) * 100
    data50['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
    data50['D'] = data50['K'].ewm(com=2).mean()

    enddatemonth = enddate[~enddate["契約月份"].str.contains("W")]['最後結算日']
    data50['end_low'] = 0
    data50['end_high'] = 0
    #data50
    for datei in data50.index:
        
        month_low = data50[(data50.index >= enddatemonth[enddatemonth<datei].max())&(data50.index<=datei)]["min"].min()
        month_high = data50[(data50.index >= enddatemonth[enddatemonth<datei].max())&(data50.index<=datei)]['max'].max()
        data50.loc[datei,'end_low'] =  data50.loc[datei,'max'] - month_low
        data50.loc[datei,'end_high'] = data50.loc[datei,'min'] - month_high
        
    data50["MAX_MA"] = data50["max"] - data50["20MA"]
    data50["MIN_MA"] = data50["min"] - data50["20MA"]

    #詢問
    ds = 2
    data50['uline'] = data50['max'].rolling(ds, min_periods=1).max()
    data50['dline'] = data50['min'].rolling(ds, min_periods=1).min()

    data50["all_kk"] = 0
    barssince5 = 0
    barssince6 = 0
    data50['labelb'] = 1
    data50 = data50[~data50.index.duplicated(keep='first')]
    for i in range(2,len(data50.index)):
        try:
            #(data50.loc[data50.index[i],'close'] > data50.loc[data50.index[i-1],"uline"])
            condition51 = (data50.loc[data50.index[i-1],"max"] < data50.loc[data50.index[i-2],"min"] ) and (data50.loc[data50.index[i],"min"] > data50.loc[data50.index[i-1],"max"] )
            condition52 = (data50.loc[data50.index[i-1],'close'] < data50.loc[data50.index[i-2],"min"]) and (data50.loc[data50.index[i-1],'成交金額'] > data50.loc[data50.index[i-2],'成交金額']) and (data50.loc[data50.index[i],'close']>data50.loc[data50.index[i-1],"max"] )
            condition53 = (data50.loc[data50.index[i],'close'] > data50.loc[data50.index[i-1],"uline"]) and (data50.loc[data50.index[i-1],'close'] <= data50.loc[data50.index[i-1],"uline"])

            condition61 = (data50.loc[data50.index[i-1],"min"] > data50.loc[data50.index[i-2],"max"] ) and (data50.loc[data50.index[i],"max"] < data50.loc[data50.index[i-1],"min"] )
            condition62 = (data50.loc[data50.index[i-1],'close'] > data50.loc[data50.index[i-2],"max"]) and (data50.loc[data50.index[i-1],'成交金額'] > data50.loc[data50.index[i-2],'成交金額']) and (data50.loc[data50.index[i],'close']<data50.loc[data50.index[i-1],"min"] )
            condition63 = (data50.loc[data50.index[i],'close'] < data50.loc[data50.index[i-1],"dline"]) and (data50.loc[data50.index[i-1],'close'] >= data50.loc[data50.index[i-1],"dline"])
        except:
            condition51 = True
            condition52 = True
            condition53 = True
            condition61 = True
            condition63 = True
        condition54 = condition51 or condition53 #or condition52
        condition64 = condition61 or condition63 #or condition62 

        #data50['labelb'] = np.where((data50['close']> data50['upper_band1']) , 1, np.where((data50['close']< data50['lower_band1']),-1,1))

        print(i)
        if data50.loc[data50.index[i],'close'] > data50.loc[data50.index[i],'upper_band1']:
            data50.loc[data50.index[i],'labelb'] = 1
        elif data50.loc[data50.index[i],'close'] < data50.loc[data50.index[i],'lower_band1']:
            data50.loc[data50.index[i],'labelb'] = -1
        else:
            data50.loc[data50.index[i],'labelb'] = data50.loc[data50.index[i-1],'labelb']

        if condition54 == True:
            barssince5 = 1
        else:
            barssince5 += 1

        if condition64 == True:
            barssince6 = 1
        else:
            barssince6 += 1


        if barssince5 < barssince6:
            data50.loc[data50.index[i],"all_kk"] = 1
        else:
            data50.loc[data50.index[i],"all_kk"] = -1

    data50 = data50[data50.index>kbars.index[0]]

    ### 成本價及上下極限 ###

    checkb = data50["labelb"].values[0]
    bandstart = 1
    bandidx = 1
    checkidx = 0
    while bandidx < len(data50["labelb"].values):
        #checkidx = bandidx
        bandstart = bandidx-1
        checkidx = bandstart+1
        if checkidx >=len(data50["labelb"].values)-1:
            break
        while data50["labelb"].values[checkidx] == data50["labelb"].values[checkidx+1]:
            checkidx +=1
            if checkidx >=len(data50["labelb"].values)-1:
                break
        bandend = checkidx+1
        print(bandstart,bandend)
        if data50["labelb"].values[bandstart+1] == 1:
            fig.add_traces(go.Scatter(x=data50.index[bandstart:bandend], y = data50['lower_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),showlegend=False),rows=[optvrank[3]+10], cols=[1])
                
            fig.add_traces(go.Scatter(x=data50.index[bandstart:bandend], y = data50['upper_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),
                                        fill='tonexty', 
                                        fillcolor = 'rgba(256,256,0,0.4)',showlegend=False
                                        ),rows=[optvrank[3]+10], cols=[1])
        else:


            fig.add_traces(go.Scatter(x=data50.index[bandstart:bandend], y = data50['lower_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),showlegend=False),rows=[optvrank[3]+10], cols=[1])
                
            fig.add_traces(go.Scatter(x=data50.index[bandstart:bandend], y = data50['upper_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),
                                        fill='tonexty', 
                                        fillcolor = 'rgba(137, 207, 240,0.4)',showlegend=False
                                        ),rows=[optvrank[3]+10], cols=[1])
        bandidx =checkidx +1
        if bandidx >=len(data50["labelb"].values):
            break

    

    fig.add_trace(go.Scatter(x=data50.index,
                            y=data50['20MA'],
                            mode='lines',
                            line=dict(color='green'),
                            name='MA20'),row=optvrank[3]+10, col=1)
    fig.add_trace(go.Scatter(x=data50.index,
                            y=data50['200MA'],
                            mode='lines',
                            line=dict(color='blue'),
                            name='MA60'),row=optvrank[3]+10, col=1)
    fig.add_trace(go.Scatter(x=data50.index,
                            y=data50['60MA'],
                            mode='lines',
                            line=dict(color='orange'),
                            name='MA200'),row=optvrank[3]+10, col=1)

    fig.add_trace(go.Scatter(x=list(data50['IC'].index)[2:]+ICdate,
                            y=data50['IC'].values,
                            mode='lines',
                            line=dict(color='orange'),
                            name='IC操盤線'),row=optvrank[3]+10, col=1)





    ### K線圖製作 ###
    fig.add_trace(
        go.Candlestick(
            x=data50[(data50['all_kk'] == -1)&(data50['close'] >data50['open'] )].index,
            open=data50[(data50['all_kk'] == -1)&(data50['close'] >data50['open'] )]['open'],
            high=data50[(data50['all_kk'] == -1)&(data50['close'] >data50['open'] )]['max'],
            low=data50[(data50['all_kk'] == -1)&(data50['close'] >data50['open'] )]['min'],
            close=data50[(data50['all_kk'] == -1)&(data50['close'] >data50['open'] )]['close'],
            increasing_line_color=decreasing_color,
            increasing_fillcolor=no_color, #fill_increasing_color(data50.index>data50.index[50])
            decreasing_line_color=decreasing_color,
            decreasing_fillcolor=no_color,#decreasing_color,
            line=dict(width=2),
            name='OHLC',showlegend=False
        )#,
        
        ,row=optvrank[3]+10, col=1
    )


    fig.add_trace(
        go.Candlestick(
            x=data50[(data50['all_kk'] == 1)&(data50['close'] >data50['open'] )].index,
            open=data50[(data50['all_kk'] == 1)&(data50['close'] >data50['open'] )]['open'],
            high=data50[(data50['all_kk'] == 1)&(data50['close'] >data50['open'] )]['max'],
            low=data50[(data50['all_kk'] == 1)&(data50['close'] >data50['open'] )]['min'],
            close=data50[(data50['all_kk'] == 1)&(data50['close'] >data50['open'] )]['close'],
            increasing_line_color=increasing_color,
            increasing_fillcolor=no_color, #fill_increasing_color(data50.index>data50.index[50])
            decreasing_line_color=increasing_color,
            decreasing_fillcolor=no_color,#decreasing_color,
            line=dict(width=1),
            name='OHLC',showlegend=False
        )#,
        
        ,row=optvrank[3]+10, col=1
    )

    ### K線圖製作 ###
    fig.add_trace(
        go.Candlestick(
            x=data50[(data50['all_kk'] == -1)&(data50['close'] <data50['open'] )].index,
            open=data50[(data50['all_kk'] == -1)&(data50['close'] <data50['open'] )]['open'],
            high=data50[(data50['all_kk'] == -1)&(data50['close'] <data50['open'] )]['max'],
            low=data50[(data50['all_kk'] == -1)&(data50['close'] <data50['open'] )]['min'],
            close=data50[(data50['all_kk'] == -1)&(data50['close'] <data50['open'] )]['close'],
            increasing_line_color=decreasing_color,
            increasing_fillcolor=decreasing_color, #fill_increasing_color(data50.index>data50.index[50])
            decreasing_line_color=decreasing_color,
            decreasing_fillcolor=decreasing_color,#decreasing_color,
            line=dict(width=1),
            name='OHLC',showlegend=False
        )#,
        
        ,row=optvrank[3]+10, col=1
    )


    fig.add_trace(
        go.Candlestick(
            x=data50[(data50['all_kk'] == 1)&(data50['close'] <data50['open'] )].index,
            open=data50[(data50['all_kk'] == 1)&(data50['close'] <data50['open'] )]['open'],
            high=data50[(data50['all_kk'] == 1)&(data50['close'] <data50['open'] )]['max'],
            low=data50[(data50['all_kk'] == 1)&(data50['close'] <data50['open'] )]['min'],
            close=data50[(data50['all_kk'] == 1)&(data50['close'] <data50['open'] )]['close'],
            increasing_line_color=increasing_color,
            increasing_fillcolor=increasing_color, #fill_increasing_color(data50.index>data50.index[50])
            decreasing_line_color=increasing_color,
            decreasing_fillcolor=increasing_color,#decreasing_color,
            line=dict(width=1),
            name='OHLC',showlegend=False
        )#,
        
        ,row=optvrank[3]+10, col=1
    )


        
    ### 圖表設定 ###
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_annotations(font_size=12)

    fig.update_layout(
        title=u'大盤指數技術分析圖',
        #title_x=0.5,
        #title_y=0.93,
        hovermode='x unified', 
        showlegend=True,
        height=350 + 150* rowcount,
        width = 1000,
        hoverlabel_namelength=-1,
        xaxis2=dict(showgrid=False),
        yaxis2=dict(showgrid=False,tickformat = ",.0f",range=[kbars['最低指數'].min() - 200, kbars['最高指數'].max() + 200]),
        yaxis = dict(showgrid=False,showticklabels=False,range=[0, 90*10**10]),
        #yaxis = dict(range=[kbars['min'].min() - 2000, kbars['最高指數'].max() + 500]),
        dragmode = 'drawline',
        hoverlabel=dict(align='left'),
        legend_traceorder="reversed",
        
    )

    fig.update_traces(xaxis='x1',hoverlabel=dict(align='left'))

    # 隱藏周末與市場休市日期 ### 導入台灣的休市資料
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=['sat', 'mon']), # hide weekends, eg. hide sat to before mon
            dict(values=[str(holiday) for holiday in holidf[~(holidf["說明"].str.contains('開始交易') | holidf["說明"].str.contains('最後交易'))]["日期"].values]+['2023-08-03'])
        ]
    )


    #fig.update_traces(hoverlabel=dict(align='left'))

    st.plotly_chart(fig)
    

with tab2:
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
    
    #enddatemonth = enddate[~enddate["契約月份"].str.contains("W")]['最後結算日'].values[0]

    data1 = data[(data.date < enddate[~enddate["契約月份"].str.contains("W")]['最後結算日'].values[0]) & (data.contract_date == contract1)]
    data2 = data[(data.date >= enddate[~enddate["契約月份"].str.contains("W")]['最後結算日'].values[0]) & (data.contract_date == contract2)]
    df = pd.concat([data1,data2])

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


    #  設定左右子圖
    fig2 = make_subplots(
        rows = 1, 
        cols = 10, 
        horizontal_spacing = 0.02, 
        subplot_titles = titlelist
    )

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

    ## 圖五
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

    # 設定圖的標題跟長寬
    fig2.update_annotations(font_size=12)
    fig2.update_layout(title_text = "臺指選擇權 未沖銷契約數 支撐壓力圖（月）", 
                    width = 1200, 
                    height = 650)

    #fig.show()
    st.plotly_chart(fig2)


with tab3:
    kbars