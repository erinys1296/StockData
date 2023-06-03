import time

import streamlit as st

import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import numpy as np

import plotly.graph_objects as go
import pandas_market_calendars as mcal
from plotly.subplots import make_subplots


connection = sqlite3.connect('主圖資料.db')
taiex = pd.read_sql("select distinct * from taiex", connection, parse_dates=['日期'], index_col=['日期'])
taiex_vol = pd.read_sql("select distinct * from taiex_vol", connection, parse_dates=['日期'], index_col=['日期'])
cost_df = pd.read_sql("select distinct Date as [日期], Cost as [外資成本] from cost", connection, parse_dates=['日期'], index_col=['日期'])
cost_df["外資成本"] = cost_df["外資成本"].astype('int')
limit_df = pd.read_sql("select distinct * from [limit]", connection, parse_dates=['日期'], index_col=['日期'])

inves_limit = limit_df[limit_df["身份別"] == "外資"][['上極限', '下極限']]
dealer_limit = limit_df[limit_df["身份別"] == "自營商"][['上極限', '下極限']]

inves_limit.columns = ["外資上極限","外資下極限"]
dealer_limit.columns = ["自營商上極限","自營商下極限"]

kbars = taiex.join(taiex_vol).join(cost_df).join(dealer_limit).join(inves_limit)

ordervolumn = pd.read_sql("select distinct * from ordervolumn", connection, parse_dates=['日期'], index_col=['日期'])
putcallsum = pd.read_sql("select distinct * from putcallsum", connection, parse_dates=['日期'], index_col=['日期'])
kbars = kbars.join(ordervolumn).join(putcallsum)

# 計算布林帶指標
kbars['MA'] = kbars['收盤指數'].rolling(20).mean()
kbars['std'] = kbars['收盤指數'].rolling(20).std()
kbars['upper_band'] = kbars['MA'] + 2 * kbars['std']
kbars['lower_band'] = kbars['MA'] - 2 * kbars['std']

kbars['IC'] = kbars['收盤指數'] + 2 * kbars['收盤指數'].shift(1) - kbars['收盤指數'].shift(3) -kbars['收盤指數'].shift(4)

# 在k线基础上计算KDF，并将结果存储在df上面(k,d,j)
low_list = kbars['最低指數'].rolling(9, min_periods=9).min()
low_list.fillna(value=kbars['最低指數'].expanding().min(), inplace=True)
high_list = kbars['最高指數'].rolling(9, min_periods=9).max()
high_list.fillna(value=kbars['最高指數'].expanding().max(), inplace=True)
rsv = (kbars['收盤指數'] - low_list) / (high_list - low_list) * 100
kbars['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
kbars['D'] = kbars['K'].ewm(com=2).mean()


kbars = kbars.dropna()

holidf = pd.read_sql("select * from holiday", connection)


st.sidebar.header('Setting')

st.sidebar.write('主圖選擇')
option_1a = st.sidebar.checkbox('外資成本', value = True)
option_1b = st.sidebar.checkbox('自營商上下極限', value = True)
option_1c = st.sidebar.checkbox('外資上下極限', value = True)
option_1d = st.sidebar.checkbox('IC線', value = True)
options_main = [option_1a , option_1b , option_1c , option_1d]

st.sidebar.write('附圖選擇')

option_2a = st.sidebar.checkbox('成交量', value = True)
option_2b = st.sidebar.checkbox('KD指標', value = True)
option_2c = st.sidebar.checkbox('開盤賣張張數', value = True)
option_2d = st.sidebar.checkbox('價平和', value = True)
options_vice = [option_2a , option_2b , option_2c , option_2d]
#options_vice[options_vice == True]
#options_vice[0] == True
optvn = 0
optvrank = []
for opv in options_vice:
    if opv == True:
        optvn += 1
        optvrank.append(optvn+1)
    else:
        optvrank.append(0)
subtitle_all = ['OHLC', 'Volumn', 'KD', 'OrderVolumn','價平和']
subtitle =['OHLC']
for i in range(1,5):
    if optvrank[i-1] != 0:
        subtitle.append(subtitle_all[i])    


#subtitle

st.title('選擇權')
rowh = [0.6, 0.1,0.1, 0.1, 0.1]
fig = make_subplots(
    rows=optvn + 1, cols=1,
    shared_xaxes=True, 
    vertical_spacing=0.06,
    row_heights= rowh[:optvn+1],
    shared_yaxes=False,
    subplot_titles=subtitle
)

increasing_color = 'rgb(239, 83, 80)'
decreasing_color = 'rgb(38, 166, 154)'





### 成本價及上下極限 ###
fig.add_trace(go.Scatter(x=kbars.index,
                 y=kbars['外資成本'],
                 mode='lines',
                 line=dict(color='yellow'),
                 name='外資成本'))

fig.add_trace(go.Scatter(x=kbars.index,
                 y=kbars['自營商上極限'],
                 mode='lines',
                 line=dict(color='gray'),
                 name='自營商上極限'))

fig.add_trace(go.Scatter(x=kbars.index,
                 y=kbars['自營商下極限'],
                 mode='lines',
                 line=dict(color='red'),
                 name='自營商下極限'))

fig.add_trace(go.Scatter(x=kbars.index,
                 y=kbars['外資上極限'],
                 mode='lines',
                 line=dict(color='#9467bd'),
                 name='外資上極限'))

fig.add_trace(go.Scatter(x=kbars.index,
                 y=kbars['外資下極限'],
                 mode='lines',
                 line=dict(color='#17becf'),
                 name='外資下極限'))

fig.add_trace(go.Scatter(x=kbars.index,
                         y=kbars['MA'],
                         mode='lines',
                         line=dict(color='green'),
                         name='MA'))

fig.add_trace(go.Scatter(x=list(kbars['IC'].index)[2:]+[kbars['IC'].index[-1] + timedelta(days = 1),kbars['IC'].index[-1] + timedelta(days = 2)],
                         y=kbars['IC'].shift(2).values,
                         mode='lines',
                         line=dict(color='orange'),
                         name='IC操盤線'))

### K線圖製作 ###
fig.add_trace(
    go.Candlestick(
        x=kbars.index,
        open=kbars['開盤指數'],
        high=kbars['最高指數'],
        low=kbars['最低指數'],
        close=kbars['收盤指數'],
        increasing_line_color=increasing_color,
        increasing_fillcolor=increasing_color,
        decreasing_line_color=decreasing_color,
        decreasing_fillcolor=decreasing_color,
        line=dict(width=1),
        name='OHLC'
    ),
    row=1, col=1
)

if optvrank[0] != 0:
    ### 成交量圖製作 ###
    volume_colors = [increasing_color if kbars['收盤指數'][i] > kbars['收盤指數'][i-1] else decreasing_color for i in range(len(kbars['收盤指數']))]
    volume_colors[0] = decreasing_color

    fig.add_trace(go.Bar(x=kbars.index, y=kbars['成交金額'], name='Volume', marker=dict(color=volume_colors)), row=optvrank[0], col=1)


### KD線 ###
if optvrank[1] != 0:
    fig.add_trace(go.Scatter(x=kbars.index, y=kbars['K'], name='K', line=dict(width=1, color='rgb(41, 98, 255)')), row=optvrank[1], col=1)
    fig.add_trace(go.Scatter(x=kbars.index, y=kbars['D'], name='D', line=dict(width=1, color='rgb(255, 109, 0)')), row=optvrank[1], col=1)

## 委賣數量 ##
if optvrank[2] != 0:
    #volume_colors = [increasing_color if kbars['九點累積委託賣出數量	'][i] > kbars['收盤指數'][i-1] else decreasing_color for i in range(len(kbars['收盤指數']))]
    fig.add_trace(go.Bar(x=kbars.index, y=kbars['九點累積委託賣出數量'], name='Volume'), row=optvrank[2], col=1)

## 價平和
if optvrank[3] != 0:
    fig.add_trace(go.Bar(x=kbars.index, y=kbars['價平和'], name='PCsum'), row=optvrank[3], col=1)

### 圖表設定 ###
fig.update(layout_xaxis_rangeslider_visible=False)
fig.update_annotations(font_size=12)

fig.update_layout(
    title=u'大盤指數技術分析圖',
    title_x=0.5,
    title_y=0.93,
    hovermode='x unified', 
    showlegend=True,
    height=600,
    hoverlabel_namelength=-1
)

# 隱藏周末與市場休市日期 ### 導入台灣的休市資料
fig.update_xaxes(
     rangebreaks=[
         dict(bounds=['sat', 'mon']), # hide weekends, eg. hide sat to before mon
         dict(values=[str(holiday) for holiday in holidf[~(holidf["說明"].str.contains('開始交易') | holidf["說明"].str.contains('最後交易'))]["日期"].values])
     ]
 )

st.plotly_chart(fig)

st.write('Data')
kbars