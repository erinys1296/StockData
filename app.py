import time

import streamlit as st

import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import numpy as np

import plotly.graph_objects as go
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

enddate = pd.read_sql("select * from end_date", connection, parse_dates=['最後結算日'])

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

enddatemonth = enddate.groupby(enddate['最後結算日'].dt.month)['最後結算日'].max()
kbars['end_low'] = 0
kbars['end_high'] = 0
for datei in kbars.index:
    
    low = kbars[(kbars.index >= enddatemonth[enddatemonth<datei].max())&(kbars.index<=datei)]["最低指數"].min()
    high = kbars[(kbars.index >= enddatemonth[enddatemonth<datei].max())&(kbars.index<=datei)]['最高指數'].max()
    kbars.loc[datei,'end_low'] = kbars.loc[datei,'收盤指數'] - low
    kbars.loc[datei,'end_high'] = high - kbars.loc[datei,'收盤指數']
    
kbars["MAX_MA"] = kbars["最高指數"] - kbars["MA"]
kbars["MIN_MA"] = kbars["最低指數"] - kbars["MA"]

kbars = kbars.dropna()

holidf = pd.read_sql("select * from holiday", connection)


st.sidebar.header('Setting')

st.sidebar.write('主圖選擇')
option_1b = st.sidebar.checkbox('自營商上下極限', value = True)
option_1c = st.sidebar.checkbox('外資上下極限', value = True)
option_1d = st.sidebar.checkbox('上下極限', value = True)
options_main = [option_1b , option_1c , option_1d]

st.sidebar.write('附圖選擇')

option_2a = st.sidebar.checkbox('成交量', value = True)
option_2b = st.sidebar.checkbox('KD指標', value = True)
option_2c = st.sidebar.checkbox('開盤賣張張數', value = True)
option_2d = st.sidebar.checkbox('價平和', value = True)
option_2e = st.sidebar.checkbox('20MA_GAP', value = True)
option_2f = st.sidebar.checkbox('月結算日差', value = True)
options_vice = [option_2a , option_2b , option_2c , option_2d, option_2e , option_2f]
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
subtitle_all = ['OHLC', 'Volumn', 'KD', 'OrderVolumn','價平和','20MA_GAP','月結算日差']
subtitle =['OHLC']
for i in range(1,7):
    if optvrank[i-1] != 0:
        subtitle.append(subtitle_all[i])    



#subtitle
enddate = pd.read_sql("select * from end_date", connection, parse_dates=['最後結算日'])

st.title('選擇權')
rowh = [0.5, 0.5/6, 0.5/6, 0.5/6, 0.5/6, 0.5/6, 0.5/6]
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

red_color = 'rgb(239, 83, 80)'
green_color = 'rgb(38, 166, 154)'

no_color = 'rgb(256, 256, 256)'

def fill_increasing_color(label):
    if label == True:
        return red_color
    else:
        return no_color
    
def fill_decreasing_color(label):
    if label == True:
        return green_color
    else:
        return no_color






### 成本價及上下極限 ###
fig.add_trace(go.Scatter(x=kbars.index,
                 y=kbars['外資成本'],
                 mode='lines',
                 line=dict(color='yellow'),
                 name='外資成本'),row=1, col=1)


#自營商上下極限
fig.add_scatter(x=np.concatenate([kbars.index,kbars.index[::-1]]), y=np.concatenate([kbars['自營商下極限'], kbars['自營商上極限'][::-1]]), 
                fill='toself',fillcolor= 'rgba(0,0,256,0.1)', line_width=0,name='自營商上下極限',row=1, col=1 )

#fig.add_trace(go.Scatter(x=kbars.index,
#                 y=kbars['自營商上極限'],
#                 mode='lines',
#                 line=dict(color='gray'),
#                 name='自營商上極限'))

#fig.add_trace(go.Scatter(x=kbars.index,
#                 y=kbars['自營商下極限'],
#                 mode='lines',
#                 line=dict(color='red'),
#                 name='自營商下極限'))


#外資上下極限
fig.add_scatter(x=np.concatenate([kbars.index,kbars.index[::-1]]), y=np.concatenate([kbars['外資下極限'], kbars['外資上極限'][::-1]]), 
                fill='toself',fillcolor= 'rgba(256,0,0,0.1)', line_width=0,name='外資上下極限',row=1, col=1)

#上下極限
fig.add_scatter(x=np.concatenate([kbars.index,kbars.index[::-1]]), y=np.concatenate([kbars['lower_band'], kbars['upper_band'][::-1]]), 
                fill='toself',fillcolor= 'rgba(0,256,0,0.1)', line_width=0,name='上下極限',row=1, col=1)


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
                         name='MA'),row=1, col=1)

fig.add_trace(go.Scatter(x=list(kbars['IC'].index)[2:]+[kbars['IC'].index[-1] + timedelta(days = 1),kbars['IC'].index[-1] + timedelta(days = 2)],
                         y=kbars['IC'].shift(2).values,
                         mode='lines',
                         line=dict(color='orange'),
                         name='IC操盤線'),row=1, col=1)

for i in enddate.groupby(enddate['最後結算日'].dt.month)['最後結算日'].max():
    if i > kbars.index[0] and i!=enddate.groupby(enddate['最後結算日'].dt.month)['最後結算日'].max()[5]:
        fig.add_vline(x=i, line_width=1,  line_color="green")

#enddate['最後結算日'].values
for i in list(enddate['最後結算日'].values):
    if i > kbars.index[0] and i!=enddate.groupby(enddate['最後結算日'].dt.month)['最後結算日'].max()[5]:
#        fig.add_vline(x=i, line_width=1,  line_color="green")
        fig.add_vline(x=i, line_width=1,line_dash="dash", line_color="blue")#, line_dash="dash"
#fig.add_hrect(y0=0.9, y1=2.6, line_width=0, fillcolor="red", opacity=0.2)



### K線圖製作 ###
fig.add_trace(
    go.Candlestick(
        x=kbars.index,
        open=kbars['開盤指數'],
        high=kbars['最高指數'],
        low=kbars['最低指數'],
        close=kbars['收盤指數'],
        increasing_line_color=increasing_color,
        increasing_fillcolor=no_color, #fill_increasing_color(kbars.index>kbars.index[50])
        decreasing_line_color=decreasing_color,
        decreasing_fillcolor=decreasing_color,
        line=dict(width=1),
        name='OHLC'
    )#,
    #row=1, col=1
)


if optvrank[0] != 0:
    ### 成交量圖製作 ###
    volume_colors = [increasing_color if kbars['收盤指數'][i] > kbars['收盤指數'][i-1] else decreasing_color for i in range(len(kbars['收盤指數']))]
    volume_colors[0] = decreasing_color

    fig.add_trace(go.Bar(x=kbars.index, y=kbars['成交金額'], name='Volume', marker=dict(color=volume_colors),showlegend=False), row=optvrank[0], col=1)


### KD線 ###
if optvrank[1] != 0:
    fig.add_trace(go.Scatter(x=kbars.index, y=kbars['K'], name='K', line=dict(width=1, color='rgb(41, 98, 255)'),showlegend=False), row=optvrank[1], col=1)
    fig.add_trace(go.Scatter(x=kbars.index, y=kbars['D'], name='D', line=dict(width=1, color='rgb(255, 109, 0)'),showlegend=False), row=optvrank[1], col=1)

## 委賣數量 ##
if optvrank[2] != 0:
    #volume_colors = [increasing_color if kbars['九點累積委託賣出數量	'][i] > kbars['收盤指數'][i-1] else decreasing_color for i in range(len(kbars['收盤指數']))]
    fig.add_trace(go.Bar(x=kbars.index, y=kbars['九點累積委託賣出數量'], name='Volume',showlegend=False), row=optvrank[2], col=1)

## 價平和
if optvrank[3] != 0:
    fig.add_trace(go.Bar(x=kbars.index, y=kbars['價平和'], name='PCsum',showlegend=False), row=optvrank[3], col=1)

## MA差
if optvrank[4] != 0:
    fig.add_trace(go.Bar(x=kbars.index, y=kbars['MAX_MA'], name='MAX_MA',showlegend=False), row=optvrank[4], col=1)
    fig.add_trace(go.Bar(x=kbars.index, y=kbars['MIN_MA'], name='MIN_MA',showlegend=False), row=optvrank[4], col=1)

## 結算差
if optvrank[5] != 0:
    fig.add_trace(go.Bar(x=kbars.index, y=kbars['end_high'], name='MAX_END',showlegend=False), row=optvrank[5], col=1)
    fig.add_trace(go.Bar(x=kbars.index, y=kbars['end_low'], name='MIN_END',showlegend=False), row=optvrank[5], col=1)

### 圖表設定 ###
fig.update(layout_xaxis_rangeslider_visible=False)
fig.update_annotations(font_size=12)

fig.update_layout(
    title=u'大盤指數技術分析圖',
    title_x=0.5,
    #title_y=0.93,
    hovermode='x unified', 
    showlegend=True,
    height=1000,
    hoverlabel_namelength=-1,
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False)
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
