import requests
import pandas as pd
from datetime import datetime, timedelta,time

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3

connection = sqlite3.connect('主圖資料.sqlite3')
#connectionfuture = sqlite3.connect('FutureData.sqlite3')

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0wNy0zMCAyMzowMTo0MSIsInVzZXJfaWQiOiJqZXlhbmdqYXUiLCJpcCI6IjExNC4zNC4xMjEuMTA0In0.WDAZzKGv4Du5JilaAR7o7M1whpnGaR-vMDuSeTBXhhA"
url = "https://api.finmindtrade.com/api/v4/data?"

holidf = pd.read_sql("select * from holiday", connection)

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


def get_stock_data(stocknumber):
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0wNy0zMCAyMzowMTo0MSIsInVzZXJfaWQiOiJqZXlhbmdqYXUiLCJpcCI6IjExNC4zNC4xMjEuMTA0In0.WDAZzKGv4Du5JilaAR7o7M1whpnGaR-vMDuSeTBXhhA"
    url = "https://api.finmindtrade.com/api/v4/data?"
    parameter = {
    "dataset": "TaiwanStockPrice",
    "data_id": stocknumber,
    "start_date": "2021-04-02",
    "end_date": datetime.strftime(datetime.today(),'%Y-%m-%d'),
    "token": token, # 參考登入，獲取金鑰
    }
    resp = requests.get(url, params=parameter)
    data = resp.json()
    data = pd.DataFrame(data["data"]) 
    data.date = pd.to_datetime(data.date)
    data.index = data.date 
    stockdata = data.copy()

    # 計算布林帶指標
    stockdata['20MA'] = stockdata['close'].rolling(20).mean()
    stockdata['60MA'] = stockdata['close'].rolling(60).mean()
    stockdata['200MA'] = stockdata['close'].rolling(200).mean()
    stockdata['std'] = stockdata['close'].rolling(20).std()
    stockdata['upper_band'] = stockdata['20MA'] + 2 * stockdata['std']
    stockdata['lower_band'] = stockdata['20MA'] - 2 * stockdata['std']
    stockdata['upper_band1'] = stockdata['20MA'] + 1 * stockdata['std']
    stockdata['lower_band1'] = stockdata['20MA'] - 1 * stockdata['std']

    stockdata['IC'] = stockdata['close'] + 2 * stockdata['close'].shift(1) - stockdata['close'].shift(3) -stockdata['close'].shift(4)

    # 在k线基础上计算KDF，并将结果存储在df上面(k,d,j)
    low_list = stockdata['min'].rolling(9, min_periods=9).min()
    low_list.fillna(value=stockdata['min'].expanding().min(), inplace=True)
    high_list = stockdata['max'].rolling(9, min_periods=9).max()
    high_list.fillna(value=stockdata['max'].expanding().max(), inplace=True)
    rsv = (stockdata['close'] - low_list) / (high_list - low_list) * 100
    stockdata['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
    stockdata['D'] = stockdata['K'].ewm(com=2).mean()



    #詢問
    ds = 2
    stockdata['uline'] = stockdata['max'].rolling(ds, min_periods=1).max()
    stockdata['dline'] = stockdata['min'].rolling(ds, min_periods=1).min()

    stockdata["all_kk"] = 0
    barssince5 = 0
    barssince6 = 0
    stockdata['labelb'] = 1
    stockdata = stockdata[~stockdata.index.duplicated(keep='first')]
    for i in range(2,len(stockdata.index)):
        try:
            #(stockdata.loc[stockdata.index[i],'close'] > stockdata.loc[stockdata.index[i-1],"uline"])
            condition51 = (stockdata.loc[stockdata.index[i-1],"max"] < stockdata.loc[stockdata.index[i-2],"min"] ) and (stockdata.loc[stockdata.index[i],"min"] > stockdata.loc[stockdata.index[i-1],"max"] )
            #condition52 = (stockdata.loc[stockdata.index[i-1],'close'] < stockdata.loc[stockdata.index[i-2],"min"]) and (stockdata.loc[stockdata.index[i-1],'成交金額'] > stockdata.loc[stockdata.index[i-2],'成交金額']) and (stockdata.loc[stockdata.index[i],'close']>stockdata.loc[stockdata.index[i-1],"max"] )
            condition53 = (stockdata.loc[stockdata.index[i],'close'] > stockdata.loc[stockdata.index[i-1],"uline"]) and (stockdata.loc[stockdata.index[i-1],'close'] <= stockdata.loc[stockdata.index[i-1],"uline"])

            condition61 = (stockdata.loc[stockdata.index[i-1],"min"] > stockdata.loc[stockdata.index[i-2],"max"] ) and (stockdata.loc[stockdata.index[i],"max"] < stockdata.loc[stockdata.index[i-1],"min"] )
            #condition62 = (stockdata.loc[stockdata.index[i-1],'close'] > stockdata.loc[stockdata.index[i-2],"max"]) and (stockdata.loc[stockdata.index[i-1],'成交金額'] > stockdata.loc[stockdata.index[i-2],'成交金額']) and (stockdata.loc[stockdata.index[i],'close']<stockdata.loc[stockdata.index[i-1],"min"] )
            condition63 = (stockdata.loc[stockdata.index[i],'close'] < stockdata.loc[stockdata.index[i-1],"dline"]) and (stockdata.loc[stockdata.index[i-1],'close'] >= stockdata.loc[stockdata.index[i-1],"dline"])
        except:
            condition51 = True
            condition52 = True
            condition53 = True
            condition61 = True
            condition63 = True
        condition54 = condition51 or condition53 #or condition52
        condition64 = condition61 or condition63 #or condition62 

        #stockdata['labelb'] = np.where((stockdata['close']> stockdata['upper_band1']) , 1, np.where((stockdata['close']< stockdata['lower_band1']),-1,1))

        #print(i)
        if stockdata.loc[stockdata.index[i],'close'] > stockdata.loc[stockdata.index[i],'upper_band1']:
            stockdata.loc[stockdata.index[i],'labelb'] = 1
        elif stockdata.loc[stockdata.index[i],'close'] < stockdata.loc[stockdata.index[i],'lower_band1']:
            stockdata.loc[stockdata.index[i],'labelb'] = -1
        else:
            stockdata.loc[stockdata.index[i],'labelb'] = stockdata.loc[stockdata.index[i-1],'labelb']

        if condition54 == True:
            barssince5 = 1
        else:
            barssince5 += 1

        if condition64 == True:
            barssince6 = 1
        else:
            barssince6 += 1


        if barssince5 < barssince6:
            stockdata.loc[stockdata.index[i],"all_kk"] = 1
        else:
            stockdata.loc[stockdata.index[i],"all_kk"] = -1

    try:
            
        ICdate = []
        datechecki = 1
        #(kbars['IC'].index[-1] + timedelta(days = 1)).weekday() == 5
        while (stockdata['IC'].index[-1] + timedelta(days = datechecki)).weekday() in [5,6] or (stockdata['IC'].index[-1] + timedelta(days = datechecki)) in pd.to_datetime(holidf["日期"]).values:
            datechecki +=1
        ICdate.append((stockdata['IC'].index[-1] + timedelta(days = datechecki)))
        datechecki +=1
        while (stockdata['IC'].index[-1] + timedelta(days = datechecki)).weekday() in [5,6] or (stockdata['IC'].index[-1] + timedelta(days = datechecki)) in pd.to_datetime(holidf["日期"]).values:
            datechecki +=1
        ICdate.append((stockdata['IC'].index[-1] + timedelta(days = datechecki)))
    except:
        ICdate = []

    stockdata = stockdata[stockdata.index>stockdata.index[-120]]
            
    return stockdata,ICdate




def plot_stockdata(fig,holidf,stockdata,r,c,ICdate):
    


    


    

    fig.add_trace(go.Scatter(x=stockdata.index,
                            y=stockdata['20MA'],
                            mode='lines',
                            line=dict(color='green'),
                            name='MA20'),row=r, col=c, secondary_y= True)
    fig.add_trace(go.Scatter(x=stockdata.index,
                            y=stockdata['200MA'],
                            mode='lines',
                            line=dict(color='blue'),
                            name='MA200'),row=r, col=c, secondary_y= True)
    fig.add_trace(go.Scatter(x=stockdata.index,
                            y=stockdata['60MA'],
                            mode='lines',
                            line=dict(color='orange'),
                            name='MA60'),row=r, col=c, secondary_y= True)

    fig.add_trace(go.Scatter(x=list(stockdata['IC'].index)[1:]+ICdate,
                            y=stockdata['IC'].values,
                            mode='lines',
                            line=dict(color='orange'),
                            name='IC操盤線'),row=r, col=c, secondary_y= True)





    ### 成本價及上下極限 ###

    checkb = stockdata["labelb"].values[0]
    bandstart = 1
    bandidx = 1
    checkidx = 0
    while bandidx < len(stockdata["labelb"].values):
        #checkidx = bandidx
        bandstart = bandidx-1
        checkidx = bandstart+1
        if checkidx >=len(stockdata["labelb"].values)-1:
            break
        while stockdata["labelb"].values[checkidx] == stockdata["labelb"].values[checkidx+1]:
            checkidx +=1
            if checkidx >=len(stockdata["labelb"].values)-1:
                break
        bandend = checkidx+1
        #print(bandstart,bandend)
        if stockdata["labelb"].values[bandstart+1] == 1:
            fig.add_traces(go.Scatter(x=stockdata.index[bandstart:bandend], y = stockdata['lower_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),showlegend=False,hoverinfo='none'),rows=[r], cols=[c],secondary_ys= [True,True])

            fig.add_traces(go.Scatter(x=stockdata.index[bandstart:bandend], y = stockdata['upper_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),
                                        fill='tonexty', 
                                        fillcolor = 'rgba(256,256,0,0.2)',showlegend=False,hoverinfo='none'
                                        ),rows=[r], cols=[c],secondary_ys= [True,True])
        else:


            fig.add_traces(go.Scatter(x=stockdata.index[bandstart:bandend], y = stockdata['lower_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),showlegend=False,hoverinfo='none'),rows=[r], cols=[c],secondary_ys= [True,True])

            fig.add_traces(go.Scatter(x=stockdata.index[bandstart:bandend], y = stockdata['upper_band'].values[bandstart:bandend],
                                        line = dict(color='rgba(0,0,0,0)'),
                                        fill='tonexty', 
                                        fillcolor = 'rgba(137, 207, 240,0.2)',showlegend=False,hoverinfo='none'
                                        ),rows=[r], cols=[c],secondary_ys= [True,True])
        bandidx =checkidx +1
        if bandidx >=len(stockdata["labelb"].values):
            break






    ### K線圖製作 ###
    fig.add_trace(
        go.Candlestick(
            x=stockdata[(stockdata['all_kk'] == -1)&(stockdata['close'] >stockdata['open'] )].index,
            open=stockdata[(stockdata['all_kk'] == -1)&(stockdata['close'] >stockdata['open'] )]['open'],
            high=stockdata[(stockdata['all_kk'] == -1)&(stockdata['close'] >stockdata['open'] )]['max'],
            low=stockdata[(stockdata['all_kk'] == -1)&(stockdata['close'] >stockdata['open'] )]['min'],
            close=stockdata[(stockdata['all_kk'] == -1)&(stockdata['close'] >stockdata['open'] )]['close'],
            increasing_line_color=decreasing_color,
            increasing_fillcolor=no_color, #fill_increasing_color(stockdata.index>stockdata.index[50])
            decreasing_line_color=decreasing_color,
            decreasing_fillcolor=no_color,#decreasing_color,
            line=dict(width=1),
            name='OHLC',showlegend=False
        )#,

        ,row=r, col=c, secondary_y= True
    )


    fig.add_trace(
        go.Candlestick(
            x=stockdata[(stockdata['all_kk'] == 1)&(stockdata['close'] >stockdata['open'] )].index,
            open=stockdata[(stockdata['all_kk'] == 1)&(stockdata['close'] >stockdata['open'] )]['open'],
            high=stockdata[(stockdata['all_kk'] == 1)&(stockdata['close'] >stockdata['open'] )]['max'],
            low=stockdata[(stockdata['all_kk'] == 1)&(stockdata['close'] >stockdata['open'] )]['min'],
            close=stockdata[(stockdata['all_kk'] == 1)&(stockdata['close'] >stockdata['open'] )]['close'],
            increasing_line_color=increasing_color,
            increasing_fillcolor=no_color, #fill_increasing_color(stockdata.index>stockdata.index[50])
            decreasing_line_color=increasing_color,
            decreasing_fillcolor=no_color,#decreasing_color,
            line=dict(width=1),
            name='OHLC',showlegend=False
        )#,

        ,row=r, col=c, secondary_y= True
    )

    ### K線圖製作 ###
    fig.add_trace(
        go.Candlestick(
            x=stockdata[(stockdata['all_kk'] == -1)&(stockdata['close'] <stockdata['open'] )].index,
            open=stockdata[(stockdata['all_kk'] == -1)&(stockdata['close'] <stockdata['open'] )]['open'],
            high=stockdata[(stockdata['all_kk'] == -1)&(stockdata['close'] <stockdata['open'] )]['max'],
            low=stockdata[(stockdata['all_kk'] == -1)&(stockdata['close'] <stockdata['open'] )]['min'],
            close=stockdata[(stockdata['all_kk'] == -1)&(stockdata['close'] <stockdata['open'] )]['close'],
            increasing_line_color=decreasing_color,
            increasing_fillcolor=decreasing_color, #fill_increasing_color(stockdata.index>stockdata.index[50])
            decreasing_line_color=decreasing_color,
            decreasing_fillcolor=decreasing_color,#decreasing_color,
            line=dict(width=1),
            name='OHLC',showlegend=False
        )#,

        ,row=r, col=c, secondary_y= True
    )


    fig.add_trace(
        go.Candlestick(
            x=stockdata[(stockdata['all_kk'] == 1)&(stockdata['close'] <stockdata['open'] )].index,
            open=stockdata[(stockdata['all_kk'] == 1)&(stockdata['close'] <stockdata['open'] )]['open'],
            high=stockdata[(stockdata['all_kk'] == 1)&(stockdata['close'] <stockdata['open'] )]['max'],
            low=stockdata[(stockdata['all_kk'] == 1)&(stockdata['close'] <stockdata['open'] )]['min'],
            close=stockdata[(stockdata['all_kk'] == 1)&(stockdata['close'] <stockdata['open'] )]['close'],
            increasing_line_color=increasing_color,
            increasing_fillcolor=increasing_color, #fill_increasing_color(stockdata.index>stockdata.index[50])
            decreasing_line_color=increasing_color,
            decreasing_fillcolor=increasing_color,#decreasing_color,
            line=dict(width=1),
            name='OHLC',showlegend=False
        )#,

        ,row=r, col=c, secondary_y= True
    )
    ### 成交量圖製作 ###
    volume_colors = [red_color if stockdata['close'][i] > stockdata['close'][i-1] else green_color for i in range(len(stockdata['close']))]
    volume_colors[0] = green_color

    #fig.add_trace(go.Bar(x=kbars.index, y=kbars['成交金額'], name='Volume', marker=dict(color=volume_colors),showlegend=False), row=optvrank[0], col=1)
    fig.add_trace(go.Bar(x=stockdata.index, y=stockdata['Trading_money'], name='成交金額', marker=dict(color=volume_colors)), row=r, col=c)



    fig.update_annotations(font_size=12)
    #fig.update_layout(title_text = "", hovermode='x unified',
    #                #width = 1200, 
    #                #height = 600,
    #                hoverlabel_namelength=-1,
    #                dragmode = 'drawline',
    #                
    #                hoverlabel=dict(align='left',bgcolor='rgba(255,255,255,0.5)',font=dict(color='black')),
    #                legend_traceorder="reversed"#,row=r, col=c
    #                )
    
    


     ### 圖表設定 ###
    fig.update(layout_xaxis_rangeslider_visible=False)



    #fig.update_traces(xaxis='x1',hoverlabel=dict(align='left'))




    # 隱藏周末與市場休市日期 ### 導入台灣的休市資料
    fig.update_xaxes(
        rangeslider= {'visible':False},
        rangebreaks=[
            dict(bounds=['sat', 'mon']), # hide weekends, eg. hide sat to before mon
            dict(values=[str(holiday) for holiday in holidf[~(holidf["說明"].str.contains('開始交易') | holidf["說明"].str.contains('最後交易'))]["日期"].values]+['2023-08-03'])
        ],
                    row = r, 
                    col = c
    )