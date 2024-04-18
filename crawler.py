import requests
from bs4 import BeautifulSoup
import urllib3


import pandas as pd
from datetime import datetime, date

#例外處理
import time 


# 大盤指數成交量
def retry_requests(url, headers):
    
    for i in range(3):
        try:
            return requests.get(url, headers=headers)
        except:
            print('發生錯誤，等待1分鐘後嘗試')
            time.sleep(60)
    
    return None


def get_taiex_vol(date):
    
    url = "https://www.twse.com.tw/exchangeReport/FMTQIK?response=json&date={}&_={}"

    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en,zh-TW;q=0.9,zh;q=0.8,en-US;q=0.7",
        "Connection": "keep-alive",
        "Host": "www.twse.com.tw",
        "Referer": "https://www.twse.com.tw/zh/page/trading/exchange/FMTQIK.html",
        "sec-ch-ua": '"Chromium";v="104", " Not A;Brand";v="99", "Google Chrome";v="104"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
    }

    response = retry_requests(url.format(date, int(datetime.now().timestamp())), headers=headers)
    
    try:
        response_data = response.json()
        df = pd.DataFrame(response_data['data'], columns=response_data['fields'])
        df.replace(',', '', regex=True, inplace=True)
        df = df.apply(pd.to_numeric, errors='ignore')
        #df['日期'] = df['日期'].apply(lambda date: pd.to_datetime('{}/{}'.format(int(date.split('/', 1)[0]) + 1911, date.split('/', 1)[1])))
        #df = df.set_index('日期')
        print(f'{date} 下載完成')
    except:
        df = pd.DataFrame()
        print(f'{date} 無資料')

    return df


def get_taiex(date):
    
    url = "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?response=json&date={}&_={}"

    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en,zh-TW;q=0.9,zh;q=0.8,en-US;q=0.7",
        "Connection": "keep-alive",
        "Host": "www.twse.com.tw",
        "Referer": "https://www.twse.com.tw/zh/page/trading/exchange/FMTQIK.html",
        "sec-ch-ua": '"Chromium";v="104", " Not A;Brand";v="99", "Google Chrome";v="104"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
    }

    response = retry_requests(url.format(date, int(datetime.now().timestamp())), headers=headers)

    try:
        response_data = response.json()
        df = pd.DataFrame(response_data['data'], columns=response_data['fields'])
        df.replace(',', '', regex=True, inplace=True)
        df = df.apply(pd.to_numeric, errors='ignore')
        #df['日期'] = df['日期'].apply(lambda date: pd.to_datetime('{}/{}'.format(int(date.split('/', 1)[0]) + 1911, date.split('/', 1)[1])))
        #df = df.set_index('日期')
        print(f'{date} 下載完成')
    except:
        df = pd.DataFrame()
        print(f'{date} 無資料')

    return df


def catch_cost(date):
    # 設定欲爬取的日期 -> 直接從function 的 input data 索取
    # date = "2023/4/19"

    # 製作 POST 請求
    url = "https://www.taifex.com.tw/cht/3/futContractsDate"
    data = {
        "queryDate": date,
        "commodity_id": "TX",
        "commodity_id2": "",
        "commodity_id3": "",
        "syear": "",
        "smonth": "",
        "eyear": "",
        "emonth": "",
        "datestart": "",
        "dateend": "",
        "commodity_idt": "",
        "MarketCode": "",
        "commodity_id2t": "",
        "commodity_id3t": "",
        "queryDate1": "",
        "MarketCode1": "",
        "commodity_idt1": "",
        "commodity_id2t1": "",
        "commodity_id3t1": "",
    }
    res = requests.post(url, data=data)

    # 使用 BeautifulSoup 解析 HTML 內容
    soup = BeautifulSoup(res.text, "html.parser")
    

    try:
        # 找出表格標籤 (table) 與表格行標籤 (tr)
        table = soup.find("table", {"class": "table_f"})
        #rows = table.find_all("tr")


        # 將表格資料轉換為 pandas 的 dataframe 格式
        df=pd.read_html(str(table))[0]
    except:
        return


    
    newcol = []
    for i in range(3):
        newcol.append(df.columns[i][2])

    for i in range(3,15):
        newcol.append('{}_{}_{}'.format(df.columns[i][0],df.columns[i][1],df.columns[i][2]))
    df.columns = newcol
        
    Longdata = df[(df["商品名稱"]=="臺股期貨") & (df["身份別"]=="外資")][["交易口數與契約金額_多方_口數","交易口數與契約金額_多方_契約金額"]].values[0]
    
    #多方平均交易價格資料
    Longdata = df[(df["商品名稱"] == "臺股期貨")&(df["身份別"] == "外資")][["交易口數與契約金額_多方_口數","交易口數與契約金額_多方_契約金額"]].values[0]
    Long_avg = (int(Longdata[1])*1000/int(Longdata[0]))/200
    
    #空方平均交易價格資料
    df[(df["商品名稱"] == "臺股期貨")&(df["身份別"] == "外資")][["交易口數與契約金額_空方_口數","交易口數與契約金額_空方_契約金額"]]
    
    #空方平均交易價格資料
    Shortdata = df[(df["商品名稱"] == "臺股期貨")&(df["身份別"] == "外資")][["交易口數與契約金額_空方_口數","交易口數與契約金額_空方_契約金額"]].values[0]
    Short_avg = (int(Shortdata[1])*1000/int(Shortdata[0]))/200
    
    cost = 0
    if Long_avg > Short_avg:
        cost = Long_avg
    else:
        cost = Short_avg
        
    return cost


def callputtable(querydate):

    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0wNy0zMCAyMzowMTo0MSIsInVzZXJfaWQiOiJqZXlhbmdqYXUiLCJpcCI6IjExNC4zNC4xMjEuMTA0In0.WDAZzKGv4Du5JilaAR7o7M1whpnGaR-vMDuSeTBXhhA"
    url = "https://api.finmindtrade.com/api/v4/data?"

    parameter = {
        "dataset": "TaiwanOptionDaily",
        "data_id":"TXO",
        "start_date": querydate.replace('/','-'),
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
        enddate = datedf[datedf[datedf.columns[0]]>querydate][datedf.columns[0]].min()
        weekfilter = datedf[datedf[datedf.columns[0]] == enddate]["契約月份"].values[0]
        df = df[df["到期月份(週別)"] == weekfilter]

    except:
        if querydate in datedf[datedf.columns[0]].values:
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


def catch_limit(querydate):
    
    
    
    # 利用相同的手法，抓取每日選擇權買賣權分計
    time.sleep(2)
    url = 'https://www.taifex.com.tw/cht/3/callsAndPutsDate'


    # 建立查詢表單數據
    data = {
        'queryType': '1',
        'goDay' : '',
        'doQuery' : '1',
        'dateaddcnt' : '',
        'queryDate': querydate,
        'commodityId': ''
    }


    # 向網站發送POST請求
    response = requests.post(url, data=data)

    # 解析HTML標記
    soup = BeautifulSoup(response.text, 'lxml')
    
    try:
        # 找到表格元素
        table = soup.find('table', {'class': 'table_f'})

        # 將表格數據轉換成Pandas數據框
        df = pd.read_html(str(table))[0]
    except:
        return
    
    Calltable,Puttable = callputtable(querydate)
    
    newcol = []
    for i in range(4):
        newcol.append(df.columns[i][2])

    for i in range(4,16):
        newcol.append('{}_{}_{}'.format(df.columns[i][0],df.columns[i][1],df.columns[i][2]))
    

    #處理空格問題
    newcol = [stri.replace(' ','') for stri in newcol]
    df.columns = newcol
    

        
    dfnew = df[(df["商品名稱"] == "臺指選擇權")&((df["身份別"] == "自營商")|(df["身份別"] == "外資"))][newcol[:4]+["未平倉餘額_賣方_口數","未平倉餘額_賣方_契約金額"]]
    
    #資料轉換型態
    dfnew["未平倉餘額_賣方_口數"] = dfnew["未平倉餘額_賣方_口數"].astype('int')
    dfnew["未平倉餘額_賣方_契約金額"] = dfnew["未平倉餘額_賣方_契約金額"].astype('int')
    
    #計算SC成本 "注意這裡的轉換型態方法"
    dfnew["SC成本"] = (round(dfnew["未平倉餘額_賣方_契約金額"]*1000/50/dfnew["未平倉餘額_賣方_口數"],0)).astype('int')
    
    
    dfnew["位置"] = 0
    for i in dfnew.index:
        if dfnew.loc[i,"權別"] == "買權":
            dfnew.loc[i,"位置"] = Calltable[(Calltable["最後成交價"]<dfnew.loc[i,"SC成本"])&(Calltable["最後成交價"]!=0)]["履約價"].min()
        else: #賣權
            dfnew.loc[i,"位置"] = Puttable[(Puttable["最後成交價"]<dfnew.loc[i,"SC成本"])&(Puttable["最後成交價"]!=0)]["履約價"].max()
            
    #轉換talbe並計算上下極限
    #上下極限計算
    dfarr = []
    for i in dfnew["身份別"].unique():
        Callcost = dfnew[(dfnew["身份別"] == i)&(dfnew["權別"] == '買權')]["SC成本"].values[0]
        Putcost = dfnew[(dfnew["身份別"] == i)&(dfnew["權別"] == '賣權')]["SC成本"].values[0]
        Callposition = dfnew[(dfnew["身份別"] == i)&(dfnew["權別"] == '買權')]["位置"].values[0]
        Putposition = dfnew[(dfnew["身份別"] == i)&(dfnew["權別"] == '賣權')]["位置"].values[0]
        maxlimit = Callposition + Callcost + Putcost
        minlimit = Putposition - Callcost - Putcost
        dfarr.append(["臺指選擇權",querydate,i,maxlimit,minlimit])
    
    return pd.DataFrame(dfarr,columns = ["商品名稱","日期","身份別","上極限","下極限"])



def catch_volumn(date):

    url = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_5MINS?response=json&date={}&_={}"
    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en,zh-TW;q=0.9,zh;q=0.8,en-US;q=0.7",
        "Connection": "keep-alive",
        "Host": "www.twse.com.tw",
        "Referer": "https://www.twse.com.tw/zh/page/trading/exchange/FMTQIK.html",
        "sec-ch-ua": '"Chromium";v="104", " Not A;Brand";v="99", "Google Chrome";v="104"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
    }
    try:
        response = requests.get(url.format(date, int(datetime.now().timestamp())), headers=headers)
        response_data = response.json()
        df = pd.DataFrame(response_data['data'], columns=response_data['fields'])
    except:
        return
    return int(df["累積委託賣出數量"].values[0].replace(',' , ''))



def query_put_call(start_date,end_date):
    http = urllib3.PoolManager()
    url = "https://www.taifex.com.tw/cht/3/pcRatio"
    res = http.request(
         'GET',
          url,
          fields={
             'queryStartDate': start_date,
             'queryEndDate': end_date
          }
     )

    html_doc = res.data
    soup = BeautifulSoup(html_doc, 'html.parser')
    table = soup.table
    df = pd.read_html(str(table))
    
    pc_ratio = df[0]
    for row in range(pc_ratio.shape[0]):
        date2 = pc_ratio.iloc[row,0].split('/')
        pc_ratio.iloc[row, 0] = datetime(int(date2[0]), int(date2[1]), int(date2[2]))

    return pc_ratio



def get_MTX_Ratio(date):
    # 製作 POST 請求
    url = "https://www.taifex.com.tw/cht/3/futDailyMarketReport"
    data = {
        "queryType": '2',
        "commodity_id": "MTX",
        "commodity_id2": "",
        "dateaddcnt": "",
        "MarketCode": "0",
        "queryDate": date,
        "commodity_idt": "MTX",
        "commodity_id2t": "",
        "commodity_id2t2": "",
    }
    res = requests.post(url, data=data)
    # 使用 BeautifulSoup 解析 HTML 內容
    soup = BeautifulSoup(res.text, "html.parser")



    # 找出表格標籤 (table) 與表格行標籤 (tr)
    table = soup.find("table", {"class": "table_f"})
    #rows = table.find_all("tr")


    # 將表格資料轉換為 pandas 的 dataframe 格式
    df=pd.read_html(str(table))[0]
    a = df["*未沖銷契約量"].values[-1]
    
    # 製作 POST 請求
    url = "https://www.taifex.com.tw/cht/3/futContractsDate"
    data = {
        "queryType": '1',
        "commodityId": "MXF",
        "goDay": "",
        "doQuery": "1",
        "dateaddcnt": "",
        "queryDate": date,

    }
    res = requests.post(url, data=data)
    # 使用 BeautifulSoup 解析 HTML 內容
    soup = BeautifulSoup(res.text, "html.parser")



    # 找出表格標籤 (table) 與表格行標籤 (tr)
    table = soup.find("table", {"class": "table_f"})
    #rows = table.find_all("tr")


    # 將表格資料轉換為 pandas 的 dataframe 格式
    df=pd.read_html(str(table))[0]    
    
    
    b = int(df[9].values[-1])
    c = int(df[11].values[-1])
    
    return ((a-b)-(a-c))/a


def get_margin(querydate):
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0wNy0zMCAyMzowMTo0MSIsInVzZXJfaWQiOiJqZXlhbmdqYXUiLCJpcCI6IjExNC4zNC4xMjEuMTA0In0.WDAZzKGv4Du5JilaAR7o7M1whpnGaR-vMDuSeTBXhhA"
    url = "https://api.finmindtrade.com/api/v4/data?"

    # filter stock_id
    parameter = {
        "dataset": "TaiwanStockInfo",
        "token": token, # 參考登入，獲取金鑰
    }
    data = requests.get(url, params=parameter)
    TaiwanStockInfo = data.json()
    TaiwanStockInfo = pd.DataFrame(TaiwanStockInfo['data'])
    TaiwanStockInfo['is_bt'] = TaiwanStockInfo['stock_name'].map(lambda x: True if '乙特' in x else False)
    TaiwanStockInfo['is_at'] = TaiwanStockInfo['stock_name'].map(lambda x: True if '甲特' in x else False)

    mask = (
        TaiwanStockInfo['type'].isin(['twse']) & 
        ~TaiwanStockInfo['industry_category'].isin(['ETF', '大盤']) &
        ~TaiwanStockInfo['is_bt'] &
        ~TaiwanStockInfo['is_at']
    )
    #print(TaiwanStockInfo[mask].tail())
    stock_type_list = TaiwanStockInfo[mask]['stock_id'].unique()

    # 獲得個股每日收盤價
    parameter = {
        "dataset": "TaiwanStockPrice",
        "start_date": querydate,
        "token": token, # 參考登入，獲取金鑰
    }
    resp = requests.get(url, params=parameter)
    TaiwanStockPrice = resp.json()
    TaiwanStockPrice = pd.DataFrame(TaiwanStockPrice["data"])
    #print(TaiwanStockPrice[["date", "stock_id", "close"]].head())

    # 獲得個股每日融資張數
    parameter = {
        "dataset": "TaiwanStockMarginPurchaseShortSale",
        "start_date": querydate,
        "token": token, # 參考登入，獲取金鑰
    }
    resp = requests.get(url, params=parameter)
    TaiwanStockMarginPurchaseShortSale = resp.json()
    TaiwanStockMarginPurchaseShortSale = pd.DataFrame(TaiwanStockMarginPurchaseShortSale["data"])
    TaiwanStockMarginPurchaseShortSale = TaiwanStockMarginPurchaseShortSale[
        ['date', 'stock_id', 'MarginPurchaseTodayBalance']
    ]
    #print(TaiwanStockMarginPurchaseShortSale.head())



    # 獲得大盤融資餘額
    parameter = {
        "dataset": "TaiwanStockTotalMarginPurchaseShortSale",
        "start_date": querydate,
        "token": token, # 參考登入，獲取金鑰
    }
    data = requests.get(url, params=parameter)
    TaiwanStockTotalMarginPurchaseShortSale = data.json()
    TaiwanStockTotalMarginPurchaseShortSale = pd.DataFrame(TaiwanStockTotalMarginPurchaseShortSale['data'])
    TaiwanStockTotalMarginPurchaseShortSale = TaiwanStockTotalMarginPurchaseShortSale[TaiwanStockTotalMarginPurchaseShortSale['name']=='MarginPurchaseMoney']
    TaiwanStockTotalMarginPurchaseShortSale = TaiwanStockTotalMarginPurchaseShortSale[TaiwanStockTotalMarginPurchaseShortSale['date']==querydate]
    #print(TaiwanStockTotalMarginPurchaseShortSale[["date", "TodayBalance"]].tail())

    # 計算2022-06-29 大盤融資維持率
    merge_data = pd.merge(TaiwanStockPrice, TaiwanStockMarginPurchaseShortSale, on=['date', 'stock_id'], how='left')
    merge_data['MarginPurchaseTotalValue'] = merge_data['MarginPurchaseTodayBalance'] * merge_data['close'] * 1000
    value = merge_data[merge_data['stock_id'].isin(stock_type_list)]['MarginPurchaseTotalValue'].sum()
    #print(TaiwanStockTotalMarginPurchaseShortSale)
    return value / TaiwanStockTotalMarginPurchaseShortSale['TodayBalance'].values[0]

def callputtable_month(querydate):

    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0wNy0zMCAyMzowMTo0MSIsInVzZXJfaWQiOiJqZXlhbmdqYXUiLCJpcCI6IjExNC4zNC4xMjEuMTA0In0.WDAZzKGv4Du5JilaAR7o7M1whpnGaR-vMDuSeTBXhhA"
    url = "https://api.finmindtrade.com/api/v4/data?"

    parameter = {
        "dataset": "TaiwanOptionDaily",
        "data_id":"TXO",
        "start_date": querydate.replace('/','-'),
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
    df = df.drop(df.index[-1])
    df = df.dropna()

    

    #處理欄位空格
    #newcol = [stri.replace(' ','') for stri in datedf.columns]
    #datedf.columns = newcol

    weekfilter = df[~(df[df.columns[2]].str.contains("W"))][df.columns[2]].min()
    #print(weekfilter)

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
