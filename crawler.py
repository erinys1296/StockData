import requests
from bs4 import BeautifulSoup


import json
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


    
    newcol = list(df.loc[2,:].dropna().unique()[:3])
    for col1 in df.loc[0,:].dropna().unique():
        for col2 in df.loc[1,:].dropna().unique():
            for col3 in df.loc[2,:].dropna().unique()[3:]: #只要後兩個
                newcol.append('{}_{}_{}'.format(col1,col2,col3))
    
    newcol = [stri.replace(' ','') for stri in newcol]
    df.columns = newcol
    for i in range(3):
        df = df.drop(df.index[0])
        
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

    # 建立查詢表單數據
    data = {
        'queryType': '2',
        'marketCode': '0', # 0: 一般交易時段; 1: "盤後交易時段"
        'dateaddcnt' : '1',
        'commodity_id': 'TXO',
        'commodity_id2': '',
        'queryDate' : querydate,
        'MarketCode' : '1',
        'commodity_idt': 'TXO',
        'commodity_id2t' : '',
        'commodity_id2t2' : ''
    }

    # 向網站發送POST請求
    response = requests.post('https://www.taifex.com.tw/cht/3/optDailyMarketReport', data=data)

    # 解析HTML標記
    soup = BeautifulSoup(response.text, "lxml")

    # 找到表格元素
    table = soup.find("table", {"class": "table_f"})

    # 將表格數據轉換成Pandas數據框
    df = pd.read_html(str(table))[0]
    
    #處理欄位空格
    newcol = [stri.replace(' ','') for stri in df.columns]
    df.columns = newcol
    
    #抓取契約結算日
    
    # 將結算日的爬蟲寫到 function外 (因為不會隨著時間改變而改變，減少爬蟲次數)
    response = requests.get('https://www.taifex.com.tw/cht/5/optIndxFSP')

    # 解析HTML標記
    soup = BeautifulSoup(response.text, "lxml")

    # 找到表格元素
    table = soup.find("table", {"class": "table_c"}) 

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
        weekfilter = df["到期月份(週別)"].unique()[0]
        df = df[df["到期月份(週別)"] == weekfilter]
    
    
    #將 Call 跟 Put 分成兩個 table，並只取 "履約價","最後成交價" 這兩個欄位
    Calltable = df[df["買賣權"] == 'Call'][["履約價","最後成交價"]]
    Puttable = df[df["買賣權"] == 'Put'][["履約價","最後成交價"]]
    
    #轉換型態及資料處理
    Calltable["履約價"] = Calltable["履約價"].astype('int')
    Puttable["履約價"] = Puttable["履約價"].astype('int')

    Calltable.loc[Calltable["最後成交價"] == '-',"最後成交價"] = None
    Calltable = Calltable.dropna()
    Puttable.loc[Puttable["最後成交價"] == '-',"最後成交價"] = None
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
    
    newcol = list(df.loc[2,:].dropna().unique()[:4])
    for col1 in df.loc[0,:].dropna().unique():
        for col2 in df.loc[1,:].dropna().unique():
            for col3 in df.loc[2,:].dropna().unique()[4:]:
                newcol.append('{}_{}_{}'.format(col1,col2,col3))

    #處理空格問題
    newcol = [stri.replace(' ','') for stri in newcol]
    
    #將欄位名稱設定為新的欄位後，將前三欄的內容刪除
    df.columns = newcol
    for i in range(3):
        df = df.drop(df.index[0])
        
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
