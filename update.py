import requests
from bs4 import BeautifulSoup
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

#爬蟲
import crawler

#日期處理
from datetime import datetime, timedelta

#例外處理
from time import sleep

#資料庫處理
import sqlite3
import csv


connection = sqlite3.connect('主圖資料.sqlite3')


taiex = pd.read_sql("select distinct * from taiex", connection, parse_dates=['日期'])
taiex_vol = pd.read_sql("select distinct * from taiex_vol", connection, parse_dates=['日期'])
cost_df = pd.read_sql("select distinct * from cost", connection)
limit_df = pd.read_sql("select distinct * from [limit]", connection)

enddate = pd.read_sql("select * from end_date order by 最後結算日 desc", connection)

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
datedf.columns = ['最後結算日', '契約月份', '臺指選擇權（TXO）', '電子選擇權（TEO）', '金融選擇權（TFO）']
#print(datedf)
datedf.to_sql('end_date', connection, if_exists='append', index=False) 
#connection.executemany('INSERT INTO end_date VALUES (?, ?, ?, ?, ?)', np.array(datedf))

ordervolumn = pd.read_sql("select distinct * from ordervolumn where 九點累積委託賣出數量 not null", connection)
putcallsum = pd.read_sql("select 日期, max(價平和) as 價平和 from putcallsum group by 日期", connection)
putcallsum = putcallsum[putcallsum["價平和"]>0.1]
putcallgap = pd.read_sql("select 日期, max(價外買賣權價差) as 價外買賣權價差 from putcallgap group by 日期", connection)

bank8 = pd.read_sql("select distinct * from bank", connection)
#test = crawler.catch_cost('20230601')
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

maxtime = datetime.strptime(cost_df["Date"].max(), '%Y/%m/%d')
#print(cost_df)

for i in range((datetime.today() - maxtime).days):#
   
    try:
        querydate = datetime.strftime(datetime.today()- timedelta(days=i),'%Y/%m/%d')
        result = crawler.catch_cost(querydate)
        if result != None:
            cost_df = pd.concat([cost_df,pd.DataFrame([[querydate,result]],columns = ["Date","Cost"])])
    except:
        sleep(5)
        try:
            querydate = datetime.strftime(datetime.today()- timedelta(days=i),'%Y/%m/%d')
            result = crawler.catch_cost(querydate)
            if result != None:
                cost_df = pd.concat([cost_df,pd.DataFrame([[querydate,result]],columns = ["Date","Cost"])])
        except:
            print(querydate,"query error")
            
print(cost_df)
cost_df.to_sql('cost', connection, if_exists='replace', index=False) 
#connection.executemany('INSERT INTO cost VALUES (?, ?)', np.array(cost_df))

maxtime = datetime.strptime(limit_df["日期"].max(), '%Y/%m/%d')

for i in range((datetime.today() - maxtime).days):
    try:
        querydate = datetime.strftime(datetime.today()- timedelta(days=i),'%Y/%m/%d')
        
        limit_df = pd.concat([limit_df,crawler.catch_limit(querydate)])
    except:
        sleep(5)
        try:
            querydate = datetime.strftime(datetime.today()- timedelta(days=i),'%Y/%m/%d')
            limit_df = pd.concat([limit_df,crawler.catch_limit(querydate)])
        except:
            print(querydate,"query error")
print(limit_df.tail(5))
connection = sqlite3.connect('主圖資料.sqlite3')
limit_df.to_sql('limit', connection, if_exists='replace', index=False) 

#connection.executemany('replace INTO limit VALUES (?, ?, ?, ?, ?)', np.array(limit_df))
"""
taxidatestart = taiex["日期"].max().strftime("%Y%m")+"01"
taxidateend = datetime.strftime(datetime.today(),'%Y%m')+"01"
for date in pd.date_range(taxidatestart, taxidateend, freq='MS').strftime('%Y%m%d'):
    try:
        df = crawler.get_taiex(date)
        df['日期']= df['日期'].apply(lambda date: pd.to_datetime('{}/{}'.format(int(date.split('/', 1)[0]) + 1911, date.split('/', 1)[1])))
        taiex = pd.concat([taiex, df], sort=False)

        sleep(5)
    except:
        continue
for date in pd.date_range(taxidatestart, taxidateend, freq='MS').strftime('%Y%m%d'):
    try:
        df = crawler.get_taiex_vol(date)
        df['日期']= df['日期'].apply(lambda date: pd.to_datetime('{}/{}'.format(int(date.split('/', 1)[0]) + 1911, date.split('/', 1)[1])))
        taiex_vol = pd.concat([taiex_vol, df], sort=False)
    except:
        continue

    sleep(5)


taiex_vol.to_sql('taiex_vol', connection, if_exists='replace', index=False) 
taiex.to_sql('taiex', connection, if_exists='replace', index=False) 
"""
#connection.executemany('replace INTO taiex_vol VALUES (?, ?, ?, ?, ?, ?)', np.array(taiex_vol))
#connection.executemany('replace INTO taiex VALUES (?, ?, ?, ?, ?)', np.array(taiex))

#累積委託量
maxtime = datetime.strptime(ordervolumn["日期"].max(), '%Y%m%d')

for i in range((datetime.today() - maxtime).days):#
   
    try:
        querydate = datetime.strftime(datetime.today()- timedelta(days=i),'%Y%m%d')
        
        ordervolumn = pd.concat([ordervolumn,pd.DataFrame([[querydate,crawler.catch_volumn(querydate)]],columns = ["日期","九點累積委託賣出數量"])])
    except:
        sleep(5)
        try:
            querydate = datetime.strftime(datetime.today()- timedelta(days=i),'%Y%m%d')
            ordervolumn = pd.concat([ordervolumn,pd.DataFrame([[querydate,crawler.catch_volumn(querydate)]],columns = ["日期","九點累積委託賣出數量"])])
        except:
            print(querydate,"query error")

ordervolumn.to_sql('ordervolumn', connection, if_exists='replace', index=False) 
#connection.executemany('replace INTO ordervolumn VALUES (?, ?)', np.array(ordervolumn))

maxtime = datetime.strptime(putcallsum["日期"].max(), '%Y/%m/%d')

#價平和 ＆ 價外買賣權價差
for i in range((datetime.today() - maxtime).days+7):
    querydate = datetime.strftime(datetime.today()- timedelta(days=i),'%Y/%m/%d')
    #CT,PT = crawler.callputtable(querydate)
    #print(querydate)
    try:
        CT,PT = crawler.callputtable(querydate)
        CT.columns = ["履約價","CT成交價"]
        PT.columns = ["履約價","PT成交價"]
        sumdf = CT.join(PT.set_index("履約價"),on=["履約價"],lsuffix='_left', rsuffix='_right')
        sumdf["CTPT差"] = np.abs(sumdf["CT成交價"] - sumdf["PT成交價"])
        sumdf["CTPT和"] = sumdf["CT成交價"] + sumdf["PT成交價"]
        sumdf = sumdf[sumdf["CTPT差"] == sumdf["CTPT差"].min()]
        sumdf = sumdf[sumdf["CTPT和"] == sumdf["CTPT和"].min()]

        cn = sumdf[sumdf["CTPT差"] == sumdf["CTPT差"].min()]['履約價'].values[0]+200
        pn = sumdf[sumdf["CTPT差"] == sumdf["CTPT差"].min()]['履約價'].values[0]-200
        
    except:
        if (datetime.today()- timedelta(days=i)).weekday() not in [5,6]:
            print(querydate,'error')
        continue
    try:
       result = sumdf[sumdf["CTPT差"] == sumdf["CTPT差"].min()][["CTPT和"]].values.mean()
       putcallsum = pd.concat([putcallsum,pd.DataFrame([[querydate,result]],columns = ["日期","價平和"])])
    except:
        if (datetime.today()- timedelta(days=i)).weekday() not in [5,6]:
            print(querydate,'error')
        
    try:
       result2 = round(CT[CT["履約價"] == cn]["CT成交價"].values[0] / PT[PT["履約價"] == pn]["PT成交價"].values[0],3) - 1
       putcallgap = pd.concat([putcallgap,pd.DataFrame([[querydate,result2]],columns = ["日期","價外買賣權價差"])])
    except:
        if (datetime.today()- timedelta(days=i)).weekday() not in [5,6]:
            print(querydate,'error')
        
    
    
    

putcallgap.to_sql('putcallgap', connection, if_exists='replace', index=False) 
putcallsum.to_sql('putcallsum', connection, if_exists='replace', index=False) 
print(putcallsum.tail(20))
#connection.executemany('replace INTO putcallsum VALUES (?, ?)', np.array(putcallsum))
print('putcallsum complete')
# 將結算日的爬蟲寫到 function外 (因為不會隨著時間改變而改變，減少爬蟲次數)
try:
    data = {'ityIds': '2',
    'commodityIds': '8',
    'commodityIds': '9',
    'commodityIds': '11',
    'commodityIds': '14',
    'commodityIds': '16',
    'all' : 'all',
    '_all':'on',
    'start_year': '2019',
    'start_month': '06',
    'end_year': '2023',
    'end_month': '06',
        'button':'送出查詢'}

    #response = requests.post('https://www.taifex.com.tw/cht/5/optIndxFSP', data=data)

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

    datedf.columns = ['最後結算日', '契約月份', '臺指選擇權（TXO）', '電子選擇權（TEO）', '金融選擇權（TFO）']
    datedf.to_sql('end_date', connection, if_exists='replace', index=False)
except:
    print('enddate error') 
#connection.executemany('replace INTO end_date VALUES (?, ?, ?, ?, ?)', np.array(datedf))
#CPratio = pd.read_sql("select distinct * from putcallratio", connection, parse_dates=['日期'])
result=pd.DataFrame()
for i in range(3):
    
    try:
        start_date = datetime.strftime(datetime.today()- timedelta(days=(i+1)*30),'%Y/%m/%d')
        end_date = datetime.strftime(datetime.today()- timedelta(days=i*30),'%Y/%m/%d')
        tempdf = crawler.query_put_call(start_date,end_date)
        if result.empty and tempdf is not None:
            result = tempdf
        else:
            result = pd.concat([result,tempdf])
        print(start_date,end_date,"query success")
    except:
        print(start_date,end_date,"query fail 1")
        continue
try:
   result.to_sql('putcallratio', connection, if_exists='replace', index=False)
except:
   print("ratio fail")
print('ratio complete')
#connection.executemany('replace INTO putcallratio VALUES (?, ?, ?, ?, ?, ?, ?)', np.array(result))     


def eight_bank(datetime):
    url = "https://api.finmindtrade.com/api/v4/data"
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRlIjoiMjAyMy0wNy0wOSAyMDo0OToyOSIsInVzZXJfaWQiOiJqZXlhbmdqYXUiLCJpcCI6IjExNC4zNC4xMjEuMTA0In0.ZFpDG2LY-jW2bWjd_WTBq4g0AM6I8myU1CTVh12Ka5Q"


    parameter = {
        "dataset": "TaiwanStockGovernmentBankBuySell",
        "start_date": datetime,
        "token": token, # 參考登入，獲取金鑰
    }
    data = requests.get(url, params=parameter)
    data = data.json()
    data = pd.DataFrame(data['data'])
    return (data.buy_amount.sum() - data.sell_amount.sum())

maxtime = datetime.strptime(bank8["日期"].max(), '%Y-%m-%d')


for i in range((datetime.today() - maxtime).days):#
   
    try:
        querydate = datetime.strftime(datetime.today()- timedelta(days=i),'%Y-%m-%d')
        result = eight_bank(querydate)
        if result != None:
            bank8 = pd.concat([bank8,pd.DataFrame([[querydate,result]],columns = ["日期","八大行庫買賣超金額"])])
    except:
        sleep(5)
        try:
            querydate = datetime.strftime(datetime.today()- timedelta(days=i),'%Y-%m-%d')
            result = eight_bank(querydate)
            if result != None:
                cbank8 = pd.concat([bank8,pd.DataFrame([[querydate,result]],columns = ["日期","八大行庫買賣超金額"])])
        except:
            print(querydate,"八大error")

bank8.to_sql('bank', connection, if_exists='replace', index=False) 

dfMTX = pd.read_sql("select distinct * from dfMTX", connection)
maxtime = datetime.strptime(dfMTX["Date"].max(), '%Y/%m/%d')
for i in range((datetime.today() - maxtime).days):#
   
    try:
        querydate = datetime.strftime(datetime.today()- timedelta(days=i),'%Y/%m/%d')
        result = crawler.get_MTX_Ratio(querydate)
        if result != None:
            dfMTX = pd.concat([dfMTX,pd.DataFrame([[querydate,result]],columns = ["Date","MTXRatio"])])
    except:
        sleep(5)
        try:
            querydate = datetime.strftime(datetime.today()- timedelta(days=i),'%Y/%m/%d')
            result = crawler.get_MTX_Ratio(querydate)
            if result != None:
                dfMTX = pd.concat([dfMTX,pd.DataFrame([[querydate,result]],columns = ["Date","MTXRatio"])])
        except:
            print(querydate,"query error")
            
dfMTX.to_sql('dfMTX', connection, if_exists='replace', index=False) 


dfMargin = pd.read_sql("select distinct * from dfMargin", connection)
maxtime = datetime.strptime(dfMargin["Date"].max(), '%Y-%m-%d')
for i in range((datetime.today() - maxtime).days):#
    try:
        querydate = datetime.strftime(datetime.today()- timedelta(days=i),'%Y-%m-%d')
        result = crawler.get_margin(querydate)
        if result != None:
            dfMargin = pd.concat([dfMargin,pd.DataFrame([[querydate,result]],columns = ["Date","MarginRate"])])
    except:
        sleep(5)
        try:
            querydate = datetime.strftime(datetime.today()- timedelta(days=i),'%Y-%m-%d')
            result = crawler.get_margin(querydate)
            if result != None:
                dfMargin = pd.concat([dfMargin,pd.DataFrame([[querydate,result]],columns = ["Date","MarginRate"])])
        except:
            print(querydate,"Margin query error")
            
dfMargin.to_sql('dfMargin', connection, if_exists='replace', index=False) 


dfbuysell = pd.read_sql("select distinct * from dfbuysell", connection)
maxtime = datetime.strptime(dfbuysell["Date"].max(), '%Y%m%d')

for i in range((datetime.today() - maxtime).days):#
   
    try:
        querydate = datetime.strftime(datetime.today()- timedelta(days=i),'%Y%m%d')
        url = "https://www.twse.com.tw/fund/BFI82U?response=json&dayDate="+querydate+"&type=day"
        res = requests.get(url)
        response_data = res.json()
        df = pd.DataFrame(response_data['data'], columns=response_data['fields'])
        df.replace(',', '', regex=True, inplace=True)
        df = df.apply(pd.to_numeric, errors='ignore')
        result = int(df[df['單位名稱'] == "外資及陸資(不含外資自營商)"]["買賣差額"].values[0])/100000000
        
        if result != None:
            dfbuysell = pd.concat([dfbuysell,pd.DataFrame([[querydate,result]],columns = ["Date","ForeBuySell"])])
    except:
        sleep(5)
        try:
            querydate = datetime.strftime(datetime.today()- timedelta(days=i),'%Y%m%d')
            url = "https://www.twse.com.tw/fund/BFI82U?response=json&dayDate="+querydate+"&type=day"
            res = requests.get(url)
            response_data = res.json()
            df = pd.DataFrame(response_data['data'], columns=response_data['fields'])
            df.replace(',', '', regex=True, inplace=True)
            df = df.apply(pd.to_numeric, errors='ignore')
            result = int(df[df['單位名稱'] == "外資及陸資(不含外資自營商)"]["買賣差額"].values[0])/100000000
        
            if result != None:
                dfbuysell = pd.concat([dfbuysell,pd.DataFrame([[querydate,result]],columns = ["Date","ForeBuySell"])])
        except:
            print(querydate,"query error")
            
dfbuysell.to_sql('dfbuysell', connection, if_exists='replace', index=False) 


check = 0
checki = 0
while check == 0 and checki<5:
    try:
        url = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
        data = {
            "queryStartDate": datetime.strftime(datetime.today()- timedelta(days=90),'%Y/%m/%d'),
            "queryEndDate": datetime.strftime(datetime.today()- timedelta(days=0),'%Y/%m/%d'),
            "commodityId": "TXF",

        }
        res = requests.post(url, data=data)
        check = 1
        checki +=1
    
        
    except:
        print("error")
        continue



try:
    tempdf = pd.DataFrame(csv.reader(res.text.splitlines()[:]))
    tempdf.columns = tempdf.loc[0,:]
    futdf = tempdf[tempdf["身份別"] == "外資及陸資"][["日期","多空未平倉口數淨額"]]
    futdf.to_sql('futdf', connection, if_exists='replace', index=False)
except:
    print("final error")


check = 0
checki = 0
while check == 0 and checki<5:
    try:
        url = "https://www.taifex.com.tw/cht/3/callsAndPutsDateDown"
        data = {
            "queryStartDate": datetime.strftime(datetime.today()- timedelta(days=90),'%Y/%m/%d'),
            "queryEndDate": datetime.strftime(datetime.today()- timedelta(days=0),'%Y/%m/%d'),
            "commodityId": "TXO",

        }
        res = requests.post(url, data=data)
        check = 1
        checki +=1
    except:
        continue
try:
    tempdf = pd.DataFrame(csv.reader(res.text.splitlines()[:]))
    tempdf.columns = tempdf.loc[0,:]
    tempdf = tempdf.drop([0]).apply(pd.to_numeric, errors='ignore')
    TXOOIdf = tempdf[(tempdf["身份別"] == "外資及陸資")&(tempdf["買賣權別"] == "CALL")][["日期"]]
    TXOOIdf["買買賣賣"] = tempdf[(tempdf["身份別"] == "外資及陸資")&(tempdf["買賣權別"] == "CALL")]["買方未平倉契約金額(千元)"].values + tempdf[(tempdf["身份別"] == "外資及陸資")&(tempdf["買賣權別"] == "PUT")]["賣方未平倉契約金額(千元)"].values
    TXOOIdf["買賣賣買"] = tempdf[(tempdf["身份別"] == "外資及陸資")&(tempdf["買賣權別"] == "CALL")]["賣方未平倉契約金額(千元)"].values + tempdf[(tempdf["身份別"] == "外資及陸資")&(tempdf["買賣權別"] == "PUT")]["買方未平倉契約金額(千元)"].values
    TXOOIdf.to_sql('TXOOIdf', connection, if_exists='replace', index=False)
except:
    print("final error2")

putcallsum_month = pd.read_sql("select 日期, max(月選擇權價平和) as 月選擇權價平和 from putcallsum_month group by 日期", connection)
putcallgap_month = pd.read_sql("select 日期, max(價外買賣權價差) as 價外買賣權價差 from putcallgap_month group by 日期", connection)
for i in range(5):
    querydate = datetime.strftime(datetime.today()- timedelta(days=i),'%Y/%m/%d')

    try:
        CT,PT = crawler.callputtable_month(querydate)
        CT.columns = ["履約價","CT成交價"]
        PT.columns = ["履約價","PT成交價"]
        sumdf = CT.join(PT.set_index("履約價"),on=["履約價"],lsuffix='_left', rsuffix='_right')
        sumdf["CTPT差"] = np.abs(sumdf["CT成交價"] - sumdf["PT成交價"])
        sumdf["CTPT和"] = sumdf["CT成交價"] + sumdf["PT成交價"]
        sumdf = sumdf[sumdf["CTPT差"] == sumdf["CTPT差"].min()]
        sumdf = sumdf[sumdf["CTPT和"] == sumdf["CTPT和"].min()]

        result = sumdf[sumdf["CTPT差"] == sumdf["CTPT差"].min()][["CT成交價","PT成交價"]].values.sum()
        sumdf = sumdf.reset_index()
        centeridx = sumdf[sumdf["CTPT差"] == sumdf["CTPT差"].min()].index[0]
        cn = sumdf[sumdf["CTPT差"] == sumdf["CTPT差"].min()]['履約價'].values[0]+300
        pn = sumdf[sumdf["CTPT差"] == sumdf["CTPT差"].min()]['履約價'].values[0]-300
        result = sumdf[sumdf["CTPT差"] == sumdf["CTPT差"].min()][["CT成交價","PT成交價"]].values.sum()
        
        result = sumdf[sumdf["CTPT差"] == sumdf["CTPT差"].min()][["CT成交價","PT成交價"]].values.sum()
        putcallsum_month = pd.concat([putcallsum_month,pd.DataFrame([[querydate,result]],columns = ["日期","月選擇權價平和"])])
        
        print(querydate,result)
    except:
        if (datetime.today()- timedelta(days=i)).weekday() not in [5,6]:
            print(querydate,'error')
        continue
    try:
        result = sumdf[sumdf["CTPT差"] == sumdf["CTPT差"].min()][["CT成交價","PT成交價"]].values.sum()
        result2 = round(CT[CT["履約價"] == cn]["CT成交價"].values[0] / PT[PT["履約價"] == pn]["PT成交價"].values[0],3) - 1
        putcallgap_month = pd.concat([putcallgap_month,pd.DataFrame([[querydate,result2]],columns = ["日期","價外買賣權價差"])])
        print(querydate,result2)

    except:
        try:
            cn = cn - 50
            result2 = round(CT[CT["履約價"] == cn]["CT成交價"].values[0] / PT[PT["履約價"] == pn]["PT成交價"].values[0],3) - 1
            putcallgap_month = pd.concat([putcallgap_month,pd.DataFrame([[querydate,result2]],columns = ["日期","價外買賣權價差"])])
            print(querydate,result2)
        except:
            try:
                pn = pn + 50
                cn = cn + 50
                result2 = round(CT[CT["履約價"] == cn]["CT成交價"].values[0] / PT[PT["履約價"] == pn]["PT成交價"].values[0],3) - 1
                putcallgap_month = pd.concat([putcallgap_month,pd.DataFrame([[querydate,result2]],columns = ["日期","價外買賣權價差"])])
                print(querydate,result2)
            except:
                try:
                    cn = cn - 50
                    result2 = round(CT[CT["履約價"] == cn]["CT成交價"].values[0] / PT[PT["履約價"] == pn]["PT成交價"].values[0],3) - 1
                    putcallgap_month = pd.concat([putcallgap_month,pd.DataFrame([[querydate,result2]],columns = ["日期","價外買賣權價差"])])
                    print(querydate,result2)
                except:
                    if (datetime.today()- timedelta(days=i)).weekday() not in [5,6]:
                        print(querydate,'error')
                    continue
    
    
    #putcallsum = pd.concat([putcallsum,pd.DataFrame([[querydate,result]],columns = ["日期","價平和"])])
putcallgap_month.to_sql('putcallgap_month', connection, if_exists='replace', index=False) 
print(putcallgap_month.tail())

putcallsum_month.to_sql('putcallsum_month', connection, if_exists='replace', index=False) 
print(putcallsum_month.tail())

#connection.executemany('replace INTO bank VALUES (?, ?, ?)', np.array(bank8))     
connection.close()
