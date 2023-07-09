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
table = soup.find("table", {"class": "table_c"}) 

# 將表格數據轉換成Pandas數據框
datedf = pd.read_html(str(table))[0]

#處理欄位空格
newcol = [stri.replace(' ','') for stri in datedf.columns]
datedf.columns = newcol
datedf.columns = ['最後結算日', '契約月份', '臺指選擇權（TXO）', '電子選擇權（TEO）', '金融選擇權（TFO）']
print(datedf)
datedf.to_sql('end_date', connection, if_exists='append', index=False) 


ordervolumn = pd.read_sql("select distinct * from ordervolumn", connection)
putcallsum = pd.read_sql("select 日期, max(價平和) as 價平和 from putcallsum group by 日期", connection)

#test = crawler.catch_cost('20230601')
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

maxtime = datetime.strptime(putcallsum["日期"].max(), '%Y/%m/%d')

#價平和
for i in range((datetime.today() - maxtime).days+10):
    querydate = datetime.strftime(datetime.today()- timedelta(days=i),'%Y/%m/%d')
    #print(querydate)
    try:
        CT,PT = crawler.callputtable(querydate)
    except:
        continue
    CT.columns = ["履約價","CT成交價"]
    PT.columns = ["履約價","PT成交價"]
    sumdf = CT.join(PT.set_index("履約價"),on=["履約價"],lsuffix='_left', rsuffix='_right')
    sumdf["CTPT差"] = np.abs(sumdf["CT成交價"] - sumdf["PT成交價"])
    result = sumdf[sumdf["CTPT差"] == sumdf["CTPT差"].min()][["CT成交價","PT成交價"]].values.sum()
    putcallsum = pd.concat([putcallsum,pd.DataFrame([[querydate,result]],columns = ["日期","價平和"])])

putcallsum.to_sql('putcallsum', connection, if_exists='replace', index=False) 


# 將結算日的爬蟲寫到 function外 (因為不會隨著時間改變而改變，減少爬蟲次數)
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



connection.close()
