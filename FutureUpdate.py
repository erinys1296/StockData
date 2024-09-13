import requests
from bs4 import BeautifulSoup
import pandas as pd


#日期處理
from datetime import datetime, timedelta

#例外處理
from time import sleep

#資料庫處理
import sqlite3

import shioaji as sj
import APIKEY

connection = sqlite3.connect('FutureData.sqlite3')

SIM_MODE = True

api = sj.Shioaji(simulation=SIM_MODE)  #建立shioaji api物件，simulation=True代表使用模擬環境 

#登入
api.login(
    api_key= APIKEY.get_Key(SIM_MODE),     # 從APIKEY讀取
    secret_key= APIKEY.get_Secret(SIM_MODE)   # 從APIKEY讀取
)

def resample_df(original_df, frequency):
    df_resample = original_df.resample(frequency)

    df = pd.DataFrame()
    df['Open'] = df_resample['Open'].first()
    df['Low'] = df_resample['Low'].min()
    df['Volume'] = df_resample['Volume'].sum()
    df['Close'] = df_resample['Close'].last()
    df['High'] = df_resample['High'].max()

    return df

def get_future_raw_data(start, end):
    deadline = api.Contracts.Futures.TXF.TXFR1
    k_bars = api.kbars(api.Contracts.Futures['TXF'][deadline.symbol], start=start, end=end)
    df = pd.DataFrame({**k_bars})
    df.ts = pd.to_datetime(df.ts)
    df.sort_values(["ts"], ascending=True, inplace=True)
    df.set_index('ts', inplace=True)

    return resample_df(df, 'T')


FutureData = pd.read_sql("select distinct * from futurehourly", connection, parse_dates=['ts'], index_col=['ts'])


startFuture = datetime.strftime(FutureData.index.max(),'%Y-%m-%d')
endFuture = datetime.strftime(datetime.today(),'%Y-%m-%d')

FutureDataNew = get_future_raw_data(startFuture,endFuture)
FutureDataNew = FutureDataNew.reset_index()
FutureData = FutureData.reset_index()
FutureData = pd.concat([FutureData,FutureDataNew])
FutureData = FutureData.drop_duplicates()

FutureData.to_sql('futurehourly', connection, if_exists='replace', index=False)

print('updated')
