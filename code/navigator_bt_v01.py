# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math

# %% 기준일과 매칭할 날짜 수 setting
daycount_match = 2 # 매칭할 대상 기간 : 2 days
daycount_forecast = 1 # 전망할 날짜 수 :  1 day
resample_minutes = 10 # 가격주기 : 10 mins

date_split = '2019-01-01'

# %% 1min 단위의 raw data 받아오기
df_infomax = pd.read_csv("../data/LKTB_infomax_1m.csv")
df_infomax['DATETIME'] = pd.to_datetime(df_infomax['DATE'] + ' ' + df_infomax['TIME'])
df = df_infomax.drop(['DATE', 'TIME'], axis=1)
df = df.set_index('DATETIME')

# %% resample
df = df.resample(str(resample_minutes) + 'T', closed='right').agg({
    'OPEN': 'first',
    'HIGH': 'max',
    'LOW': 'min',
    'CLOSE': 'last',
    'VOLUME': 'sum',
    'OPENINTEREST': 'last'
    }).dropna()

# %% df 2019와 이전으로 나누기
df_data = df[:date_split]
df_match = df[date_split:'2019-06-30']

# In[14]: 날짜 list 만들기
list_dates_data = df_data.resample('D') \
    .last() \
        .dropna() \
            .drop(['OPEN', 'HIGH', 'LOW', 'VOLUME', 'OPENINTEREST'], axis=1) \
                .reset_index()['DATETIME'] \
                    .dt.date \
                        .to_numpy().tolist()

list_dates_match = df_match.resample('D') \
    .last() \
        .dropna() \
            .drop(['OPEN', 'HIGH', 'LOW', 'VOLUME', 'OPENINTEREST'], axis=1) \
                .reset_index()['DATETIME'] \
                    .dt.date \
                        .to_numpy().tolist()

# %%
list_data_date1 = []
list_data_date2 = []
list_data_date3 = []
list_data_date4 = []
for i in range(len(list_dates_data) - daycount_match - daycount_forecast + 1 ):
    list_data_date1.append(list_dates_data[i])
    list_data_date2.append(list_dates_data[i + daycount_match - 1] )
    list_data_date3.append(list_dates_data[i + daycount_match] )
    list_data_date4.append(list_dates_data[i + daycount_match + daycount_forecast - 1] )

list_match_date1 = []
list_match_date2 = []
list_match_date3 = []
list_match_date4 = []
for i in range(len(list_dates_match) - daycount_match - daycount_forecast + 1):
    list_match_date1.append(list_dates_match[i])
    list_match_date2.append(list_dates_match[i + daycount_match - 1])
    list_match_date3.append(list_dates_match[i + daycount_match] )
    list_match_date4.append(list_dates_match[i + daycount_match + daycount_forecast - 1] )
# %% window 별 moves 분리
list_data_moves = []
list_match_moves = []
list_price_chg_actual = []
for i in range(len(list_data_date1)):
    list_data_moves.append(df_data.loc[list_data_date1[i].isoformat():list_data_date2[i].isoformat()]['CLOSE'].to_numpy())
for i in range(len(list_match_date1)):
    list_match_moves.append(df_match.loc[list_match_date1[i].isoformat():list_match_date2[i].isoformat()]['CLOSE'].to_numpy())
    list_price_chg_actual.append(df_match.loc[list_match_date4[i].isoformat()]['CLOSE'].to_numpy()[-1] - list_match_moves[i][-1])
    
# %% for문 돌며 dtw 계산
#list_dtw = []

from dtw import accelerated_dtw
from tqdm import tqdm
max_rank = 10
list_price_chg_forecast = []

for np_moves_match in tqdm(list_match_moves): # match_moves의 각 하루치 일자 당 반복하며
    list_dtw = []
    price_chg_forecast = 0
    for np_moves_data in (list_data_moves):
        d, _, _, _ = accelerated_dtw(stats.zscore(np_moves_match), stats.zscore(np_moves_data), 'euclidean')
        list_dtw.append(d)
        list_rank = stats.rankdata(list_dtw).tolist()
    
    sum_chg_mult_weight = 0
    sum_weight = 0    
    for rank in range(1,max_rank+1):
        index = list_rank.index(rank)
        weight = 1/list_dtw[index]
        #date3 = list_data_date3[index]
        date4 = list_data_date4[index]
        pivot_price = list_data_moves[index][-1]
        close_price = df_data.loc[date4.isoformat()]['CLOSE'].to_numpy()[-1]
        price_chg = close_price - pivot_price
        sum_chg_mult_weight += price_chg * weight
        sum_weight += weight
    
    price_chg_forecast = sum_chg_mult_weight / sum_weight
    list_price_chg_forecast.append(price_chg_forecast)

# %% result dataframe 만들기
df_result = pd.DataFrame(list(zip(list_match_date1,
                                  list_match_date2,
                                  list_match_date3,
                                  list_match_date4,
                                  list_price_chg_forecast,
                                  list_price_chg_actual)
                              ),
                         columns = ['date1', 'date2', 'date3', 'date4', 'forecast', 'actual'])