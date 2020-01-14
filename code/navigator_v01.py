# %%
import pymysql
import datetime

conn = pymysql.connect(host='localhost', user='root', password='mysql_rootuser', db='market_data',charset='utf8mb4')
curs = conn.cursor(pymysql.cursors.DictCursor)
#curs = conn.cursor()

sql = 'select Date from lktb order by Date DESC LIMIT 1'
curs.execute(sql)
row = curs.fetchall()

conn.close()

end_date_db = row[0]['Date'].date()
start_date_eikon = end_date_db + datetime.timedelta(days=1) # eikon 조회용 start_date

date_today = datetime.datetime.now().date()

# eikon 조회용 end_date : 장중 여부에 따라 -1 day
if(datetime.datetime.now().time() < datetime.time(16,10,00)):
    end_date_eikon = date_today
else:
    end_date_eikon = date_today + datetime.timedelta(days=1)

# %%
import eikon as ek
import pandas as pd
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()

import MySQLdb


ek.set_app_key('8ee143628de84818a8b12f4f55be35674e136d08')

if (end_date_eikon > start_date_eikon):
    try:
        df_eikon = ek.get_timeseries(["10TBc1"],
                                     fields=['OPEN','HIGH','LOW','CLOSE','VOLUME'],
                                     start_date = start_date_eikon.isoformat(),
                                     end_date = end_date_eikon.isoformat(),
                                     interval = "minute"
                                    )
        #df_local = df_eikon.tz_localize('UTC').tz_convert('Asia/Seoul')
        df_eikon.index += pd.offsets.Hour(9)
        db_data = 'mysql://root:' + 'mysql_rootuser' + '@localhost/market_data'
    #    db_data = 'mysql+mysqldb://root:' + 'mysql_rootuser' + '@localhost/market_data'
        engine = create_engine(db_data)
        conn = engine.connect()
    #    conn = pymysql.connect(host='localhost', user='root', password='mysql_rootuser', db='market_data',charset='utf8mb4')
    #    curs = conn.cursor()
        df_eikon.to_sql('lktb', engine, if_exists='append')
        conn.close()
        engine.dispose()
    except:
        pass
    
# %%
conn = pymysql.connect(host='localhost', user='root', password='mysql_rootuser', db='market_data',charset='utf8mb4')
curs = conn.cursor(pymysql.cursors.DictCursor)
#curs = conn.cursor()

sql = 'select Date from lktb order by Date DESC LIMIT 1'
curs.execute(sql)
row = curs.fetchall()

conn.close()

end_date_db = row[0]['Date'].date()
daycount_match = 2 # 매칭할 대상 기간 : 2 days
daycount_forecast = 1 # 전망할 날짜 수 :  1 day
resample_minutes = 10 # 가격주기 : 10 mins

# %%
import pandas as pd
from sqlalchemy import create_engine

db_data = 'mysql://root:' + 'mysql_rootuser' + '@localhost/market_data'
engine = create_engine(db_data)
conn = engine.connect()
df = pd.read_sql_table('lktb', conn, index_col='Date')
conn.close()
engine.dispose()
# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math
import datetime


# %% resample
df = df.drop(['OPENINTEREST'], axis=1).resample(str(resample_minutes) + 'T', closed='right').agg({'OPEN': 'first',
                        'HIGH': 'max',
                        'LOW': 'min',
                        'CLOSE': 'last',
                        'VOLUME': 'sum'}).dropna()


# In[14]: 날짜 list 만들기
list_day = df.resample('D') \
    .last() \
        .dropna() \
            .drop(['OPEN', 'HIGH', 'LOW', 'VOLUME'], axis=1) \
                .reset_index()['Date'] \
                    .dt.date \
                        .to_numpy().tolist()

# 매칭할 날짜 세팅
# input_end_date = list_day[-1] # data중 마지막 날짜로 세팅하는 경우

# %% 날짜 list에서 패턴 매칭대상 start_date, end_date 가져오기
x_start_date = list_day[(list_day.index(end_date_db) - daycount_match + 1)].isoformat()
x_end_date = end_date_db.isoformat()

# %% df에서 패턴매칭대상 가격Moves 가져오기
np_moves_target = df.loc[x_start_date:x_end_date]['CLOSE'].to_numpy()
np_moves_target_z = stats.zscore(np_moves_target)

# %% data df를 days_matching 개수로 분리
list_start_date = []
list_end_date = []

for i in range(len(list_day)-daycount_match + 1):
    if list_day[i + daycount_match - 1] < end_date_db:
        list_start_date.append(list_day[i])
        list_end_date.append(list_day[i + daycount_match - 1])

# %% window 별 moves 분리
list_moves = []
for i in range(len(list_start_date)):
# for i in range(10):
    list_moves.append(df.loc[list_start_date[i].isoformat():list_end_date[i].isoformat()]['CLOSE'].to_numpy())
    
    
# %% dtw 구하기
list_dtw = []

from dtw import accelerated_dtw
from tqdm import tqdm

for np_moves_data in tqdm(list_moves):
    d, _, _, _ = accelerated_dtw(np_moves_target_z, stats.zscore(np_moves_data), 'euclidean')
    list_dtw.append(d)
    
    
# %% rank 구하기
list_rank = stats.rankdata(list_dtw).tolist()

# %% result dataframe 만들기
df_result = pd.DataFrame(list(zip(list_start_date, list_end_date, list_dtw, list_rank)),
                         columns = ['start', 'end', 'dtw', 'rank'])

# %% rank 순서대로 plot
max_rank = 10

f1 = plt.figure(figsize=(15, 10))
plt.grid(b=True)

minutes_per_day = 405
points_per_day = math.ceil(minutes_per_day / resample_minutes)
points_before = points_per_day * daycount_match
points_after = points_per_day * daycount_forecast

x = np.linspace(-points_before + 1,points_after, points_before + points_after )

for rank in range(1,max_rank+1):
    index = list_rank.index(rank)
    dtw = list_dtw[index]
    moves_before = df.loc[list_start_date[index].isoformat():list_end_date[index].isoformat()]['CLOSE'].to_numpy()
    moves_after = df.loc[list_day[list_day.index(list_end_date[index]) + 1].isoformat():
                         list_day[min(list_day.index(list_end_date[index]) + daycount_forecast, len(list_day) - 1)].isoformat()]['CLOSE'].to_numpy()
    pivot_price = moves_before[-1]
    moves_before_scaled = moves_before - pivot_price
    moves_after_scaled = moves_after - pivot_price
    moves = np.concatenate((
        [np.nan]*(points_before - len(moves_before_scaled)),
        moves_before_scaled,
        moves_after_scaled,
        [np.nan]*(points_after - len(moves_after_scaled))
        ))
    plt.plot(x,moves,  label=(list_end_date[index].isoformat(),rank, round(dtw,1)))
#plt.legend(loc='best')
#plt.show()
# %%

# target moves plotting
pivot_price_target = np_moves_target[-1]
moves_target_before_scaled = np_moves_target - pivot_price_target

#list_day.index(end_date_db)


# %%

#moves_target_after = df.loc[list_day[list_day.index(input_end_date) + 1].isoformat():
#                     list_day[min(list_day.index(list_end_date[index]) + input_days_after, len(list_day) - 1)].isoformat()]['CLOSE'].to_numpy()

#list_day.index(input_end_date)


# eikon 현재data retrieve
import eikon as ek

ek.set_app_key('8ee143628de84818a8b12f4f55be35674e136d08')
try:
    start_date_eikon = end_date_db + datetime.timedelta(days=1) # eikon 조회용 start_date
    end_date_eikon = start_date_eikon + datetime.timedelta(days=1) # eikon 조회용 start_date
    df_eikon = ek.get_timeseries(["10TBc1"], 
                           start_date = start_date_eikon,
                           end_date = end_date_eikon,
                           interval = "minute"
                           )
except:
    moves_target_after_scaled = []
else:
    df_eikon = df_eikon.resample(str(resample_minutes) + 'T', closed='right').agg({'OPEN': 'first',
                            'HIGH': 'max',
                            'LOW': 'min',
                            'CLOSE': 'last',
                            'COUNT' : 'sum',
                            'VOLUME': 'sum'}).dropna()
    np_moves_target_after = df_eikon['CLOSE'].to_numpy()
    moves_target_after_scaled = np_moves_target_after - pivot_price_target

moves_target_scaled = np.concatenate((
    [np.nan]*(points_before-len(moves_target_before_scaled)),
    moves_target_before_scaled,
    moves_target_after_scaled,
    [np.nan]*(points_after-len(moves_target_after_scaled))
    ))

plt.plot(x, moves_target_scaled,'r-', label=x_end_date, linewidth=3.0)
plt.legend(loc=2)

# 추가할것 : 변동성 이동평균 반
# %%

'''
index = list_rank.index(1)
moves_before = df.loc[list_start_date[index].isoformat():list_end_date[index].isoformat()]['Close'].to_numpy()
moves_after = df.loc[list_day[list_day.index(list_end_date[index])+1].isoformat()]['Close'].to_numpy()

pivot_price = moves_before[-1]
moves_before_scaled = 



# %% plot_overlay_pivot function
def plot_overlay_pivot(df_data, list_start_date, list_end_date, list_label, input_days_after ):
    list_daily = df_data.resample('D').last().dropna() \
        .drop(['Open', 'High', 'Low', 'Volume', 'OpenInterest'], axis=1) \
            .reset_index()['Datetime'].dt.date.to_numpy().tolist()
    
    for 


# In[19]:


from scipy import stats

list_day_df_data = [[group[0], group[1]['CLOSE'].to_numpy(),
                     stats.zscore(group[1]['CLOSE'].to_numpy())] for group in df_data.groupby(['DATE'])]
list_day_df_2019 = [[group[0], group[1]['CLOSE'].to_numpy(),
                     stats.zscore(group[1]['CLOSE'].to_numpy())] for group in df_2019.groupby(['DATE'])]


# In[20]:


list_day_df_2019[-3]


# In[21]:


from dtw import accelerated_dtw

x = list_day_df_2019[-1][2]
#x = stats.zscore(df_10m.to_numpy())  # for eikon

#dist_dtw = []
for daily_data in list_day_df_data:
    d, _, _, _ = accelerated_dtw(x, daily_data[2], 'euclidean')
    #dist_dtw.append(d)
    daily_data.append(d)
x - 1
#dist_dtw


# In[22]:


df = pd.DataFrame(list_day_df_data, columns=['Date','Moves','Moves_zscore','DTW'])
#df_sorted = df.set_index('Date')


# In[23]:


df['Rank'] = df['DTW'].rank(method='min')


# In[24]:


df.head()


# In[25]:


max_rank = 20
f1 = plt.figure(figsize=(15, 10))
for rank in range(1,max_rank+1):
    close_price = df[df['Rank']==rank]['Moves'].values[0][-1]
    array_before = (df[df['Rank']==rank]['Moves'].values[0] - close_price)
    array_after = df.loc[df[df['Rank']==rank].index+1]['Moves'].values[0] - close_price
    date = df[df['Rank']==rank]['Date'].values[0]
    plt.plot(np.concatenate(
        ([np.nan]*(41-len(array_before)),
         array_before,
         array_after,
         [np.nan]*(41-len(array_after))
        )),label=date)

x2 = list_day_df_2019[-1][1]
#x2 = df_10m.to_numpy() # for eikon
close_price_x = x2[-1]
array_x = x2 - close_price_x
plt.plot(np.concatenate(([np.nan]*(41-len(array_x)),array_x)),'ro-')
#np.concatenate([np.nan]*(41-len(array_x)),array_x)
#print(np.concatenate([np.nan]*(41-len(array_x)),array_x))
plt.legend(loc=2)

# In[2]:

df_infomax = pd.read_csv("../data/LKTB_infomax_1m.csv")
df_infomax['DATETIME'] = pd.to_datetime(
        df_infomax['DATE'] + ' ' + df_infomax['TIME'])
df = df_infomax.drop(['DATE', 'TIME'], axis=1)

df = df.set_index('DATETIME')


'''