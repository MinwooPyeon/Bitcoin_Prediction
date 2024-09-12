#!/usr/bin/env python
# coding: utf-8

# In[34]:


pip install xgboost

# In[28]:


pip install mplfinance

# In[1]:


import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from xgboost import XGBRegressor

# In[2]:


# 일봉 조회
df = pd.read_csv('BTC-USD.csv')

# 'Date' 열을 datetime 형태로 변환
df['Date'] = pd.to_datetime(df['Date'])

# 'Date' 열을 인덱스로 설정
df.set_index('Date', inplace=True)

print(df)

# In[3]:


df.head()

# In[4]:


mpf.plot(df, type='line')

# In[5]:


mpf.plot(df, volume = True, type='line')

# - 2021년에 거래량 및 가격이 압도적으로 높음

# In[6]:


Op=np.array(df["Open"])
Hi=np.array(df["High"])
Lo=np.array(df["Low"])
AC=np.array(df["Adj Close"])
V=np.array(df["Volume"])

# - 정확한 예측을 위해 종가가 아닌 수정 종가를 반영

# In[7]:


input=np.column_stack((Op,Hi,Lo,V))
output=AC

# In[8]:


# input 중 op,hi,lo는 가격을 의미하고 v는 거래량을 의미해 다른 사이즈를 맞추는 작업
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
input_scaled=SS.fit_transform(input)

# In[9]:


from sklearn.model_selection import train_test_split
train_input,test_input,train_output,test_output=train_test_split(input_scaled,output)

# In[10]:


from xgboost import XGBRegressor, plot_importance

XGBR=XGBRegressor(max_depth=3, reg_alpha=0.1)
XGBR.fit(train_input,train_output)

# In[11]:


plot_importance(XGBR).set_yticklabels(['Open','High','Low','Volume'])

# In[12]:


plt.plot(XGBR.predict(test_input),'w')
plt.plot(test_output,'b')

# - 파란색 그래프가 예측값
# - 흰색이 실제값

# In[13]:


XGBR.score(test_input,test_output)
# 모델평가 99.9점

# In[ ]:



