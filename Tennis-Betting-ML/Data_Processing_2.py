#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
original_dir = os.getcwd()
print(original_dir)
os.chdir('./Tournaments_Data')
print(os.getcwd())
files = os.listdir(cwd)


# In[2]:


#concatenate all tournament csv's
import pandas as pd
final_df = pd.read_csv('aachen_challenger2000.csv', index_col=0)
for file in files:
    if str(file) == 'aachen_challenger2000.csv':
        continue
    temp = pd.read_csv(str(file), index_col=0)
    final_df = pd.concat([final_df,temp])
    temp = None
    


# In[3]:


final_df

# In[4]:


#turn the start_date and end_date columns into date format
# yy-mm-dd is the format used in the dataset
final_df.head()
final_df['start_date'] = pd.to_datetime(final_df['start_date'], format = '%Y-%m-%d')
final_df.head()


# In[5]:


final_df.iloc[500]['start_date']


# In[6]:


final_df.loc[final_df['start_date'] == '2/21/2011']


# In[7]:

os.chdir(original_dir)
final_df.to_csv('final_df.csv',index=False)


# In[8]:


final_df_2 = pd.read_csv('final_df.csv',parse_dates = [3,4],infer_datetime_format=True)


# In[9]:


final_df_2.iloc[777]['end_date']


# In[10]:


final_df_2 =  final_df_2.sort_values(by='start_date')


# In[11]:

final_df_2.to_csv('final_df.csv',index=False)