#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
print("read in all player data")
data = pd.read_csv("all_players.csv")
data.head()


# In[2]:

print("read in atp player data")
players_data = pd.read_csv("atp_players.csv",header=0,names=['id','name','surname','hand','dob','country','height','wikidata_id'])
players_data['dob'] = pd.to_datetime(players_data['dob'], format='%Y%m%d', errors='coerce')



# In[3]:


print(players_data.head())


# In[4]:


len(data)


# In[ ]:

print("add underscore to name")
data.loc[data['player_id'].str.contains('_', na=False), 'player_id'] = 'No'
print("done adding underscore")


# In[6]:


data = data.loc[data['player_id'] != "No"]


# In[7]:


cleaned_players_data = data
print(cleaned_players_data.head())


# In[8]:


players_data['player_id'] = " "
players_data['player_id'] = players_data['name'].replace(" ","-") + "-" + players_data['surname'].replace(' ','-')
players_data['player_id'] = players_data['player_id'].str.lower()


# In[9]:


players_data['player_id']


# In[10]:


players_data


# In[11]:


cleaned_players_data


# In[12]:


final_players = cleaned_players_data.merge(players_data,left_on='player_id',right_on='player_id',how='outer')


# In[13]:


final_players = final_players.loc[(final_players['dob'].isna()) == False]
final_players['dob'] = pd.to_datetime(final_players['dob'], errors='coerce')



# In[14]:


final_players


# In[15]:


#reset index
final_players.reset_index(drop=True)


# In[19]:


final_players = final_players.reset_index(drop=True)


# In[20]:


final_players


# In[21]:


#drop useless col
final_players=final_players.drop(['country_x','id','name','surname'],axis=1)


# In[25]:


print(final_players.head())


# In[26]:


final_players.to_csv('players_data.csv', index=False)
final_players.to_pickle('players_data.pkl')


# In[1]:


#load players data
import pandas as pd
players_data = pd.read_csv('players_data.csv', parse_dates=True)
players_data


# In[8]:

print("replacing space in player_id")
players_data['player_id'] = players_data['player_id'].str.replace(' ', '-')


# In[9]:


players_data


# In[11]:


players_data.to_csv('players_data.csv', index=False)


# In[ ]:




