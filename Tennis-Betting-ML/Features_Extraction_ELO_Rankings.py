#!/usr/bin/env python
# coding: utf-8

# In[7]:


print("this is working")
#initialize elo ranking - set E_i(0) = 1500 in other words set initial elo ranking to 1500 for all players
#get players list
import pandas as pd
import time

def print_elapsed_time():
    current_time = time.time()
    if not hasattr(print_elapsed_time, 'last_time'):
        print("Elapsed time since start:", current_time)
    else:
        print("Elapsed time since last call:", current_time - print_elapsed_time.last_time)
    print_elapsed_time.last_time = current_time

players_list = pd.read_csv('players_data_1.csv')
players_list = players_list['player_id']
players_list = players_list.to_list()
players_list


# In[8]:


#load matches data
matches_df = pd.read_csv('final_df.csv', parse_dates=True)


# In[41]:


# Get a unique list of all players in the DataFrame
players_list = pd.unique(matches_df[['player_id', 'opponent_id']].values.ravel())
print(players_list[:5])

print("Grouping dataframe")
print_elapsed_time()

# Group the DataFrame by player_id and opponent_id columns and sort each group by start_date
grouped = matches_df.groupby(['player_id', 'opponent_id'], group_keys=False, sort=False).apply(lambda x: x.sort_values(by='start_date'))
grouped.to_csv('grouped.csv')

print("Finished grouping dataframe, saved csv")
print_elapsed_time()

print("Initializing Elo Rankings for players in player_list")
print_elapsed_time()

# Initialize elo rankings for each player based on their earliest matches
index_1 = []
index_2 = []
for player in players_list:
    print(player)
    temp = grouped.loc[player]
    index = temp.index[0]
    if temp.iloc[0]['player_id'] == str(player):
        index_1.append(index)
    if temp.iloc[0]['opponent_id'] == str(player):
        index_2.append(index)
print("Finished Initializing Elo Rankings for players in player_list")
print_elapsed_time()

# In[63]:


len(index_1) + len(index_2) == len(players_list)


# In[57]:


#Set initial elo's to 1500
matches_df.loc[index_1, 'elo_1'] = 1500
matches_df.loc[index_2, 'elo_2'] = 1500


# In[64]:


matches_df.loc[(matches_df['elo_1'] == 1500) | (matches_df['elo_2']==1500)]


# In[1]:


#checkpoint: save df to csv
matches_df.to_csv('final_df.csv')


# In[1]:


#start from checkpoint
matches_df = pd.read_csv('final_df.csv')


# In[20]:


#check for missing values in the player_1_victory and player_2_victory columns, these columns are essential so every row
#with missing values has to be removed

matches_df[matches_df['player_2_victory'].isna()].index


# In[21]:


#removing these rows
index_remove = matches_df[matches_df['player_2_victory'].isna()].index
matches_df.drop(index=index_remove,inplace=True)


# In[23]:


#check again
matches_df[matches_df['player_1_victory'].isna()].index


# In[24]:


#calculations of elo are done on a separate copy of matches_df then transferred with each iteration to matches_df
copy =  matches_df[['player_id', 'opponent_id', 'start_date', 'player_1_victory', 'player_2_victory', 'elo_1', 'elo_2']]

# In[28]:

print("Calculating games played in player's career")
print_elapsed_time()

#we pre-calculate the m's , m is the number of games played by a player in his career
def get_m(player_id, data):
    temp = data.loc[(data['player_id'].isin([player_id])) | (data['opponent_id'].isin([player_id]))]
    m = len(temp)
    return m

# create an empty dictionary to store m values
m_dict = {}

for i in range(1, len(copy)):
    # get player_id and opponent_id for current row
    player_id = copy.iloc[i]['player_id']
    opponent_id = copy.iloc[i]['opponent_id']
    
    # check if m value for player_id is already in m_dict
    if player_id in m_dict:
        m_1.append(m_dict[player_id])
    else:
        # calculate m value for player_id
        m_value = get_m(player_id, copy)
        m_dict[player_id] = m_value
        m_1.append(m_value)
        
    # check if m value for opponent_id is already in m_dict
    if opponent_id in m_dict:
        m_2.append(m_dict[opponent_id])
    else:
        # calculate m value for opponent_id
        m_value = get_m(opponent_id, copy)
        m_dict[opponent_id] = m_value
        m_2.append(m_value)

print("Finished calculating games played in player's career")
print_elapsed_time()

# In[41]:


m_1.insert(0,0)
m_2.insert(0,0)


# In[42]:


copy['m_1'] = m_1
copy['m_2'] = m_2
copy


# In[43]:


#save copy
copy.to_csv('auxiliary_df.csv')


# In[2]:


#checkpoint
copy = pd.read_csv('auxiliary_df.csv')
matches_df = pd.read_csv('final_df.csv')

# In[27]:


#create steps list - incremental list of integers
steps = [50,100,150,200,250,500,750,1000,1250,1500,2000,2500,3000,4000,5000,10000,20000,30000,40000,50000,75000,100000,125000,150000]


# In[5]:


#second search function, the first one was too slow, let's try a different approach

def search_rows2(start_row, player_id, data):
    cache_key = (start_row, player_id)
    if cache_key in search_cache:
        return search_cache[cache_key]
    
    m, last_elo, victory = None, None, None
    for step in steps:
        j = max(start_row - step, 0)
        temp = data.loc[((data['player_id'] == player_id) & (data['elo_1'] != 0))
                        | ((data['opponent_id'] == player_id) & (data['elo_2'] != 0))
                        , ['m_1', 'elo_1', 'm_2', 'elo_2', 'player_1_victory', 'player_2_victory']
                        ].iloc[j:start_row + 1]
        if len(temp) == 0:
            continue
        row = temp.iloc[-1]
        if row['player_id'] == player_id:
            m = row['m_1']
            last_elo = row['elo_1']
            victory = row['player_1_victory']
        else:
            m = row['m_2']
            last_elo = row['elo_2']
            victory = row['player_2_victory']
        if victory == 't':
            victory = 1
        elif victory == 'f':
            victory = 0
        else:
            victory = None
        if victory is not None:
            break

    result = m, last_elo, victory
    search_cache[cache_key] = result
    return result


# In[6]:


import numpy as np

def elo_calc(elo_a,elo_b,m_a,w_a):
    expected_pi = 1.0/(1+10**((elo_b - elo_a)/400))
    decay = 250.0/((5+m_a)**0.4)
    updated_elo_a = elo_a + decay*(w_a - expected_pi)
    return updated_elo_a


# In[28]:


#compute elo rankings
#from 0 to 20000 using first function search_rows
#from 20 000 using second function search_rows2
print("Calculating ELO rankings")
print_elapsed_time()

# Keep track of the rows that have been modified
modified_rows = []

# Loop over the rows in the dataframe
for i in range(len(matches_df)):
    # Check if either player's rating is 0
    if matches_df.iloc[i]['elo_1'] == 0 or matches_df.iloc[i]['elo_2'] == 0:
        # Compute the Elo ratings for both players
        player_1_id = matches_df.iloc[i]['player_id']
        player_2_id = matches_df.iloc[i]['opponent_id']
        m_1, elo_1, w_1 = search_rows2(i, player_1_id, matches_df)
        m_2, elo_2, w_2 = search_rows2(i, player_2_id, matches_df)
        if matches_df.iloc[i]['elo_1'] == 0:
            matches_df.at[i, 'elo_1'] = elo_calc(elo_1, elo_2, m_1, w_1)
        if matches_df.iloc[i]['elo_2'] == 0:
            matches_df.at[i, 'elo_2'] = elo_calc(elo_2, elo_1, m_2, w_2)
        modified_rows.append(i)
    
    # If 100 rows have been modified, update the copy of the dataframe
    if len(modified_rows) >= 100:
        copy = matches_df.copy()
        copy.loc[modified_rows, ['elo_1', 'elo_2']] = matches_df.loc[modified_rows, ['elo_1', 'elo_2']]
        modified_rows = []

    # Print progress every 1000 rows
    if i % 1000 == 0:
        print(f"Processed {i} rows")
    
# Update the final copy of the dataframe
if len(modified_rows) > 0:
    copy = matches_df.copy()
    copy.loc[modified_rows, ['elo_1', 'elo_2']] = matches_df.loc[modified_rows, ['elo_1', 'elo_2']]


print("Calculating ELO rankings")
print_elapsed_time()

# In[1]:


#save matches_df and copy
copy.to_csv('auxiliary_df.csv')
matches_df.to_csv('final_df.csv')


# In[3]:

print("carpet to grass")
#turn carpet into grass
for i in range(0,len(matches_df)):
    if matches_df.iloc[i]['court_surface'] == 'Carpet':
        matches_df.at[i,'court_surface'] = 'Grass'
pd.unique(matches_df['court_surface'])


# In[45]:


#elo by surface: we have three surfaces clay,hard,grass(merged with carpet because they are the same thing)

#CLAY

#create copy df with only clay matches
import numpy as np
copy_surface = pd.DataFrame()
copy_surface['player_id'] = matches_df['player_id']
copy_surface['opponent_id'] = matches_df['opponent_id']
copy_surface['start_date'] = matches_df['start_date']
copy_surface['surface'] = matches_df['court_surface']
copy_surface['player_1_victory'] = matches_df['player_1_victory']
copy_surface['player_2_victory'] = matches_df['player_2_victory']

copy_clay = copy_surface.copy(deep=True)
copy_clay = copy_clay.loc[copy_clay['surface']=='Clay']
copy_clay['elo_1_clay'] = 0
copy_clay['elo_2_clay'] = 0
copy_clay.reset_index(inplace=True)
#initialize elo_clay to 1500

#load players list
players_list = pd.read_csv('players_data_1.csv')
players_list = players_list['player_id']
players_list = players_list.to_list()
players_list
#indexes initialize
index_1_clay = []
index_2_clay = []

print("calculate elo ranking for clay")
for item in players_list:
    temp = copy_clay.loc[ (copy_clay['player_id'] == item) | (copy_clay['opponent_id'] == item)]
    if len(temp) == 0:
        continue
    temp = temp.sort_values(by='start_date')
    index = temp.index[0]
    if temp.iloc[0]['player_id'] == str(item):
        copy_clay.at[index,'elo_1_clay'] = 1500
    if temp.iloc[0]['opponent_id'] == str(item):
        copy_clay.at[index,'elo_2_clay'] = 1500
    temp = None
    index = None

#Set initial elo's to 1500

#HARD

copy_hard = copy_surface.copy(deep=True)
copy_hard = copy_hard.loc[copy_hard['surface']=='Hard']
copy_hard['elo_1_hard'] = 0
copy_hard['elo_2_hard'] = 0
copy_hard.reset_index(inplace=True)
index_1_hard = []
index_2_hard = []

print("calculate elo ranking for hard")
for item in players_list:
    temp = copy_hard.loc[ (copy_hard['player_id'] == item) | (copy_hard['opponent_id'] == item)]
    if len(temp) == 0:
        continue
    temp = temp.sort_values(by='start_date')
    index = temp.index[0]
    if temp.iloc[0]['player_id'] == str(item):
        copy_hard.at[index,'elo_1_hard'] = 1500
    if temp.iloc[0]['opponent_id'] == str(item):
        copy_hard.at[index,'elo_2_hard'] = 1500
    temp = None
    index = None
    
#GRASS

copy_grass = copy_surface.copy(deep=True)
copy_grass = copy_grass.loc[copy_grass['surface']=='Grass']
copy_grass['elo_1_grass'] = 0
copy_grass['elo_2_grass'] = 0
copy_grass.reset_index(inplace=True)
index_1_grass = []
index_2_grass = []
 
print("calculate elo ranking for grass")
for item in players_list:
    temp = copy_grass.loc[ (copy_grass['player_id'] == item) | (copy_grass['opponent_id'] == item)]
    if len(temp) == 0:
        continue
    temp = temp.sort_values(by='start_date')
    index = temp.index[0]
    if temp.iloc[0]['player_id'] == str(item):
        copy_grass.at[index,'elo_1_grass'] = 1500
    if temp.iloc[0]['opponent_id'] == str(item):
        copy_grass.at[index,'elo_2_grass'] = 1500
    temp = None
    index = None
    


# In[82]:


#let's see the results, we test for some players to see if the elo was initiated correctly
copy_clay.loc[(copy_clay['player_id']=='roger-federer')|(copy_clay['opponent_id']=='roger-federer')]


# In[83]:


copy_hard.loc[(copy_hard['player_id']=='roger-federer')|(copy_hard['opponent_id']=='roger-federer')]


# In[84]:


copy_grass.loc[(copy_grass['player_id']=='roger-federer')|(copy_grass['opponent_id']=='roger-federer')]


# In[49]:


len(copy_clay) + len(copy_hard) + len(copy_grass)


# In[28]:


len(matches_df)


# In[50]:


copy_clay


# In[51]:


copy_hard


# In[52]:


copy_grass


# In[53]:


#get surface specific m's
m_1_clay = []
m_2_clay = []
m_1_hard = []
m_2_hard = []
m_1_grass = []
m_2_grass = []
# get_m is the same function used before
def get_m(player_id,data,row):
    temp = data.iloc[:row]
    temp = temp.loc[(temp['player_id'] == player_id) | (temp['opponent_id']==player_id)]
    m = len(temp)
    return m

for i in range(1,len(copy_clay)):
    m_1_clay.append(get_m(copy_clay.iloc[i]['player_id'],copy_clay,i))
    m_2_clay.append(get_m(copy_clay.iloc[i]['opponent_id'],copy_clay,i))
    print(i)

m_1_clay.insert(0,0)
m_2_clay.insert(0,0)


# In[55]:


# get surface specific m's for Hard surface
print("get matches played on hard")
for i in range(1,len(copy_hard)):
    m_1_hard.append(get_m(copy_hard.iloc[i]['player_id'],copy_hard,i))
    m_2_hard.append(get_m(copy_hard.iloc[i]['opponent_id'],copy_hard,i))
    print(i)
m_1_hard.insert(0,0)
m_2_hard.insert(0,0)


# In[56]:


#get surface specific m's for Grass surface
print("get matches played on grass")
for i in range(1,len(copy_grass)):
    m_1_grass.append(get_m(copy_grass.iloc[i]['player_id'],copy_grass,i))
    m_2_grass.append(get_m(copy_grass.iloc[i]['opponent_id'],copy_grass,i))
    print(i)
    
m_1_grass.insert(0,0)
m_2_grass.insert(0,0)


# In[79]:


#transfer m's to each surface specific dataset
copy_clay['m_1_clay'] = m_1_clay
copy_clay['m_2_clay'] = m_2_clay
copy_hard['m_1_hard'] = m_1_hard
copy_hard['m_2_hard'] = m_2_hard
copy_grass['m_1_grass'] = m_1_grass
copy_grass['m_2_grass'] = m_2_grass


# In[58]:


#calculate the elo's of each dataset
#re-adapted search function

steps = [50,100,150,200,250,500,750,1000,1250,1500,2000,2500,3000,4000,5000,10000,20000,30000,40000,50000,75000,100000,125000,150000]

def search_rows2_clay(start_row,player_id,data):
    k = start_row
    m = None
    last_elo = None
    victory = None
    for step in steps:
        j = k - step
        if j < 0 :
            j = 0
        temp = data.iloc[j:k+1]
        temp = temp.loc[((temp['player_id']==player_id)&(temp['elo_1_clay'] !=0))|((temp['opponent_id']==player_id)&(temp['elo_2_clay']!=0))]
        if len(temp) == 0:
            continue
        if temp.iloc[-1]['player_id'] == player_id:
            m = temp.iloc[-1]['m_1_clay']
            last_elo = temp.iloc[-1]['elo_1_clay']
            victory = temp.iloc[-1]['player_1_victory']
        if temp.iloc[-1]['opponent_id'] == player_id:
            m = temp.iloc[-1]['m_2_clay']
            last_elo = temp.iloc[-1]['elo_2_clay']
            victory = temp.iloc[-1]['player_2_victory']
        if victory == 't':
            victory = 1
        if victory == 'f':
            victory = 0
        break
        print(m,last_elo,victory)
    return m,last_elo,victory


# In[59]:


#re-adapted search function for hard surface
def search_rows2_hard(start_row,player_id,data):
    k = start_row
    m = None
    last_elo = None
    victory = None
    for step in steps:
        j = k - step
        if j < 0 :
            j = 0
        temp = data.iloc[j:k+1]
        temp = temp.loc[((temp['player_id']==player_id)&(temp['elo_1_hard'] !=0))|((temp['opponent_id']==player_id)&(temp['elo_2_hard']!=0))]
        if len(temp) == 0:
            continue
        if temp.iloc[-1]['player_id'] == player_id:
            m = temp.iloc[-1]['m_1_hard']
            last_elo = temp.iloc[-1]['elo_1_hard']
            victory = temp.iloc[-1]['player_1_victory']
        if temp.iloc[-1]['opponent_id'] == player_id:
            m = temp.iloc[-1]['m_2_hard']
            last_elo = temp.iloc[-1]['elo_2_hard']
            victory = temp.iloc[-1]['player_2_victory']
        if victory == 't':
            victory = 1
        if victory == 'f':
            victory = 0
        break
        print(m,last_elo,victory)
    return m,last_elo,victory


# In[60]:


#re-adapted search function for grass
def search_rows2_grass(start_row,player_id,data):
    k = start_row
    m = None
    last_elo = None
    victory = None
    for step in steps:
        j = k - step
        if j < 0 :
            j = 0
        temp = data.iloc[j:k+1]
        temp = temp.loc[((temp['player_id']==player_id)&(temp['elo_1_grass'] !=0))|((temp['opponent_id']==player_id)&(temp['elo_2_grass']!=0))]
        if len(temp) == 0:
            continue
        if temp.iloc[-1]['player_id'] == player_id:
            m = temp.iloc[-1]['m_1_grass']
            last_elo = temp.iloc[-1]['elo_1_grass']
            victory = temp.iloc[-1]['player_1_victory']
        if temp.iloc[-1]['opponent_id'] == player_id:
            m = temp.iloc[-1]['m_2_grass']
            last_elo = temp.iloc[-1]['elo_2_grass']
            victory = temp.iloc[-1]['player_2_victory']
        if victory == 't':
            victory = 1
        if victory == 'f':
            victory = 0
        break
        print(m,last_elo,victory)
    return m,last_elo,victory


# In[66]:


matches_df['elo_1_clay'] = 0
matches_df['elo_2_clay'] = 0
matches_df['elo_1_hard'] = 0
matches_df['elo_2_hard'] = 0
matches_df['elo_1_grass'] = 0
matches_df['elo_2_grass'] = 0

for i in range(0,len(copy_clay)):
    if copy_clay.iloc[i]['elo_1_clay'] == 1500:
        matches_df.at[copy_clay.iloc[i]['index'],'elo_1_clay'] = 1500
    if copy_clay.iloc[i]['elo_2_clay'] == 1500:
        matches_df.at[copy_clay.iloc[i]['index'],'elo_2_clay'] = 1500
        
for i in range(0,len(copy_hard)):
    if copy_hard.iloc[i]['elo_1_hard'] == 1500:
        matches_df.at[copy_hard.iloc[i]['index'],'elo_1_hard'] = 1500
    if copy_hard.iloc[i]['elo_2_hard'] == 1500:
        matches_df.at[copy_hard.iloc[i]['index'],'elo_2_hard'] = 1500

for i in range(0,len(copy_grass)):
    if copy_grass.iloc[i]['elo_1_grass'] == 1500:
        matches_df.at[copy_grass.iloc[i]['index'],'elo_1_grass'] = 1500
    if copy_grass.iloc[i]['elo_2_grass'] == 1500:
        matches_df.at[copy_grass.iloc[i]['index'],'elo_2_grass'] = 1500


# In[85]:


matches_df.loc[matches_df['court_surface']=='Clay']


# In[74]:


import numpy as np

def elo_calc(elo_a,elo_b,m_a,w_a):
    expected_pi = 1.0/(1+10**((elo_b - elo_a)/400))
    decay = 250.0/((5+m_a)**0.4)
    updated_elo_a = elo_a + decay*(w_a - expected_pi)
    return updated_elo_a


# In[115]:


#elo clay calculations
print("calculate clay elo")
for i in range(1,len(copy_clay)):
    elo = None
    player_id = None
    if copy_clay.iloc[i]['elo_1_clay'] == 0:
        #compute elo
        player_id = copy_clay.iloc[i]['player_id']
        m_1,elo_1,w_1 = search_rows2_clay(i,player_id,copy_clay)
        opponent = copy_clay.iloc[i]['opponent_id']
        m_2,elo_2,w_2 = search_rows2_clay(i,opponent,copy_clay)
        elo = elo_calc(elo_1,elo_2,m_1,w_1)
        copy_clay.at[i,'elo_1_clay'] = elo
        elo = None
        player_id = None
        m_1 = None
        opponent = None
        m_2 = None
        w_2 = None
        elo_2 = None
        elo_1 = None
        w_1 = None
        
    if copy_clay.iloc[i]['elo_2_clay'] == 0:
        player_id = copy_clay.iloc[i]['opponent_id']
        m_2,elo_2,w_2 = search_rows2_clay(i,player_id,copy_clay)
        opponent = copy_clay.iloc[i]['player_id']
        m_1,elo_1,w_1 = search_rows2_clay(i,opponent,copy_clay)
        elo = elo_calc(elo_2,elo_1,m_2,w_2)
        copy_clay.at[i,'elo_2_clay'] = elo
        elo = None
        player_id = None
        m_1 = None
        opponent = None
        m_2 = None
        w_2 = None
        elo_2 = None
        elo_1 = None
        w_1 = None
    print(i)


# In[112]:


copy_clay = copy_clay.fillna(0)


# In[116]:


copy_clay


# In[117]:

print("calculate hard elo")
#elo calc for Hard surface
for i in range(0,len(copy_hard)):
    elo = None
    player_id = None
    if copy_hard.iloc[i]['elo_1_hard'] == 0:
        #compute elo
        player_id = copy_hard.iloc[i]['player_id']
        m_1,elo_1,w_1 = search_rows2_hard(i,player_id,copy_hard)
        opponent = copy_hard.iloc[i]['opponent_id']
        m_2,elo_2,w_2 = search_rows2_hard(i,opponent,copy_hard)
        elo = elo_calc(elo_1,elo_2,m_1,w_1)
        copy_hard.at[i,'elo_1_hard'] = elo
        elo = None
        player_id = None
        m_1 = None
        opponent = None
        m_2 = None
        w_2 = None
        elo_2 = None
        elo_1 = None
        w_1 = None
        
    if copy_hard.iloc[i]['elo_2_hard'] == 0:
        player_id = copy_hard.iloc[i]['opponent_id']
        m_2,elo_2,w_2 = search_rows2_hard(i,player_id,copy_hard)
        opponent = copy_hard.iloc[i]['player_id']
        m_1,elo_1,w_1 = search_rows2_hard(i,opponent,copy_hard)
        elo = elo_calc(elo_2,elo_1,m_2,w_2)
        copy_hard.at[i,'elo_2_hard'] = elo
        elo = None
        player_id = None
        m_1 = None
        opponent = None
        m_2 = None
        w_2 = None
        elo_2 = None
        elo_1 = None
        w_1 = None
    print(i)


# In[118]:

print("calculate grass elo")
#elo calc for grass surface
for i in range(0,len(copy_grass)):
    elo = None
    player_id = None
    if copy_grass.iloc[i]['elo_1_grass'] == 0:
        #compute elo
        player_id = copy_grass.iloc[i]['player_id']
        m_1,elo_1,w_1 = search_rows2_grass(i,player_id,copy_grass)
        opponent = copy_grass.iloc[i]['opponent_id']
        m_2,elo_2,w_2 = search_rows2_grass(i,opponent,copy_grass)
        elo = elo_calc(elo_1,elo_2,m_1,w_1)
        copy_grass.at[i,'elo_1_grass'] = elo
        elo = None
        player_id = None
        m_1 = None
        opponent = None
        m_2 = None
        w_2 = None
        elo_2 = None
        elo_1 = None
        w_1 = None
        
    if copy_grass.iloc[i]['elo_2_grass'] == 0:
        player_id = copy_grass.iloc[i]['opponent_id']
        m_2,elo_2,w_2 = search_rows2_grass(i,player_id,copy_grass)
        opponent = copy_grass.iloc[i]['player_id']
        m_1,elo_1,w_1 = search_rows2_grass(i,opponent,copy_grass)
        elo = elo_calc(elo_2,elo_1,m_2,w_2)
        copy_grass.at[i,'elo_2_grass'] = elo
        elo = None
        player_id = None
        m_1 = None
        opponent = None
        m_2 = None
        w_2 = None
        elo_2 = None
        elo_1 = None
        w_1 = None
    print(i)


# In[119]:


#UPDATE matches_df with elo_surface values cumputed above

print("update df")
for i in range(0,len(copy_clay)):
    matches_df.at[copy_clay.iloc[i]['index'],'elo_1_clay'] = copy_clay.iloc[i]['elo_1_clay']
    matches_df.at[copy_clay.iloc[i]['index'],'elo_2_clay'] = copy_clay.iloc[i]['elo_2_clay']
    
for i in range(0,len(copy_hard)):
    matches_df.at[copy_hard.iloc[i]['index'],'elo_1_hard'] = copy_hard.iloc[i]['elo_1_hard']
    matches_df.at[copy_hard.iloc[i]['index'],'elo_2_hard'] = copy_hard.iloc[i]['elo_2_hard']
    
for i in range(0,len(copy_grass)):
    matches_df.at[copy_grass.iloc[i]['index'],'elo_1_grass'] = copy_grass.iloc[i]['elo_1_grass']
    matches_df.at[copy_grass.iloc[i]['index'],'elo_2_grass'] = copy_grass.iloc[i]['elo_2_grass']


# In[120]:


copy_clay.to_csv('clay.csv')
copy_hard.to_csv('hard.csv')
copy_grass.to_csv('grass.csv')
matches_df.to_csv('final_df.csv')


# In[124]:


matches_df.loc[matches_df['court_surface']=='Grass']


# In[125]:


# now merge the clay.hard and grass columns into one columns elo_1_surface
#same thing for elo_2_surface
elo_1_surface = matches_df['elo_1_clay'] + matches_df['elo_1_hard'] + matches_df['elo_1_grass']


# In[126]:


elo_1_surface


# In[127]:


elo_2_surface = matches_df['elo_2_clay'] + matches_df['elo_2_hard'] + matches_df['elo_2_grass']


# In[128]:


elo_2_surface


# In[129]:


matches_df['elo_1_surface'] = elo_1_surface
matches_df['elo_2_surface'] = elo_2_surface
matches_df.to_csv('final_df.csv')

