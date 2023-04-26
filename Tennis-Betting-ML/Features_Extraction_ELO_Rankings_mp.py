#!/usr/bin/env python

#initialize elo ranking - set E_i(0) = 1500 in other words set initial elo ranking to 1500 for all players
#get players list
import pandas as pd
import os
from helper_functions import print_elapsed_time
import numpy as np
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count
import functools
import pickle
from tqdm import tqdm
import logging
import time
import sys

# players_list = pd.read_csv('players_data_1.csv')
# players_list = players_list['player_id']
# players_list = players_list.to_list()
# players_list

# #load matches data
# matches_df = pd.read_csv('final_df.csv', parse_dates=True)

# # Get a unique list of all players in the DataFrame
# players_list = pd.unique(matches_df[['player_id', 'opponent_id']].values.ravel())

# # Group the DataFrame by player_id and opponent_id columns and sort each group by start_date
# print("\nGrouping dataframe")
# print_elapsed_time()

# grouped = matches_df.groupby(['player_id', 'opponent_id'], group_keys=False, sort=False).apply(lambda x: x.sort_values(by='start_date'))
# grouped.to_csv('grouped.csv')

# print("\nFinished grouping dataframe, saved csv")
# print_elapsed_time()

# print("\nInitializing Elo Rankings for players in player_list")
# print_elapsed_time()

# # Initialize elo rankings for each player based on their earliest matches
# index_1 = []
# index_2 = []
# for player_id in players_list:
#     player_matches = grouped.loc[(grouped['player_id'] == str(player_id)) | (grouped['opponent_id'] == str(player_id))]
#     earliest_match = player_matches.sort_values('start_date').iloc[0]
#     if earliest_match['player_id'] == str(player_id):
#         index_1.append(earliest_match.name)
#     else:
#         index_2.append(earliest_match.name)

# print("\nFinished Initializing Elo Rankings for players in player_list")
# print_elapsed_time()

# #Set initial elo's to 1500
# len(index_1) + len(index_2) == len(players_list)
# matches_df.loc[index_1, 'elo_1'] = 1500
# matches_df.loc[index_2, 'elo_2'] = 1500

# print(matches_df.head())

# #checkpoint: save df to csv
# print("\nSaving dataframe as a checkpoint")
# print_elapsed_time()
# matches_df.to_csv('final_df.csv', index=False)

# #start from checkpoint
# print("\nPick up from checkpoint")
# print_elapsed_time()
# matches_df = pd.read_csv('final_df.csv', parse_dates=True)

# #check for missing values in the player_1_victory and player_2_victory columns, these columns are essential so every row
# #with missing values has to be removed
# print("\nRemove matches with missing winner information")
# print_elapsed_time()
# matches_df = matches_df.loc[matches_df['player_2_victory'].notna()]

# #calculations of elo are done on a separate copy of matches_df then transferred with each iteration to matches_df
# copy =  matches_df[['player_id', 'opponent_id', 'start_date', 'player_1_victory', 'player_2_victory', 'elo_1', 'elo_2']]

# # In[28]:

# print("\nCalculating games played in player's career")
# print_elapsed_time()

# def get_matches(player_id, data):
#     """Calculate the number of matches played by a player in their career.

#     Args:
#         player_id (int or str): The ID of the player.
#         data (pandas.DataFrame): A DataFrame containing match data.

#     Returns:
#         int: The number of matches played by the player.
#     """
#     matches = data.loc[(data['player_id'] == player_id) | (data['opponent_id'] == player_id)].groupby(['player_id', 'opponent_id']).size().sum()
#     return matches


# # create an empty dictionary to store m values
# m_dict = {}

# # calculate m values for each player and opponent in the DataFrame
# for index, row in copy.iterrows():
#     player_id = row['player_id']
#     opponent_id = row['opponent_id']
#     if player_id not in m_dict:
#         m_dict[player_id] = get_matches(player_id, copy)
#     if opponent_id not in m_dict:
#         m_dict[opponent_id] = get_matches(opponent_id, copy)

# # use the m_dict to populate m_1 and m_2
# m_1 = [m_dict[row['player_id']] for index, row in copy.iterrows()]
# m_2 = [m_dict[row['opponent_id']] for index, row in copy.iterrows()]


# print("\nFinished calculating games played in player's career")
# print_elapsed_time()

# #save copy
# print("\nSaving matches played dataframe as CSV as checkpoint")
# print_elapsed_time()
# copy['m_1'] = m_1
# copy['m_2'] = m_2
# copy.to_csv('auxiliary_df.csv', index=False)

copy = pd.read_csv('auxiliary_df.csv', parse_dates=True)
copy = copy[:500]

# Create a list of step sizes to use in the Elo calculation
# steps = [50, 100, 150, 200, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000, 125000, 150000, 200000]

# Create a logger object and add a file handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('my_log_file.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def search_rows(start_row: int, player_id: str, data: pd.DataFrame, steps: List[int], starting_elo: float = 1500) -> Tuple[float, int]:
    """
    Searches the rows of the data frame starting from the given start row to find previous matches
    played by the player with the given player_id. Calculates the player's Elo rating based on their
    performance in those matches. This function uses a memoized version of the search_rows function
    to store and reuse results to improve performance.

    Args:
        start_row (int): The index of the row to start the search from.
        player_id (str): The ID of the player to search for.
        data (pandas.DataFrame): The data frame containing the matches data.
        steps (List[int]): A list of integers representing the step sizes to use in the Elo calculation.
        starting_elo (float): The starting Elo rating to use for players without an existing rating.

    Returns:
        Tuple[float, int]: A tuple containing the player's Elo rating and the number of matches used to calculate it.
    """
    print("Searched Rows function called successfuly")
    data_hash = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)  # Compute a hashable representation of the data
    return search_rows_memoized(start_row, player_id, data_hash, tuple(steps), starting_elo)

@functools.lru_cache(maxsize=10000)
def search_rows_memoized(start_row: int, player_id: str, data_hash: bytes, steps: Tuple[int], starting_elo: float = 1500) -> Tuple[float, int]:
    """
    Memoized version of the search_rows function. Searches the rows of the data frame starting from the given start row
    to find previous matches played by the player with the given player_id. Calculates the player's Elo rating based on
    their performance in those matches. This function is memoized using functools.lru_cache to store results and improve
    performance.

    Args:
        start_row (int): The index of the row to start the search from.
        player_id (str): The ID of the player to search for.
        data_hash (bytes): The hashable representation of the data frame containing the matches data.
        steps (Tuple[int]): A tuple of integers representing the step sizes to use in the Elo calculation.
        starting_elo (float): The starting Elo rating to use for players without an existing rating.

    Returns:
        Tuple[float, int]: A tuple containing the player's Elo rating and the number of matches used to calculate it.
    """
    print("Searched Rows Memoized function called successfuly")
    data = pickle.loads(data_hash)  # Reconstruct the data frame from the hash
    m, last_elo, victory = None, None, None

    for step in steps:
        print("Step wise search function called successfuly")
        j = max(start_row - step, 0)
        step_bin = data.loc[((data['player_id'] == player_id) & (data['elo_1'] != 0)) | ((data['opponent_id'] == player_id) & (data['elo_2'] != 0)), ['player_id', 'opponent_id', 'm_1', 'elo_1', 'm_2', 'elo_2', 'player_1_victory', 'player_2_victory']].iloc[j:start_row + 1]

        if len(step_bin) == 0:
            continue

        row= step_bin.iloc[-1]

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

    result = m, last_elo, victory
    return result

def update_last_processed_row(file_name: str, row: int) -> None:
    """
    Update the last processed row in a file.

    Args:
        file_name (str): The name of the file where the last processed row value should be stored.
        row (int): The value of the last processed row to store in the file.
    """
    print("Updating last processed row")
    with open(file_path, 'w') as f:
        f.write(str(last_processed_row))

def read_last_processed_row(file_name: str) -> int:
    """
    Read the last processed row from a file.

    Args:
        file_name (str): The name of the file containing the last processed row value.

    Returns:
        int: The last processed row value, or 0 if the file does not exist or the content is not a valid integer.
    """
    print("Reading last processed row")
    try:
        with open(file_path, 'r') as f:
            last_processed_row = int(f.read())
    except FileNotFoundError:
        last_processed_row = -1
    return last_processed_row

def process_chunk(args):
    """
    Process a chunk of the dataframe to compute Elo ratings using the compute_elo_ratings function.

    Args:
        args (tuple): A tuple containing:
            chunk (pandas.DataFrame): A chunk of the matches dataframe to process.
            steps (int): Number of iterations to use when computing the Elo rating.

    Returns:
        pandas.DataFrame: A DataFrame with updated Elo ratings for the provided chunk.
    """
    chunk = args
    try:
        logging.info("Processing the dataframe chunk")
        return compute_elo_ratings(chunk)
    except Exception as e:
        logging.exception(f"Error processing chunk: {e}")

    
def divide_dataframe(df, n_chunks):
    """
    Divide a given dataframe into a specified number of smaller chunks.

    Args:
        df (pandas.DataFrame): The dataframe to divide into chunks.
        n_chunks (int): The number of chunks to divide the dataframe into.

    Returns:
        List[pandas.DataFrame]: A list of smaller dataframes (chunks) derived from the original dataframe.
    """
    chunk_size = len(df) // n_chunks
    print("Dividing dataframe into " + str(n_chunks) + " of size " + str(chunk_size))
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    return chunks

# Define the function to calculate the Elo rating
def elo_calc(elo_1: float, elo_2: float, w_1: int, w_2: int, m_1: int, m_2: int, initial_elo=1500) -> Tuple[float, float]:
    """
    Calculates the updated Elo rating of a player.

    Args:
        elo_1 (float): The current Elo rating of the player.
        elo_2 (float): The Elo rating of the opponent.
        m_1 (int): The number of matches played by the player.
        m_2 (int): The number of matches played by the opponent.
        w_1 (int): The number of victories achieved by the player.
        w_2 (int): The number of victories achieved by the opponent.

    Returns:
        float: The updated Elo rating of the player and opponent.
    """
    print("Calculating ELO")
    if elo_1 is None:
        elo_1 = initial_elo
    if elo_2 is None:
        elo_2 = initial_elo
    expected_1 = 1 / (1 + 10 ** ((elo_2 - elo_1) / 400))
    expected_2 = 1 / (1 + 10 ** ((elo_1 - elo_2) / 400))
    decay_1 = 250.0 / ((5 + m_1) ** 0.4)
    decay_2 = 250.0 / ((5 + m_2) ** 0.4)
    new_elo_1 = elo_1 + decay_1 * (w_1 - expected_1)
    new_elo_2 = elo_2 + decay_2 * (w_2 - expected_2)

    return new_elo_1, new_elo_2

def compute_elo_ratings(matches_df, steps=100, update_last_processed_row=None, last_processed_row_file=None, update_interval=1000, output_file='elo.csv', temp_output_file='elo_temp.csv'):
    """
    Compute Elo ratings for each player based on their match history.

    Args:
    - matches_df: Pandas DataFrame containing match data, including columns for player_id, opponent_id, 
      elo_1, elo_2, and date.
    - steps: Number of iterations to use when computing the Elo rating.
    - search_cache: Optional dict of cached search results, to speed up computation.
    - update_last_processed_row: Optional function to update the last processed row in a file.
    - last_processed_row_file: Optional file path to the file storing the last processed row.
    - update_interval: Interval at which to update the output and temp files.
    - output_file: Output file for the final Elo ratings.
    - temp_output_file: Temporary output file for storing Elo ratings during computation.

    Returns:
    - A copy of matches_df with updated Elo ratings.
    """
    # Create a copy of the dataframe to store the updated Elo ratings
    updated_df = matches_df.copy()

    player_ids = set(matches_df['player_id']).union(set(matches_df['opponent_id']))
    player_elo = {pid: 1500 for pid in player_ids}

    if update_last_processed_row is not None:
        last_processed_row = read_last_processed_row(last_processed_row_file)
        updated_df = updated_df.iloc[last_processed_row + 1:]
        print("Resuming from " + str(last_processed_row))

    for i, row in enumerate(updated_df.iterrows()):
        player_1_id = row[1]['player_id']
        player_2_id = row[1]['opponent_id']
        elo_1 = player_elo.get(player_1_id, 1500)
        elo_2 = player_elo.get(player_2_id, 1500)
        m_1, _, w_1 = search_rows(i, player_1_id, updated_df, steps)
        m_2, _, w_2 = search_rows(i, player_2_id, updated_df, steps)
        player_elo[player_1_id], player_elo[player_2_id] = elo_calc(elo_1, elo_2, w_1, w_2, m_1, m_2)
        updated_df.at[i, 'elo_1'] = player_elo[player_1_id]
        updated_df.at[i, 'elo_2'] = player_elo[player_2_id]

        if update_last_processed_row is not None and i % update_interval == 0:
            print("THIS SHOULD UPDATE LAST PROCSSED ROW")
            update_last_processed_row(i, last_processed_row_file)
            updated_df.to_csv(temp_output_file, index=False)

    if update_last_processed_row is not None:
        update_last_processed_row(len(updated_df) - 1, last_processed_row_file)

    updated_df.to_csv(output_file, index=False)

    return updated_df

if __name__ == "__main__":
    print("\nReading in files from checkpoint")
    print_elapsed_time()

    # Load the matches data
    matches_df = pd.read_csv('final_df.csv', parse_dates=True)

    cwd = os.getcwd()
    last_processed_row_file = 'last_processed_row.txt'
    file_path = os.path.join(cwd, last_processed_row_file)
    n_processes = cpu_count()
    n_chunks = n_processes * 2  # You can adjust this value depending on your system
    print(n_chunks)
    n_chunks = 10
    print(n_chunks)

    # Read the last processed row
    last_processed_row = read_last_processed_row(last_processed_row_file)
    if last_processed_row is not None:
        print(f"Resuming from last processed row: {last_processed_row}")
        copy = copy.loc[last_processed_row + 1:]

    matches_cache = {}
    chunks = divide_dataframe(copy, n_chunks)

    try:
        with Pool(n_processes) as pool:
            # Wrap the iterable with tqdm to display progress
            results = list(tqdm(pool.imap(process_chunk, [chunk for chunk in chunks]), total=n_chunks))
    except IndexError as e:
        logging.exception('Caught an exception', exc_info=e)

    # Combine the results
    matches_df = pd.concat(results)

    # Save the combined dataframe
    print(matches_df.head())
    matches_df.to_csv('elo.csv', index=False)

    # Reset the last processed row
    update_last_processed_row(-1)



# matches_cache = {}
# matches_df = compute_elo_ratings(copy, steps, matches_cache)
# print(matches_df.head())

# print("\nELO Calculations complete")
# print_elapsed_time()

# #save matches_df and copy
# print("\nCreate checpoint CSVs")
# print_elapsed_time()
# copy.to_csv('auxiliary_df.csv', index=False)
# matches_df.to_csv('elo.csv', index=False)

sys.exit(0)


# Convert court_surface, carpet, to grass
print("\nConvert court surface carpet to grass")
print_elapsed_time()
matches_df.loc[matches_df['court_surface'] == 'Carpet', 'court_surface'] = 'Grass'

# Create a dictionary to store the data for each surface
data_dict = {}

print("\nCreate surface specific dataframes")
print_elapsed_time()
# Loop through the surfaces
for surface in matches_df['court_surface'].unique():
    
    # Create a copy of the matches_df for the current surface
    copy_surface = matches_df[matches_df['court_surface'] == surface][['player_id', 'opponent_id', 'start_date', 'court_surface', 'player_1_victory', 'player_2_victory']].copy(deep=True)
    
    # Add columns for the Elo ranking
    copy_surface['elo_1'] = 1500
    copy_surface['elo_2'] = 1500
    
    # Add the copy to the data dictionary
    data_dict[surface] = copy_surface

print("\nCalculate surface specific ELO")
print_elapsed_time()
# Access the data for the clay surface, for example:
for surface, surface_df in data_dict.items():
    print("Starting Calculation for " + surface)
    surface_cache = {}
    surface_df = compute_elo_ratings(surface_df, steps, surface_cache, print_progress=False)
    data_dict[surface] = surface_df
    print("\nELO Calculation completed for " + surface)
    print_elapsed_time()

sys.exit(0)

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

