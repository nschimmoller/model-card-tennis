#!/usr/bin/env python

import pandas as pd
from helper_functions import print_elapsed_time

print("Reading in all player data")
print_elapsed_time()
data = pd.read_csv("all_players.csv")

print("Reading in atp player_data")
print_elapsed_time()
players_data = pd.read_csv("atp_players.csv",header=0,names=['id','name','surname','hand','dob','country','height','wikidata_id'])
players_data['dob'] = pd.to_datetime(players_data['dob'], format='%Y%m%d', errors='coerce')

print("Add underscore to name")
print_elapsed_time()
data.loc[data['player_id'].str.contains('_', na=False), 'player_id'] = 'No'

print("Filter out missing player ids")
print_elapsed_time()
data = data.loc[data['player_id'] != "No"]


cleaned_players_data = data

print("Clean player name")
print_elapsed_time()
players_data['player_id'] = " "
players_data['player_id'] = players_data['name'].replace(" ","-") + "-" + players_data['surname'].replace(' ','-')
players_data['player_id'] = players_data['player_id'].str.lower()

print("Join clean and original player data")
print_elapsed_time()
final_players = cleaned_players_data.merge(players_data,left_on='player_id',right_on='player_id',how='outer')

print("Remove players with missing date of birth (DOB), and format as date")
print_elapsed_time()
final_players = final_players.loc[(final_players['dob'].isna()) == False]
final_players['dob'] = pd.to_datetime(final_players['dob'], errors='coerce')

#reset index
final_players = final_players.reset_index(drop=True)

#drop useless col
final_players=final_players.drop(['country_x','id','name','surname'],axis=1)

print("Export data to players_data.csv and players_data.pkl")
print_elapsed_time()
final_players.to_csv('players_data.csv', index=False)
final_players.to_pickle('players_data.pkl')