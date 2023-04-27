#!/usr/bin/env python

#Imports
import pandas as pd
from helper_functions import print_elapsed_time

# Load matches data
print_elapsed_time('Importing match data')
matches_dataset_final = pd.read_csv('final_kaggle_dataset.csv', parse_dates=True)

# Load players data
print_elapsed_time('Importing player data')
players_dataset = pd.read_csv('players_data.csv', parse_dates=True)
players_dataset['dob'] = pd.to_datetime(players_dataset['dob'], errors='coerce')

# Merge matches dataset with players dataset
print_elapsed_time('Merging match data with player data')
players_in_matches = pd.merge(matches_dataset_final[['player_id']], players_dataset, on='player_id')

# Remove players born before 1960
print_elapsed_time('Removing players born before 1960')
players_born_after_1960 = players_in_matches[players_in_matches['dob'].dt.year > 1959]

# Save data
print_elapsed_time('Saving player data')
players_born_after_1960.to_csv('players_data.csv', index=False)