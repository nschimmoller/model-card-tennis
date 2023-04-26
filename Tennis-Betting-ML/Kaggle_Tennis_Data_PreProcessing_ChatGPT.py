#!/usr/bin/env python

import pandas as pd
import os
import sys

from helper_functions import setup_logger

# Set up the logger
logger = setup_logger(__name__, 'my_log_file.log')
logger.info('Starting script execution')

try:
    from helper_functions import print_elapsed_time

    # Step 1: Read in data
    print_elapsed_time('Reading in data')
    all_matches = pd.read_csv("all_matches.csv", parse_dates=True)
    tournaments = pd.read_csv("all_tournaments.csv", parse_dates=True)

    # Create unique match identifiers
    all_matches['match_id_player_1'] = all_matches['player_id'] + '_' + all_matches['opponent_id'] + '_' + all_matches['tournament'] + '_' + all_matches['start_date']
    all_matches['match_id_player_2'] = all_matches['opponent_id'] + '_' + all_matches['player_id'] + '_' + all_matches['tournament'] + '_' + all_matches['start_date']

    # Step 2: Remove tournaments & matches prior to the year 2000
    print_elapsed_time('Removing old tournaments')
    tournaments = tournaments[tournaments['year'] >= 2000]
    all_matches = all_matches[all_matches['year'] >= 2000]

    # Step 3: Create a unique identifier column in the tournaments dataframe
    print_elapsed_time('Creating unique tournament_ids')
    tournaments['tournament_id'] = tournaments['tournament'] + '_' + tournaments['year'].astype(str)
    tournaments['tournament_id'] = tournaments['tournament_id'].str.replace('/', '-')
    all_matches['tournament_id'] = all_matches['tournament'] + '_' + all_matches['year'].astype(str)
    all_matches['tournament_id'] = all_matches['tournament_id'].str.replace('/', '-')

    # Step 4: Remove doubles matches from the data dataframe
    print_elapsed_time('Removing Doubles Matches')
    all_matches = all_matches[all_matches['doubles'] != 't']

    # Step 5: Remove any match where num_sets is null / N/A / nan etc.
    print_elapsed_time('Removing erroneous matches')
    all_matches = all_matches[all_matches['num_sets'].notna()]

    #Split the dataset into two DataFrames, one for each player
    player_1_matches = all_matches.add_suffix('_1')
    player_2_matches = all_matches.add_suffix('_2')

    # Merge the renamed DataFrames
    print_elapsed_time('Perform self join')
    final_dataset = player_1_matches.merge(player_2_matches, left_on='match_id_player_1_1', right_on='match_id_player_2_2')

    # Drop the _y columns and rename the _x columns
    print_elapsed_time('Rename columns with suffixes')
    shared_columns = ['player_id', 'opponent_id', 'start_date', 'court_surface', 'currency', 'end_date', 'location', 'num_sets', 'opponent_name', 'player_name', 'prize_money', 'round', 'tournament', 'year', 'duration', 'nation', 'match_id_player_1', 'match_id_player_2', 'tournament_id']
    for col in final_dataset.columns:
        if col.endswith('_1'):
            if col[:-2] in shared_columns:
                final_dataset.drop(columns=[col], inplace=True)
        elif col.endswith('_2'):
            if col[:-2] in shared_columns:
                final_dataset.rename(columns={col: col[:-2]}, inplace=True)

    # Reorder columns
    print_elapsed_time('Reorder columns')
    no_suffix_columns = [col for col in final_dataset.columns if not col.endswith('_1') and not col.endswith('_2')]
    suffix_1_columns = [col for col in final_dataset.columns if col.endswith('_1')]
    suffix_2_columns = [col for col in final_dataset.columns if col.endswith('_2')]
    ordered_columns = no_suffix_columns + suffix_1_columns + suffix_2_columns
    final_dataset = final_dataset[ordered_columns]

    #Saving dataframe as CSV
    print_elapsed_time('Save dataframe as CSV')
    final_dataset.to_csv('final_kaggle_dataset.csv', index=False)

    #Saving dataframe as PKL
    print_elapsed_time('Save dataframe as pickle')
    final_dataset.to_pickle('final_kaggle_dataset.pkl')

    logger.info("Exiting")
    sys.exit(0)

    # Create the Tournaments_Data directory if it doesn't exist
    print_elapsed_time('Save tournament CSVs in /Tournaments_Data')
    output_dir = './Tournaments_Data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the current working directory
    original_dir = os.getcwd()

    # Change the working directory to the Tournaments_Data directory
    os.chdir(output_dir)

    # Iterate through unique tournament_ids and save each one to a separate CSV file
    unique_tournament_ids = result['tournament_id'].unique()
    print(len(unique_tournament_ids))
    for tournament_id in unique_tournament_ids:
        file_path = f'{tournament_id}.csv'
        if not os.path.exists(file_path):
            tournament_data = result[result['tournament_id'] == tournament_id]
            tournament_data.to_csv(file_path, index=False)

    # Change back to the original working directory
    print_elapsed_time('Revert current working directory (CWD)')
    os.chdir(original_dir)

    print("Exiting")

except Exception as e:
    logger.exception("An error occurred during script execution")
    sys.exit(1)