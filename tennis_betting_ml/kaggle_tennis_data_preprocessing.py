#!/usr/bin/env python

import pandas as pd
import os
import sys

from helper_functions import setup_logger

params = {}

if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        parts = arg.split('=')
        if len(parts) == 2:
            name, value = parts
            params[name] = value

if 'debug' in params:
    debug = params['debug']
else:
    debug = False

# add a way to perform these operations on incremental rows

# Set up the logger
logger = setup_logger(__name__, 'my_log_file.log')
logger.info('Starting script execution')

try:
    from helper_functions import print_elapsed_time

    if debug:
        print("Running in debug mode")

    # Step 1: Read in data
    print_elapsed_time('Reading in data')
    all_matches = pd.read_csv("all_matches.csv", parse_dates=True)
    tournaments = pd.read_csv("all_tournaments.csv", parse_dates=True)

    # Create unique match identifiers
    all_matches['match_id_player_1'] = all_matches['player_id'] + '_' + all_matches['opponent_id'] + '_' + all_matches['tournament'] + '_' + all_matches['start_date'] + '_' + all_matches['round_num'].astype(str)
    all_matches['match_id_player_2'] = all_matches['opponent_id'] + '_' + all_matches['player_id'] + '_' + all_matches['tournament'] + '_' + all_matches['start_date'] + '_' + all_matches['round_num'].astype(str)

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

    # Sort all_matches by start_date ascending
    print_elapsed_time('Resetting index')
    all_matches['start_date'] = pd.to_datetime(all_matches['start_date'])
    all_matches = all_matches.sort_values(by=['player_id', 'year', 'start_date', 'round_num'], ascending=[True, True, True, True]).reset_index(drop=True)

    # Add running_total_matches_played column
    print_elapsed_time('Calculating running total matches played')
    all_matches['running_total_matches_played'] = all_matches.groupby('player_id', sort=False).cumcount() + 1

    # Convert player_victory to 1's and 0's
    print_elapsed_time('Converting victories from t and f to 1 and 0')
    all_matches['player_victory'] = all_matches['player_victory'].map({'t': 1, 'f': 0})

    # Add running_total_victories column
    print_elapsed_time('Calculating running total victories')
    all_matches['running_total_victories'] = all_matches.groupby('player_id', sort=False)['player_victory'].cumsum()

    # Print DF for visual inspection
    if debug:
        print_elapsed_time('Examining running totals to ensure accuracy')
        pd.set_option('display.max_columns', None)
        print(all_matches[(all_matches['player_id'] == 'hugo-armando')][['player_id', 'opponent_id', 'match_id_player_1', 'match_id_player_2', 'year', 'start_date', 'round_num', 'running_total_matches_played', 'running_total_victories']].head(20))

    #Split the dataset into two DataFrames, one for each player
    player_1_matches = all_matches.add_suffix('_1')
    player_2_matches = all_matches.add_suffix('_2')

    # Visually inspect dfs before merge
    if debug:
        print_elapsed_time('Printing player 1 matches')
        print(player_1_matches[(player_1_matches['player_id_1'] == 'hugo-armando')][['player_id_1', 'opponent_id_1', 'match_id_player_1_1', 'match_id_player_2_1', 'year_1', 'start_date_1', 'round_num_1', 'running_total_matches_played_1', 'running_total_victories_1']].head(20))
        print_elapsed_time('Printing player 2 matches')
        print(player_2_matches[(player_2_matches['player_id_2'] == 'hugo-armando')][['player_id_2', 'opponent_id_2', 'match_id_player_1_2', 'match_id_player_2_2', 'year_2', 'start_date_2', 'round_num_2', 'running_total_matches_played_2', 'running_total_victories_2']].head(20))

    # Merge the renamed DataFrames
    print_elapsed_time('Perform self join')
    final_dataset = player_1_matches.merge(player_2_matches, left_on='match_id_player_1_1', right_on='match_id_player_2_2')

    # Drop the _y columns and rename the _x columns
    print_elapsed_time('Rename columns with suffixes')
    shared_columns = ['player_id', 'opponent_id', 'start_date', 'court_surface', 'currency', 'end_date', 'location', 'num_sets', 'opponent_name', 'player_name', 'prize_money', 'round', 'tournament', 'year', 'duration', 'nation', 'match_id_player_1', 'match_id_player_2', 'tournament_id']
    for col in final_dataset.columns:
        if col.endswith('_2'):
            if col[:-2] in shared_columns:
                final_dataset.drop(columns=[col], inplace=True)
        elif col.endswith('_1'):
            if col[:-2] in shared_columns:
                final_dataset.rename(columns={col: col[:-2]}, inplace=True)

    # Reorder columns
    print_elapsed_time('Reorder columns')
    no_suffix_columns = [col for col in final_dataset.columns if not col.endswith('_1') and not col.endswith('_2')]
    suffix_1_columns = [col for col in final_dataset.columns if col.endswith('_1')]
    suffix_2_columns = [col for col in final_dataset.columns if col.endswith('_2')]
    ordered_columns = no_suffix_columns + suffix_1_columns + suffix_2_columns
    final_dataset = final_dataset[ordered_columns]

    # Sort dataframe
    final_dataset = final_dataset.sort_values(by=['player_id', 'year', 'start_date', 'round_num_1'], ascending=[True, True, True, True]).reset_index(drop=True)
    if debug:
        print_elapsed_time('Visually inspecting final dataset')
        print(final_dataset[(final_dataset['player_id'] == 'hugo-armando')][['player_id', 'year', 'start_date', 'round_num_1', 'running_total_matches_played_1', 'running_total_victories_1']].head(20))

    #Saving dataframe as CSV
    print_elapsed_time('Save dataframe as CSV')
    final_dataset.to_csv('final_kaggle_dataset.csv', index=False)

    #Saving dataframe as PKL
    print_elapsed_time('Save dataframe as pickle')
    final_dataset.to_pickle('final_kaggle_dataset.pkl')

    logger.info("Exiting")
    sys.exit(0)

except Exception as e:
    logger.exception("An error occurred during script execution")
    sys.exit(1)