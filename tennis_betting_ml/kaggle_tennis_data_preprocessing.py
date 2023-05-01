#!/usr/bin/env python

import os
import sys
import re

import pandas as pd

from helper_functions import setup_logger, save_data_file, read_data
from elo_functions import apply_elo

params = {arg.split('=')[0]: arg.split('=')[1] for arg in sys.argv[1:] if '=' in arg}
debug = params.get('debug', False)

# add a way to perform these operations on incremental rows

# Set up the logger
logger = setup_logger(__name__, 'my_log_file.log')
logger.info('Starting script execution')

# Define a function to reorder columns for a given dataframe
def reorder_columns(df):
    """
    Reorders the columns of a given dataframe by moving all columns that end with '_1' to the right of columns
    that don't end with '_1' or '_2', followed by columns that end with '_2'. This function assumes that the
    dataframe has column names that follow the naming conventions established elsewhere in this codebase.

    Args:
        df (pandas.DataFrame): The dataframe to reorder.

    Returns:
        pandas.DataFrame: The reordered dataframe.
    """
    no_suffix_columns = [col for col in df.columns if not col.endswith('_1') and not col.endswith('_2')]
    suffix_1_columns = [col for col in df.columns if col.endswith('_1')]
    suffix_2_columns = [col for col in df.columns if col.endswith('_2')]
    ordered_columns = no_suffix_columns + suffix_1_columns + suffix_2_columns
    return df[ordered_columns]

# Define a function to sort rows
def sort_dataframe(df):
    """
    Sorts a dataframe by player_id, year, start_date, and round_num_1 columns in ascending order.

    Args:
        df (pandas.DataFrame): The dataframe to be sorted.

    Returns:
        pandas.DataFrame: The sorted dataframe.
    """
    return df.sort_values(by=['player_id', 'year', 'start_date', 'round_num_1'], ascending=True).reset_index(drop=True)


try:
    from helper_functions import print_elapsed_time

    if debug:
        print("Running in debug mode")

    # Read in data
    print_elapsed_time('Reading in data')
    all_matches = read_data("all_matches.csv")
    tournaments = read_data("all_tournaments.csv")

    all_matches = all_matches[:500]


    # Create unique match identifiers
    all_matches['match_id_player_1'] = all_matches['player_id'] + '_' + all_matches['opponent_id'] + '_' + all_matches['tournament'] + '_' + all_matches['start_date'] + '_' + all_matches['round_num'].astype(str)
    all_matches['match_id_player_2'] = all_matches['opponent_id'] + '_' + all_matches['player_id'] + '_' + all_matches['tournament'] + '_' + all_matches['start_date'] + '_' + all_matches['round_num'].astype(str)

    # Remove tournaments & matches prior to the year 2000
    print_elapsed_time('Removing old tournaments')
    tournaments = tournaments[tournaments['year'] >= 2000]
    all_matches = all_matches[all_matches['year'] >= 2000]

    # Create a unique identifier column in the tournaments dataframe
    print_elapsed_time('Creating unique tournament_ids')
    tournaments['tournament_id'] = tournaments['tournament'] + '_' + tournaments['year'].astype(str)
    tournaments['tournament_id'] = tournaments['tournament_id'].str.replace('/', '_')
    all_matches['tournament_id'] = all_matches['tournament'] + '_' + all_matches['year'].astype(str)
    all_matches['tournament_id'] = all_matches['tournament_id'].str.replace('/', '_')

    # Remove doubles matches from the data dataframe
    print_elapsed_time('Removing Doubles Matches')
    all_matches = all_matches[all_matches['doubles'] != 't']

    # Remove any match where num_sets is null / N/A / nan etc.
    print_elapsed_time('Removing erroneous matches')
    ###### this code is wrong because of row_num has not yet had _1 appended.
    all_matches = all_matches[all_matches['num_sets'].notna()]

    # Sort all_matches by start_date ascending
    print_elapsed_time('Resetting index')
    all_matches = sort_dataframe(all_matches)

    # Replace 'Carpet' with 'Grass'
    print_elapsed_time('Replacing Carpet with Grass')
    all_matches['court_surface'] = all_matches['court_surface'].replace('Carpet', 'Grass')

    # Convert player_victory to 1's and 0's
    print_elapsed_time('Remapping player_victory win values')
    all_matches['player_victory'] = all_matches['player_victory'].map({'t': 1, 'f': 0})

    # Add running_total_matches_played column for unfiltered dataframe
    print_elapsed_time('Calculate running total matches')
    all_matches['running_total_matches_played'] = all_matches.groupby('player_id', sort=False).cumcount() + 1

    # Add running_total_victories column for unfiltered dataframe
    print_elapsed_time('Calculate running total victories')
    all_matches['running_total_victories'] = all_matches.groupby('player_id', sort=False)['player_victory'].cumsum()

    # Create player_match_id
    print_elapsed_time('Creating player_match_id')
    all_matches['player_match_id'] = all_matches['player_id'].astype(str) + '_' + all_matches['running_total_matches_played'].astype(str)

    # Create a list of unique court_surfaces and drop NaN values
    court_surfaces = all_matches['court_surface'].dropna().unique()

    # Loop through all court_surfaces
    for surface in court_surfaces:
        # Filter the dataframe by court_surface
        df = all_matches[all_matches['court_surface'] == surface]

        # Add running_total_matches_played column with surface suffix
        print_elapsed_time(f'Calculate running total {surface} matches played')
        df[f'running_total_matches_played_{surface.lower()}'] = df.groupby('player_id', sort=False).cumcount() + 1

        # Add running_total_victories column with surface suffix
        print_elapsed_time(f'Calculate running total {surface} victories')
        df[f'running_total_victories_{surface.lower()}'] = df.groupby('player_id', sort=False)['player_victory'].cumsum()

        # Save the data to a new dataframe
        print_elapsed_time(f'Write to {surface}_matches dataframe')
        new_df_name = f"{surface}_matches"
        globals()[new_df_name] = df

    # Sort the all_matches dataframe by 'player_id' and 'running_total_matches_played', and reset the index
    print_elapsed_time('Reset index')
    all_matches = sort_dataframe(all_matches)

    # Fill NaN values by pushing down each value until the next non NaN value for specific columns
    print_elapsed_time('Forward fill all NaN values for court specific data')
    for surface in court_surfaces:
        all_matches[f'running_total_matches_played_{surface.lower()}'] = all_matches[f'running_total_matches_played_{surface.lower()}'].fillna(method='ffill')
        all_matches[f'running_total_victories_{surface.lower()}'] = all_matches[f'running_total_victories_{surface.lower()}'].fillna(method='ffill')

    # Print DF for visual inspection if in debug mode
    if debug:
        print_elapsed_time('Examining running totals to ensure accuracy')
        pd.set_option('display.max_columns', None)
        print(all_matches[(all_matches['player_id'] == 'hugo-armando')][['player_id', 'opponent_id', 'match_id_player_1', 'match_id_player_2', 'year', 'start_date', 'round_num', 'running_total_matches_played', 'running_total_victories']].head(20))

    # Split the dataset into two DataFrames, one for each player
    player_1_matches = all_matches.add_suffix('_1')
    player_2_matches = all_matches.add_suffix('_2')

    # Visually inspect DataFrames before merging if in debug mode
    if debug:
        print_elapsed_time('Printing player 1 matches')
        print(player_1_matches[(player_1_matches['player_id_1'] == 'hugo-armando')][['player_id_1', 'opponent_id_1', 'match_id_player_1_1', 'match_id_player_2_1', 'year_1', 'start_date_1', 'round_num_1', 'running_total_matches_played_1', 'running_total_victories_1']].head(20))
        print_elapsed_time('Printing player 2 matches')
        print(player_2_matches[(player_2_matches['player_id_2'] == 'hugo-armando')][['player_id_2', 'opponent_id_2', 'match_id_player_1_2', 'match_id_player_2_2', 'year_2', 'start_date_2', 'round_num_2', 'running_total_matches_played_2', 'running_total_victories_2']].head(20))

    # Merge the renamed DataFrames
    print_elapsed_time('Perform self join')
    final_dataset = player_1_matches.merge(player_2_matches, left_on='match_id_player_1_1', right_on='match_id_player_2_2')

    # Calculate ELOs for all_matches
    print_elapsed_time('Calculate ELOs')
    final_dataset = apply_elo(final_dataset)

    # Merge surface dataframes to player_1_matches and player_2_matches separately
    for surface in court_surfaces:
        print_elapsed_time(f'Perform self join for {surface} data')
        surface_df = globals()[f"{surface}_matches"]
        cols_to_merge = ['player_id', 'match_id_player_1', 'match_id_player_1', f'running_total_matches_played_{surface.lower()}', f'running_total_victories_{surface.lower()}']
        surface_df = surface_df[cols_to_merge]

        surface_df_1 = surface_df.copy()
        surface_df_1.columns = [f'{col}_1' if col != 'player_id' else col for col in cols_to_merge]

        surface_df_2 = surface_df.copy()
        surface_df_2.columns = [f'{col}_2' if col != 'player_id' else col for col in cols_to_merge]

        surface_df = surface_df_1.merge(surface_df_2, left_on='match_id_player_1_1', right_on='match_id_player_2_2')

        # Save the data to a new dataframe
        new_df_name = f"{surface}_matches"
        globals()[new_df_name] = surface_df

    # Loop through all court_surfaces and apply ELO calculation for each surface
    for surface in court_surfaces:
        print_elapsed_time(f'Calculate ELOs for {surface}')
        # Filter the dataframe by court_surface
        surface_df = final_dataset[final_dataset['court_surface'] == surface].copy()

        # Apply Elo calculation to the surface-specific dataframe
        surface_df = apply_elo(surface_df)  # Update the apply_elo function to accommodate new column names

        # Rename elo_1 and elo_2 columns
        surface_df.rename(columns={'elo_1': f'{surface}_elo_1', 'elo_2': f'{surface}_elo_2'}, inplace=True)

        # Save the data to a new dataframe
        print_elapsed_time(f'Storing {surface} data with ELO')
        new_df_name = f"{surface}_matches"
        globals()[new_df_name] = surface_df

    # Forward-fill the specified columns
    print_elapsed_time('Merge all surface datasets back to main dataset. Forward will court type metrics')
    for surface in court_surfaces:
        for suffix in ['_1', '_2']:
            final_dataset[f'running_total_matches_played_{surface.lower()}{suffix}'] = final_dataset[f'running_total_matches_played_{surface.lower()}{suffix}'].fillna(method='ffill')
            final_dataset[f'running_total_victories_{surface.lower()}{suffix}'] = final_dataset[f'running_total_victories_{surface.lower()}{suffix}'].fillna(method='ffill')
            final_dataset[f'{surface}_elo{suffix}'] = final_dataset[f'{surface}_elo{suffix}'].fillna(method='ffill')

    # Drop the _y columns and rename the _x columns for all dataframes
    shared_columns = ['player_id', 'opponent_id', 'start_date', 'court_surface', 'currency', 'end_date', 'location', 'num_sets', 'opponent_name', 'player_name', 'prize_money', 'round', 'tournament', 'year', 'duration', 'nation', 'match_id_player_1', 'match_id_player_2', 'tournament_id']
    for df in [final_dataset] + [globals()[f"{surface}_matches"] for surface in court_surfaces]:
        print_elapsed_time(f'Rename columns suffixes in {df.name}')
        for col in df.columns:
            if col.endswith('_2'):
                if col[:-2] in shared_columns:
                    df.drop(columns=[col], inplace=True)
            elif col.endswith('_1'):
                if col[:-2] in shared_columns:
                    df.rename(columns={col: col[:-2]}, inplace=True)

    # Reorder columns for final_dataset
    print_elapsed_time('Reorder columns for final_dataset')
    final_dataset = reorder_columns(final_dataset)

    # Reorder columns for surface-specific dataframes
    for surface in court_surfaces:
        print_elapsed_time(f'Reorder columns for {surface}_matches')
        globals()[f"{surface}_matches"] = reorder_columns(globals()[f"{surface}_matches"])

    # Sort dataframes
    print_elapsed_time('Sorting dataframes')
    final_dataset = sort_dataframe(final_dataset)

    for surface in court_surfaces:
        surface_df = globals()[f"{surface}_matches"]
        print_elapsed_time(f'Sorting {surface_df.name}')
        surface_df = sort_dataframe(surface_df)
        new_df_name = f"{surface}_matches"
        globals()[new_df_name] = surface_df

    # Print final datset if in debug mode
    if debug:
        print_elapsed_time('Visually inspecting final dataset')
        print(final_dataset[(final_dataset['player_id'] == 'hugo-armando')][['player_id', 'year', 'start_date', 'round_num_1', 'running_total_matches_played_1', 'running_total_victories_1']].head(20))

    #Saving dataframe as CSV
    print_elapsed_time('Save dataframe as CSV')
    final_dataset.to_csv('final_kaggle_dataset.csv', index=False)

    # Assuming court_surfaces is a list containing the surface types
    for surface in court_surfaces:
        new_df_name = f"{surface}_matches"
        dataframe = globals()[new_df_name]
        save_data_file(dataframe, 'csv', f'{surface}_matches.csv')


    #Saving dataframe as PKL
    print_elapsed_time('Save dataframe as pickle')
    final_dataset.to_pickle('final_kaggle_dataset.pkl')

    logger.info("Exiting")
    sys.exit(0)

except Exception as e:
    logger.exception("An error occurred during script execution")
    sys.exit(1)