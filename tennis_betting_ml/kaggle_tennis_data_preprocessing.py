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
    # Check if 'player_id' column is present in the DataFrame
    if 'player_id' in df.columns:
        sort_columns = ['player_id', 'year', 'start_date', 'round_num']
    else:
        sort_columns = ['player_id_1', 'year', 'start_date', 'round_num']
    
    # Sort the DataFrame based on the appropriate columns
    df = df.sort_values(by=sort_columns, ascending=True).reset_index(drop=True)
    
    return df

# Define a function to create unique match identifiers
def create_unique_match_identifiers(df):
    """
    Create unique match identifiers for both player 1 and player 2 in the input DataFrame.
    
    This function generates unique match identifiers for both player 1 and player 2 by
    concatenating the player ID, opponent ID, tournament, start date, and round number
    for each match. The resulting match identifiers are added as new columns to the
    input DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing match data with columns
            'player_id', 'opponent_id', 'tournament', 'start_date', and 'round_num'.
            
    Returns:
        pd.DataFrame: The input DataFrame with two new columns added: 'match_id_player_1'
            and 'match_id_player_2', containing the unique match identifiers for
            player 1 and player 2, respectively.
    """
    df['match_id_player_1'] = df['player_id'] + '_' + df['opponent_id'] + '_' + df['tournament'] + '_' + df['start_date'] + '_' + df['round_num'].astype(str)
    df['match_id_player_2'] = df['opponent_id'] + '_' + df['player_id'] + '_' + df['tournament'] + '_' + df['start_date'] + '_' + df['round_num'].astype(str)
    return df

# Define a function to filter and clean data before executing main script
def filter_and_clean_data(df):
    """
    Filters and cleans the input DataFrame containing tennis match data.

    This function performs the following operations:
    1. Filters the matches from the year 2000 and onwards.
    2. Replaces the 'Carpet' court surface with 'Grass'.
    3. Converts the 'player_victory' column values from 't'/'f' to 1/0.
    4. Removes the rows corresponding to doubles matches.
    5. Removes the rows with missing 'num_sets' values.

    Args:
        df (pd.DataFrame): A DataFrame containing tennis match data.

    Returns:
        pd.DataFrame: A filtered and cleaned DataFrame with updated tennis match data.
    """
    df = df[df['year'] >= 2000]
    df['court_surface'] = df['court_surface'].replace('Carpet', 'Grass')
    df['player_victory'] = df['player_victory'].map({'t': 1, 'f': 0})
    df = df[df['doubles'] != 't']
    df = df[df['num_sets'].notna()]
    return df

# Create function to caluclate running totals
def calculate_running_totals(df, player_id_col, victory_col, surface=None):
    """Calculate running totals of matches played and victories for each player.

    Args:
        df (pandas.DataFrame): The input DataFrame containing match data.
        player_id_col (str): The name of the column containing player IDs.
        victory_col (str): The name of the column indicating player victories.
        surface (str, optional): The surface type to filter matches on. Defaults to None.

    Returns:
        pandas.DataFrame: The input DataFrame with new columns added for running totals.
    """
    if surface:
        surface_suffix = f"_{surface.lower()}"
    else:
        surface_suffix = ""

    df[f"running_total_matches_played{surface_suffix}"] = df.groupby(player_id_col, sort=False).cumcount() + 1
    df[f"running_total_victories{surface_suffix}"] = df.groupby(player_id_col, sort=False)[victory_col].cumsum()

    return df

def forward_fill_court_specific_data(df, court_surfaces):
    """Forward-fill missing court-specific data for each player.

    This function fills missing values in the 'running_total_matches_played_{surface}' and
    'running_total_victories_{surface}' columns for each player using the forward-fill method.

    Args:
        df (pandas.DataFrame): The input DataFrame containing match data.
        court_surfaces (list): A list of court surface types to fill missing data for.

    Returns:
        pandas.DataFrame: The input DataFrame with missing data forward-filled for each court surface.
    """
    for surface in court_surfaces:
        df[f'running_total_matches_played_{surface.lower()}'] = df.groupby('player_id')[f'running_total_matches_played_{surface.lower()}'].fillna(method='ffill')
        df[f'running_total_victories_{surface.lower()}'] = df.groupby('player_id')[f'running_total_victories_{surface.lower()}'].fillna(method='ffill')
    return df

def print_debug_info(df, debug_player, court_surfaces, debug_rows):
    """
    Print a slice of the DataFrame for a specific player with running totals for easier debugging.

    Args:
        df (pd.DataFrame): The DataFrame containing the tennis match data.
        debug_player (str): The player ID for which to print the debugging information.
        court_surfaces (list): A list of court surface types.
        debug_rows (int): The number of rows to print.

    Returns:
        None
    """
    pd.set_option('display.max_columns', None)

    columns_to_print = ['player_id', 'opponent_id', 'match_id_player_1', 'match_id_player_2', 'year', 'start_date', 'round_num']

    if 'running_total_matches_played_1' in df.columns:
        columns_to_print.extend(['running_total_matches_played_1', 'running_total_victories_1'])
        surface_columns = [f'{prefix}_{court.lower()}_1' for court in court_surfaces for prefix in ['running_total_matches_played', 'running_total_victories']]
        columns_to_print.extend(surface_columns)
    else:
        columns_to_print.extend(['running_total_matches_played', 'running_total_victories'])
        surface_columns = [f'{prefix}_{court.lower()}' for court in court_surfaces for prefix in ['running_total_matches_played', 'running_total_victories']]
        columns_to_print.extend(surface_columns)

    print(df[(df['player_id'] == debug_player)][columns_to_print].head(debug_rows))

def merge_datasets(all_matches, surface_dfs, court_surfaces):
    """
    Split the all_matches DataFrame into player_1_matches and player_2_matches, and then
    merge them along with surface-specific dataframes.

    Args:
        all_matches (pd.DataFrame): The DataFrame containing all match data.
        surface_dfs (dictionary): A dictionary containing Pandas DataFrames for court_surface specific matches
        court_surfaces (list): A list of court surface types.

    Returns:
        dict: A dictionary containing the final_dataset and surface-specific DataFrames.
    """
    # Split the dataset into two DataFrames, one for each player
    player_1_matches = all_matches.add_suffix('_1')
    player_2_matches = all_matches.add_suffix('_2')

    final_dataset = player_1_matches.merge(player_2_matches, left_on='match_id_player_1_1', right_on='match_id_player_2_2')

    surface_dataframes = {}
    for surface in court_surfaces:
        surface_df = surface_dfs[surface.lower()]

        surface_df_1 = surface_df.add_suffix('_1')
        surface_df_2 = surface_df.add_suffix('_2')

        surface_df = surface_df_1.merge(surface_df_2, left_on='match_id_player_1_1', right_on='match_id_player_2_2')

        surface_dataframes[f"{surface}_matches"] = surface_df

    return {"final_dataset": final_dataset, **surface_dataframes}

def main(debug=False, debug_player='hugo-armando', debug_rows=5):
    try:
        from helper_functions import print_elapsed_time

        # Print to the user if the script is being run in debug mode
        if debug:
            print("Running in debug mode")
        
        # Call imported read_data function to retrieve all_matches.csv
        print_elapsed_time('Reading in data')
        all_matches = read_data("all_matches.csv")

        # Call filter_and_clean_data function to clean the all_matches dataframe
        print_elapsed_time('Filtering and cleaning data')
        all_matches = filter_and_clean_data(all_matches)
        
        # Call create_unique_match_identifiers to add match ids to all_matches dataframe
        print_elapsed_time('Adding match identifiers')
        all_matches = create_unique_match_identifiers(all_matches)

        # Sort all_matches by start_date ascending
        print_elapsed_time('Resetting index')
        all_matches = sort_dataframe(all_matches)

        # Calculate running total matches played and running total victories for all matches
        print_elapsed_time('Calculating running total matches and victories for all matches')
        all_matches = calculate_running_totals(all_matches, "player_id", "player_victory")

        # Create a list of unique court_surfaces and drop NaN values
        court_surfaces = all_matches['court_surface'].dropna().unique()

        # Calcuating running totals for each surface type
        surface_dfs = {}
        for surface in court_surfaces:
            # Filter the dataframe by court_surface
            df = all_matches[all_matches['court_surface'] == surface]

            # Add running_total_matches_played column with surface suffix
            print_elapsed_time(f"Calculate running total {surface} matches played")
            df = calculate_running_totals(df, "player_id", "player_victory", surface)

            # Save the data to a new dataframe
            print_elapsed_time(f"Add {surface} dataframe to surface_dfs dict")
            surface_dfs[surface.lower()] = df

            # Join surface specific df back to all_matches
            print_elapsed_time(f"Join {surface} dataframe to all_matches dataframe")
            all_matches = all_matches.merge(df[['match_id_player_1', f"running_total_matches_played_{surface.lower()}", f"running_total_victories_{surface.lower()}"]], on="match_id_player_1", how="left")

        # Forward Fill Court Spoecific Data
        print_elapsed_time(f"Filling running total data for each court type forward")
        all_matches = forward_fill_court_specific_data(all_matches, court_surfaces)

        # Print DF for visual inspection if in debug mode
        if debug:
            print_elapsed_time('Printing data after forward fill for validation')
            print_debug_info(all_matches, debug_player, court_surfaces, debug_rows)

        # Perform self join of all_matches and court specific dataframes to get oppponent info
        print_elapsed_time(f"Performing self joins")
        merged_data = merge_datasets(all_matches, surface_dfs, court_surfaces)

        # Drop shared columns (e.g. 'start_date', 'court_surface', etc.) that end with _2 and remove the _1 from the remaining values
        shared_columns = ['player_id', 'opponent_id', 'start_date', 'court_surface', 'currency', 'end_date', 'location', 'num_sets', 'opponent_name', 'player_name', 'prize_money', 'round', 'tournament', 'year', 'duration', 'nation', 'match_id_player_1', 'match_id_player_2', 'tournament_id', 'round_num']
        for key, value in merged_data.items():
            print_elapsed_time(f"Cleaning columns in {key.lower()} dataframe")
            for col in value.columns:
                if col.endswith('_2'):
                    if col[:-2] in shared_columns:
                        value.drop(columns=[col], inplace=True)
                elif col.endswith('_1'):
                    if col[:-2] in shared_columns:
                        value.rename(columns={col: col[:-2]}, inplace=True)

        # Reorder columns, and sort dataframes
        for key, value in merged_data.items():
            print_elapsed_time(f'Reorder columns for {key.lower()} matches')
            value = reorder_columns(value)
            print_elapsed_time(f'Sort dataframe columns for {key.lower()} matches')
            value = sort_dataframe(value)

        final_dataset = merged_data['final_dataset']

        # Print DF for visual inspection if in debug mode
        if debug:
            print_elapsed_time('Printing final dataset for validation')
            print_debug_info(final_dataset, debug_player, court_surfaces, debug_rows)

        for key, value in merged_data.items():
            df_file_name = f"{key.lower()}.csv"
            print_elapsed_time(f'Saving {df_file_name}')
            save_data_file(value, 'csv', f'{df_file_name}')

        logger.info("Exiting")
        sys.exit(0)

    except Exception as e:
        logger.exception("An error occurred during script execution")
        sys.exit(1)

if __name__ == "__main__":
    main(debug, debug_rows=20)

    # # Create player_match_id
    # print_elapsed_time('Creating player_match_id')
    # all_matches['player_match_id'] = all_matches['player_id'].astype(str) + '_' + all_matches['running_total_matches_played'].astype(str)

    