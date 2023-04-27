#!/usr/bin/env python

import argparse
import pandas as pd
import sys
from datetime import datetime
from helper_functions import setup_logger

def remove_players_born_before_date(players_data, birth_year):
    """
    Returns a DataFrame containing players born on or after the specified year.

    Args:
        players_data (pandas.DataFrame): DataFrame containing player data.
        birth_year (int): The year before which to exclude players.

    Returns:
        pandas.DataFrame: A DataFrame containing players born on or after the specified year.
    """
    players_data = players_data.copy()  # Create a copy to avoid modifying the original dataframe
    players_data['dob'] = pd.to_datetime(players_data['dob'], errors='coerce')
    players_born_after_date = players_data.loc[players_data['dob'].dt.year >= birth_year]
    return players_born_after_date


if __name__ == "__main__":
    # Set up the logger
    logger = setup_logger(__name__, 'my_log_file.log')
    logger.info('Starting script execution')

    try:
        from helper_functions import print_elapsed_time

        parser = argparse.ArgumentParser(description='Clean player data')
        parser.add_argument('--matches-data', type=str, default='final_kaggle_dataset.csv',
                            help='the file name of the matches data')
        parser.add_argument('--players-data', type=str, default='players_data.csv',
                            help='the file name of the player data')
        parser.add_argument('--birth-year', type=int, default=1960,
                            help='the year before which players will be removed')
        parser.add_argument('--verbose', action='store_true', help='print elapsed time for each step')
        args = parser.parse_args()

        if args.verbose:
            print_elapsed_time('Importing match data')
        matches_data = pd.read_csv(args.matches_data, parse_dates=True)

        if args.verbose:
            print_elapsed_time('Importing player data')
        players_data = pd.read_csv(args.players_data, parse_dates=['dob'], infer_datetime_format=True)

        if args.verbose:
            print_elapsed_time('Removing players born before 1960')
        players_data_cleaned = remove_players_born_before_date(players_data, args.birth_year)

        if args.verbose:
            print_elapsed_time('Merging match data with player data')
        players_in_matches = pd.merge(matches_data[['player_id']], players_data_cleaned, on='player_id')

        if args.verbose:
            print_elapsed_time('Saving player data')
        players_in_matches.to_csv(args.players_data, index=False)

        logger.info("Exiting")
        sys.exit(0)

    except Exception as e:
        logger.exception("An error occurred during script execution")
        sys.exit(1)