#!/usr/bin/env python

import pandas as pd

from helper_functions import setup_logger

# Set up the logger
logger = setup_logger(__name__, 'my_log_file.log')
logger.info('Starting script execution')

# Remove duplicate matches function
def remove_duplicates(data):
    """Remove duplicated rows from a DataFrame based on sorted player and opponent IDs.

    Args:
        data (pandas.DataFrame): The input DataFrame with rows to be deduplicated.

    Returns:
        pandas.DataFrame: The deduplicated DataFrame.

    Raises:
        None.
    """

    def sort_teams(row):
        """Helper function to sort and concatenate player and opponent IDs.

        Args:
            row (pandas.Series): A row of a DataFrame containing 'player_id' and 'opponent_id' columns.

        Returns:
            str or None: A concatenated and sorted string of 'player_id' and 'opponent_id', or None if either value is None.
        """
        try:
            if row['player_id'] is None or row['opponent_id'] is None:
                return None
            return '_'.join(sorted([row['player_id'], row['opponent_id']]))
        except TypeError:
            return None

    data['match'] = data.apply(sort_teams, axis=1)

    # Drop rows where the match column is None
    data.dropna(subset=['match'], inplace=True)

    # Drop duplicate rows based on the new "match" column
    data.drop_duplicates(subset='match', inplace=True)

    # Drop the "match" column, as it is no longer needed
    data.drop('match', axis=1, inplace=True)

    # Reset dataframe index
    data = data.reset_index(drop=True)

    # Try to drop unnamed index's that may still persists
    try:
        data = data.drop(columns='Unnamed: 0', axis=1)
    except:
        pass
    
    return data

try:
    from helper_functions import print_elapsed_time
    
    # Import Data
    results = pd.read_csv('final_kaggle_dataset.csv', parse_dates=True)
    
    # Call Remove Duplicates Function
    print_elapsed_time('Removing duplicate rows')
    results = remove_doubles(results)

    # Save results
    print_elapsed_time('Done removing duplicate rows')
    results.to_csv('final_df.csv', index=False)
    
    logger.info("Exiting")
    sys.exit(0)

except Exception as e:
    logger.exception("An error occurred during script execution")
    sys.exit(1)

# files = os.listdir(cwd)
# for file in files:
#     temp_dat = pd.read_csv(str(file))
#     temp_dat = remove_doubles(temp_dat)
#     temp_dat.to_csv(str(file), index=False)
# print_elapsed_time()

# os.chdir('../')



