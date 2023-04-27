# Standard library imports
import os
import functools
import pickle
import logging
import time
import sys
from typing import List, Dict, Tuple, Optional
from multiprocessing import Manager, Process, Lock, Value
from ctypes import c_double, c_int, c_void_p
import multiprocessing as mp


# Third-party imports
import pandas as pd
import numpy as np
from tqdm import tqdm

# Local imports
from helper_functions import print_elapsed_time

# Create a list of step sizes to use in the Elo calculation
steps = [50, 100, 150, 200, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000, 125000, 150000, 200000]
step_sizes = steps

# Create a logger object and add a file handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('my_log_file.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def create_shared_memory(num_unique_players, manager, initial_elo=1500) -> tuple:
    """
    Creates shared memory for player ELO ratings and number of matches played.

    Args:
        num_unique_players: An integer representing the maximum player ID value.

    Returns:
        tuple: A tuple of two multiprocessing shared arrays - `player_elo_shared` and `player_matches_shared`.
        
        player_elo_shared: A shared array of type 'd' (double precision floating-point number) with a length of max_player_id + 1. 
        It represents the ELO rating of each player. The initial value is set to 1500.
        
        player_matches_shared: A shared array of type 'i' (integer) with a length of max_player_id + 1. 
        It represents the number of matches played by each player. The initial value is set to 0.
    """
    player_elo_shared = np.zeros(num_unique_players) + initial_elo
    player_matches_shared = np.zeros(num_unique_players)

    shared_memory = manager.dict()
    shared_memory['player_elo_shared'] = Array(c_double, player_elo_shared)
    shared_memory['player_matches_shared'] = Array(c_int, player_matches_shared)

    return shared_memory

def search_rows(start_row: int, player_id: str, data: pd.DataFrame, steps: List[int], starting_elo: float = 1500) -> Tuple[float, int]:
    """
    Searches the rows of the data frame up to the given start row to find previous matches
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
    data = pickle.loads(data_hash)  # Reconstruct the data frame from the hash
    m, last_elo, victory = None, None, None

    for step in steps:
        j = max(start_row - step, 0)
        step_bin = data.loc[((data['player_id'] == player_id) & (data['elo_1'] != 0)) | ((data['opponent_id'] == player_id) & (data['elo_2'] != 0)), ['player_id', 'opponent_id', 'm_1', 'elo_1', 'm_2', 'elo_2', 'player_1_victory', 'player_2_victory']].iloc[j:start_row + 1]

        if len(step_bin) == 0:
            continue

        row = step_bin.iloc[-1]

        if row['player_id'] == player_id:
            m = row['m_1'] + 1  # Increment the match count
            last_elo = row['elo_1']
            victory = row['player_1_victory']
        else:
            m = row['m_2'] + 1  # Increment the match count
            last_elo = row['elo_2']
            victory = row['player_2_victory']

        # add break statement to stop looping through the steps once the desired match is found
        break

    result = m, last_elo, victory
    return result


def update_last_processed_row(shared_dict: Dict, row: int) -> None:
    """
    Update the last processed row in a shared dictionary.

    Args:
        shared_dict (dict): The shared dictionary containing the last processed row value.
        row (int): The value of the last processed row to store in the shared dictionary.
    """
    shared_dict.update({'last_processed_row': row})


def read_last_processed_row(shared_dict: Dict) -> int:
    """
    Read the last processed row from a shared dictionary.

    Args:
        shared_dict (dict): The shared dictionary containing the last processed row value.

    Returns:
        int: The last processed row value, or 0 if the key is not present in the shared dictionary.
    """
    return shared_dict.get('last_processed_row', 0)



def process_chunk(args):
    """
    Process a chunk of the dataframe to compute Elo ratings using the compute_elo_ratings function.

    Args:
        args (tuple): A tuple containing:
            chunk (pandas.DataFrame): A chunk of the matches dataframe to process.

    Returns:
        pandas.DataFrame: A DataFrame with updated Elo ratings for the provided chunk.
    """

    chunk, steps, shared_memory = args
    player_elo_shared_ptr = shared_memory['player_elo_shared'].value
    player_matches_shared_ptr = shared_memory['player_matches_shared'].value

    player_elo_shared = Array('d', [0] * num_unique_players * 2, lock=False)
    player_matches_shared = Array('i', [0] * num_unique_players * 2, lock=False)
    player_elo_shared._obj = player_elo_shared._obj.from_address(player_elo_shared_ptr)
    player_matches_shared._obj = player_matches_shared._obj.from_address(player_matches_shared_ptr)

    try:
        logging.info("Processing the dataframe chunk")
        return compute_elo_ratings(chunk, {'player_elo_shared': player_elo_shared, 'player_matches_shared': player_matches_shared}, steps)  # Pass the shared_memory here
    except Exception as e:
        logging.exception(f"Error processing chunk: {e}")
        print("Error:", e)
        print("Dataframe chunk:")
        print(chunk)
        raise e
    
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

    return new_elo_1, new_elo_2, w_1, w_2


def compute_elo_ratings(chunk, shared_memory, steps=None, update_last_processed_row=None, last_processed_row_dict=None, update_interval=1000, output_file='elo.csv', temp_output_file='elo_temp.csv'):
    """
    Compute Elo ratings for each player based on their match history.

    Args:
    - chunk: Pandas DataFrame containing match data, including columns for player_id, opponent_id,
      elo_1, elo_2, m_1, m_2, w_1, w_2, and date.
    - player_elo_shared: Shared memory array containing player Elo ratings.
    - player_matches_shared: Shared memory array containing player match counts.
    - steps: Number of iterations to use when computing the Elo rating.
    - update_last_processed_row: Optional function to update the last processed row in a file.
    - update_interval: Interval at which to update the output and temp files.
    - output_file: Output file for the final Elo ratings.
    - temp_output_file: Temporary output file for storing Elo ratings during computation.

    Returns:
    - A copy of chunk with updated Elo ratings.
    """

    player_elo_shared = shared_memory['player_elo_shared']
    player_matches_shared = shared_memory['player_matches_shared']

    for index, row in chunk.iterrows():
        date = row['start_date']
        player_1_id = row['player_id']
        player_2_id = row['opponent_id']
        w_1 = row['player_1_victory']
        w_2 = row['player_1_victory']

        elo_1 = player_elo_shared[player_1_id]
        elo_2 = player_elo_shared[player_2_id]

        w_1 = 1 if w_1 == 't' else 0
        w_2 = 1 if w_2 == 't' else 0

        player_elo_shared[player_1_id], player_elo_shared[player_2_id], w_1, w_2 = elo_calc(
            elo_1, elo_2, w_1, w_2, player_matches_shared[player_1_id], player_matches_shared[player_2_id]
        )

        # Increment the total matches for each player after the elo_calc call
        player_matches_shared[player_1_id] += 1
        player_matches_shared[player_2_id] += 1

        chunk.loc[index, 'elo_1'] = elo_1
        chunk.loc[index, 'elo_2'] = elo_2
        chunk.loc[index, 'm_1'] = player_matches_shared[player_1_id]
        chunk.loc[index, 'm_2'] = player_matches_shared[player_2_id]

        if steps is not None:
            steps.append(1)

        if update_last_processed_row is not None and index % update_interval == 0:
            update_last_processed_row(index)

    return chunk


if __name__ == "__main__":
    print("\nReading in files from checkpoint")
    print_elapsed_time()

    # Load the matches data
    matches_df = pd.read_csv('final_df.csv', parse_dates=True)
    
    with Manager() as manager:
        shared_dict = manager.dict()
        shared_dict['last_processed_row'] = read_last_processed_row(shared_dict)

        n_processes = cpu_count()
        n_chunks = n_processes * 2  # You can adjust this value depending on your system

        copy = pd.read_csv('auxiliary_df.csv', parse_dates=True)
        copy = copy[:1500]

        if len(shared_dict) > 0:
            print(f"Resuming from last processed row: {shared_dict['last_processed_row']}")
            copy = copy.loc[shared_dict['last_processed_row'] + 1:]

        matches_cache = {}
        chunks = divide_dataframe(copy, n_chunks)

        steps = manager.list(step_sizes)
        
        unique_player_ids = pd.concat([copy['player_id'], copy['opponent_id']]).unique()
        num_unique_players = len(unique_player_ids)

        shared_memory = create_shared_memory(num_unique_players, manager)

        chunk_args = [(chunk, steps, shared_memory) for chunk in chunks]

        with Pool(n_processes) as pool:
            processed_chunks = pool.map(process_chunk, chunk_args)

        updated_data = pd.concat(processed_chunks)

    print("\nProcessed dataframe:")
    print(updated_data.head())
    print_elapsed_time()