import time
import os
import logging
from datetime import datetime
import pandas as pd
from logging import handlers


def print_elapsed_time(text=''):
    """
    Prints the elapsed time since either the start of the program or the last call to this function.

    Args:
        text (str): Optional text to print before the elapsed time. Default is an empty string.

    Returns:
        None.
    """
    text = f'\n{text}'

    current_time = time.time()
    current_datetime = datetime.fromtimestamp(current_time)
    formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

    if not hasattr(print_elapsed_time, 'last_time'):
        print_elapsed_time.last_time = None

    if not print_elapsed_time.last_time:
        print(f'{text}\n\tCurrent Time: {formatted_datetime}')
        print_elapsed_time.start_time = current_time
    else:
        elapsed_time = current_time - print_elapsed_time.last_time
        time_unit = 'seconds'
        if elapsed_time >= 60:
            elapsed_time /= 60
            time_unit = 'minutes'
        print(f'{text}\n\tElapsed time since last call: {elapsed_time:.2f} {time_unit}')
    print_elapsed_time.last_time = current_time


def setup_logger(name, log_file, level=logging.INFO):
    """
    Sets up a logger with a file handler, formatter, and optional log rotation.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): The logging level, e.g., logging.INFO, logging.DEBUG, etc. Default is logging.INFO.

    Returns:
        logging.Logger: A logger object with the specified configuration.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a rotating file handler with log rotation
    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def save_data_file(dataframe: pd.DataFrame, file_type: str, filename: str = None) -> None:
    """
    Saves a pandas DataFrame as a CSV or Pickle file in a data folder in the current working directory.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be saved.
        file_type (str): The file type to save the DataFrame as. Either 'csv' or 'pickle'.
        filename (str, optional): The name to save the DataFrame as. If not provided, the name of the DataFrame
                                  is used, or a default name of 'datafile' is used if the DataFrame has no name.

    Raises:
        ValueError: If the file type provided is not supported.
        FileNotFoundError: If the 'data' directory does not exist.

    Returns:
        None
    """
    # Define the data directory path
    data_dir = os.path.join(os.getcwd(), 'data')

    # Create the data directory if it doesn't already exist
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # Set filename if not provided
    if filename is None:
        filename = dataframe.name + '.' + file_type if dataframe.name is not None else 'datafile.' + file_type

    # Save the dataframe as the specified file type
    if file_type == 'csv':
        with open(os.path.join(data_dir, filename), 'w') as file:
            dataframe.to_csv(file, index=False)
    elif file_type == 'pickle':
        with open(os.path.join(data_dir, filename), 'wb') as file:
            pickle.dump(dataframe, file)
    else:
        raise ValueError('File type not supported')

def read_data(file_name: str) -> pd.DataFrame:
    """
    Reads data from a CSV or Pickle file and returns a Pandas DataFrame.

    Args:
        file_name (str): The name of the file to read.

    Returns:
        pandas.DataFrame: A DataFrame containing the data from the file.

    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the 'data' directory does not exist.
    """
    data_folder = 'data'
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"{data_folder} directory not found")
    
    with open(os.path.join(data_folder, file_name)) as f:
        extension = os.path.splitext(file_name)[1]
        if extension == '.csv':
            df = pd.read_csv(f, parse_dates=True)
        elif extension == '.pkl':
            df = pd.read_pickle(f)
        else:
            raise ValueError(f'Invalid file type: {extension}')
    return df