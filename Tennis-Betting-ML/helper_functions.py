import time as time
from datetime import datetime
import logging
import logging.handlers


def print_elapsed_time(text=''):
    """Print the elapsed time since either the start of the program or the last call to this function.

    Args:
        text (str): Optional text to print before the elapsed time. Default is an empty string.

    Returns:
        None.

    Raises:
        None.
    """
    text = '\n' + '\033[1m' + text + '\033[0m'

    current_time = time.time()
    current_datetime = datetime.fromtimestamp(current_time)
    formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

    if not hasattr(print_elapsed_time, 'start_time'):
        print(text + '\n\tCurrent Time: {}'.format(formatted_datetime))
        print_elapsed_time.start_time = current_time

    else:
        elapsed_time = current_time - print_elapsed_time.last_time
        time_unit = 'seconds'
        if elapsed_time >= 60:
            elapsed_time /= 60
            time_unit = 'minutes'
        print(text + '\n\tElapsed time since last call: {:.2f} {})'.format(elapsed_time, time_unit))
    print_elapsed_time.last_time = current_time

def setup_logger(name, log_file, level=logging.INFO):
    """Set up a logger with a file handler, formatter, and optional log rotation.

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
