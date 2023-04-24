import time as time

def print_elapsed_time():
    """Print the elapsed time since either the start of the program or the last call to this function.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    current_time = time.time()
    if not hasattr(print_elapsed_time, 'last_time'):
        print("Elapsed time since start:", current_time)
    else:
        print("Elapsed time since last call:", current_time - print_elapsed_time.last_time)
    print_elapsed_time.last_time = current_time
