import pandas as pd
from typing import Tuple

player_elos = {}

def elo_calc(row: pd.Series, initial_elo=1500, max_elo_diff=800) -> Tuple[float, float]:
    """
    Calculates the new Elo ratings for two players after a match.

    Args:
        row (pd.Series): A pandas Series object containing match data.
        initial_elo (float, optional): The initial Elo rating for a new player. Defaults to 1500.
        max_elo_diff (float, optional): The maximum allowed Elo difference between two players. Defaults to 800.

    Returns:
        Tuple[float, float]: A tuple containing the new Elo ratings for both players.
    """
    player_1 = row['player_id']
    player_2 = row['opponent_id']
    elo_1 = player_elos.get(player_1, initial_elo)
    elo_2 = player_elos.get(player_2, initial_elo)
    w_1 = row['running_total_victories_1']
    w_2 = row['running_total_victories_2']
    m_1 = row['running_total_matches_played_1']
    m_2 = row['running_total_matches_played_2']

    elo_diff = min(max(elo_1 - elo_2, -max_elo_diff), max_elo_diff)
    expected_1 = 1 / (1 + 10 ** (elo_diff / 400))
    expected_2 = 1 - expected_1
    decay_1 = 250.0 / ((5 + m_1) ** 0.4)
    decay_2 = 250.0 / ((5 + m_2) ** 0.4)
    new_elo_1 = elo_1 + decay_1 * (w_1 - expected_1)
    new_elo_2 = elo_2 + decay_2 * (w_2 - expected_2)

    player_elos[player_1] = new_elo_1
    player_elos[player_2] = new_elo_2

    return new_elo_1, new_elo_2

def apply_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the Elo calculation to a DataFrame of tennis matches.

    Args:
        df (pandas.DataFrame): A DataFrame containing the player ID, opponent ID,
                               and other match statistics for each tennis match.

    Returns:
        pandas.DataFrame: The input DataFrame with two additional columns, 'elo_1'
                           and 'elo_2', containing the updated Elo ratings for the two players.

    Raises:
        None.
    """
    df['elo_1'], df['elo_2'] = zip(*df.apply(elo_calc, axis=1))
    return df