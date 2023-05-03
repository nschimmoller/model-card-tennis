from typing import Tuple
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer

# Used in script to call this pipeline object
## from my_custom_module import apply_elo

### THIS IS A WORK IN PROGRESS THIS FILE DOES NOT WORK YET

class EloTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that calculates Elo ratings for tennis players based on match results.

    Args:
        initial_elo (float, optional): The initial Elo rating for a new player. Defaults to 1500.
        max_elo_diff (float, optional): The maximum allowed Elo difference between two players. Defaults to 800.
        decay_const (float, optional): The decay constant used in the Elo rating calculation. Defaults to 0.4.

    Attributes:
        param_grid (dict): The hyperparameter grid to be used in model tuning.

    Returns:
        A pandas DataFrame with the updated Elo ratings for each player.

    """
    def __init__(self, initial_elo=1500, max_elo_diff=800, decay_const=0.4):
        """
        Initialize a new instance of EloTransformer.

        Args:
            initial_elo (float, optional): The initial Elo rating for a new player. Defaults to 1500.
            max_elo_diff (float, optional): The maximum allowed Elo difference between two players. Defaults to 800.
            decay_const (float, optional): The decay constant used in the Elo rating calculation. Defaults to 0.4.
        """
        self.initial_elo = initial_elo
        self.max_elo_diff = max_elo_diff
        self.decay_const = decay_const
        self.param_grid = {'initial_elo': [1400, 1500, 1600],
                           'max_elo_diff': [700, 800, 900],
                           'decay_const': [0.3, 0.4, 0.5]}

    def transform(self, X, y=None):
        """
        Calculates the updated Elo ratings for each player based on match results.

        Args:
            X (pandas.DataFrame): A DataFrame containing the player ID, opponent ID, and other match statistics for each tennis match.

        Returns:
            pandas.DataFrame: The input DataFrame with two additional columns, 'elo_1' and 'elo_2', containing the updated Elo ratings for the two players.

        """
        player_elos = {}
        for index, row in X.iterrows():
            player_1 = row['player_id']
            player_2 = row['opponent_id']
            elo_1 = player_elos.get(player_1, self.initial_elo)
            elo_2 = player_elos.get(player_2, self.initial_elo)
            w_1 = row['running_total_victories_1']
            w_2 = row['running_total_victories_2']
            m_1 = row['running_total_matches_played_1']
            m_2 = row['running_total_matches_played_2']
            elo_diff = min(max(elo_1 - elo_2, -self.max_elo_diff), self.max_elo_diff)
            expected_1 = 1 / (1 + 10 ** (elo_diff / 400))
            expected_2 = 1 - expected_1
            decay_1 = 250.0 / ((5 + m_1) ** self.decay_const)
            decay_2 = 250.0 / ((5 + m_2) ** self.decay_const)
            new_elo_1 = elo_1 + decay_1 * (w_1 - expected_1)
            new_elo_2 = elo_2 + decay_2 * (w_2 - expected_2)
            player_elos[player_1] = new_elo_1
            player_elos[player_2] = new_elo_2
            X.loc[index, ['elo_1', 'elo_2']] = [new_elo_1, new_elo_2]
        return X

    @property
    def param_grid(self):
        """
        Get the hyperparameter grid to be used in model tuning.

        Args:
            None

        Returns:
            A generator of parameter dictionaries for GridSearchCV.

        """
        return ParameterGrid(self._param_grid)

# Assume you have loaded your data into a pandas dataframe called 'data'
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('winner', axis=1), data['winner'], test_size=0.2, random_state=42)

# Define your custom transformer
elo_transformer = EloTransformer()

# Define your pipeline
pipeline = Pipeline([
    ('elo', elo_transformer),
    # Add other pipeline steps as necessary
    ('logreg', LogisticRegression())
])

# Define the hyperparameter grid
param_grid = {
    'elo-transformer__initial_elo': [1400, 1500, 1600],
    'elo-transformer__max_elo_diff': [700, 800, 900],
    'elo-transformer__decay_const': [0.3, 0.4, 0.5],
    'logreg__C': [0.1, 1, 10]
}

# Define the grid search object
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)

# Fit the grid search object on the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and accuracy score
print("Best hyperparameters: ", grid_search.best_params_)
print("Accuracy score: ", grid_search.score(X_test, y_test))
