import random
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, ConnectionPatch

class Match:
    """Class representing a tennis match.

    Attributes:
        player1 (Player): The first player participating in the match.
        player2 (Player): The second player participating in the match.
        best_of (int): The number of sets required to win the match.
        current_set (int): The current set number in the match.
        current_game (int): The current game number in the current set.
        current_point (int): The current point number in the current game.
        game_win_counts (dict): A dictionary of game win counts for each player in the current set.
        server_on_points (list): A list of players who serve on each point in the current game.
        advantage (Player): The player with advantage in the current game, if any.
        serve_position (Tuple[float, float]): The current position of the player who is serving.
    """

    # Define score map
    score_map = {
        0: "0",
        1: "15",
        2: "30",
        3: "40",
        4: "Adv"
    }

    def __init__(self, player1, player2, best_of, court_surface, indoor):
        """
        Initializes a new Match object.

        Args:
            player1 (Player): The first player participating in the match.
            player2 (Player): The second player participating in the match.
            best_of (int): The number of sets required to win the match.
            court_surface (str): The surface of the court for the match.
            indoor (bool): Whether the match is played indoors. If indoor is False, then a weather condition must be specified.
            weather (str): The weather condition for the match. Required if indoor is False.
        """
        self.player1 = player1
        self.player2 = player2
        self.best_of = best_of
        self.court_surface = court_surface
        self.indoor = indoor
        self.weather = None
        if not self.indoor:
            self.weather = input("Enter the weather conditions: ")
        self.current_set = 1
        self.current_game = 1
        self.sets_won = {player1: 0, player2: 0}
        self.tiebreak = False
        self.current_point = 0
        self.advantage = None
        self.match_over = False
        self.set_over = False
        self.game_over = False
        self.serve_position = (-0.1, 0.2)

    def first_serve(self):
        """Randomly chooses a player to serve first and sets the initial serving player for the match.

        Returns:
            None.
        """
        first_serving_player = random.choice([self.player1, self.player2])
        self.set_serving_player = first_serving_player
        self.first_serving_player = first_serving_player

    def update_score(self, winner, serving_player, receiving_player):
        """
        Updates the score of the match or tiebreak based on the winner of the current point.

        Args:
            winner (Player): The player who won the current point.
            serving_player (Player): The player who served the current point.
            receiving_player (Player): The player who received the current point.
        """
        winner_score = winner.score
        loser = receiving_player if winner == serving_player else serving_player
        loser_score = loser.score

        if self.tiebreak:
            if winner_score < 6 and loser_score < 6:
                # Increment winner's score if less than 6
                winner.win_point()
            elif winner_score >= 6 and loser_score >= 6:
                # Handle deuce
                if winner_score > loser_score + 1:
                    # Winner has won the tiebreak
                    self.won_set(winner)
                else:
                    # No player has advantage yet
                    winner.win_point()
            else:
                # Winner has won the tiebreak
                self.won_set(winner)
        else:
            if winner_score == 3 and winner_score >= loser_score + 1:
                # Handle win without deuce
                self.won_game(winner, serving_player, receiving_player)
            elif winner_score < 3 and loser_score <= 3:
                # Increment winner's score if less than 40
                winner.win_point()
            elif winner_score >= 3 and loser_score >= 3:
                # Handle deuce
                if winner.advantage:
                    # Winner has advantage and wins the game
                    self.won_game(winner, serving_player, receiving_player)
                    winner.advantage = False
                    loser.score = 0
                    loser.advantage = False
                elif loser.advantage:
                    # Loser had advantage, but now the score is deuce again
                    winner.score = 3
                    loser.score = 3
                    winner.advantage = False
                    loser.advantage = False
                else:
                    # No player has advantage yet
                    winner.advantage = True
                    winner.score = 4

    def won_game(self, winner, serving_player, receiving_player):
        """
        Method to handle logic for when a player wins a game.

        Args:
            winner (Player): The player who won the game.
            serving_player (Player): The player who served during the game.
            receiving_player (Player): The player who received during the game.
        """
        print(f"\nPlayer {winner.name} wins game {self.current_game}!")
        self.game_over = True

        winner.increment_games()
        winner.reset_score()
        loser = receiving_player if winner == serving_player else serving_player
        loser.reset_score()
        self.current_point = 0

        if winner.games >= 6 and winner.games >= loser.games + 2:
            self.won_set(winner)
        elif winner.games == 6 and loser.games == 6:
            self.tiebreak = True
            print(f"Set {self.current_set} score is {serving_player.name} {serving_player.games} - {receiving_player.games} {receiving_player.name}\n")
        else:
            self.current_game += 1
            print(f"Set {self.current_set} score is {serving_player.name} {serving_player.games} - {receiving_player.games} {receiving_player.name}\n")

        self.advantage = None


    def won_set(self, winner):
        """
        Method to handle logic for when a player wins a set.

        Args:
            winner (Player): The player who won the set.
        """
        self.sets_won[winner] += 1

        # Alternate the player with the first serve at the start of the new set
        serving_player, receiving_player = (self.first_serving_player, self.player2) if self.current_set % 2 == 0 else (self.player1, self.first_serving_player)

        print(f"Player {winner.name} wins set {self.current_set} ({self.sets_won[self.player1]}-{self.sets_won[self.player2]})")

        if self.sets_won[winner] == self.best_of // 2 + 1:
            print(f"Player {winner.name} wins the match!")
            self.match_over = True
            self.set_over = True
            self.game_over = True
            self.tiebreak = False
            return

        self.current_set += 1
        self.current_game += 1
        self.player1.reset_score()
        self.player2.reset_score()
        self.player1.reset_set_score()
        self.player2.reset_set_score()
        self.set_serving_player = serving_player
        self.tiebreak = False

    def format_score(self, score):
        """Formats a player's score for display.

        Args:
            score (int): The player's score.

        Returns:
            str: The formatted score.
        """
        if score < 3:
            return self.score_map[score]
        elif score == 3:
            if self.advantage:
                return "Adv" if self.advantage.score > self.player1.score else "-40"
            else:
                return "40"
        else:
            return ""

    def simulate_set(self) -> None:
        """
        Simulates a set until a player wins 6 games with a lead of 2 or until a tiebreak occurs. Prints the
        winner of the set and updates the player stats.
        """
    
        # Alternate the player with the first serve at the start of each set
        serving_player, receiving_player = (self.first_serving_player, self.player2) if self.current_set % 2 == 0 else (self.player1, self.first_serving_player)

        self.serving_player = serving_player
        self.receiving_player = receiving_player

        while not self.set_over:
            self.game_over = False
            print(f"Starting Game {self.current_game} simulation:")
            self.simulate_game()

            # Check if a player has won the set
            winner, loser = (self.player1, self.player2) if self.player1.games > self.player2.games else (self.player2, self.player1)

            if winner.games == 6 and loser.games == 6:
                print(f"Set {self.current_set} is going to tiebreak")
                self.tiebreak = True
                self.simulate_game()

    def simulate_game(self) -> None:
        """
        Simulates a game until a player wins or the game goes to deuce. Prints the winner of each point and
        updates the score of the game and players accordingly.
        """
        if self.tiebreak and self.player1.score == 0 and self.player2.score == 0:
            serving_player = self.set_serving_player
        elif self.tiebreak and (self.player1.score + self.player2.score) % 2 == 1:
            serving_player = self.player1 if self.set_serving_player == self.player2 else self.player2
        else:
            if self.current_game % 2 == 0:
                serving_player, receiving_player = self.serving_player, self.player2 if self.serving_player == self.player1 else self.player1
            else:
                serving_player, receiving_player = self.serving_player, self.first_serving_player if self.serving_player == self.player2 else self.player2

        while not self.game_over:
            if self.tiebreak:
                print(f"\tTiebreak score: {self.player1.name} {self.player1.score} - {self.player2.score} {self.player2.name}")
            else:
                print(f"\tCurrent game score: {self.player1.name} {self.score_map[self.player1.score]} - {self.score_map[self.player2.score]} {self.player2.name}")

            if self.current_point % 2 == 0:
                court_position = (-0.1, 0.2)  # Right-hand side
            else:
                court_position = (-0.1, 0.8)  # Left-hand side
            
            probability_of_winning_point = None
            if serving_player.fault():
                print(f"\t\t{serving_player.name}'s first serve")
                probability_of_winning_point = serving_player.calculate_point_probability(opponent=receiving_player, court_position=court_position, weather_conditions="sunny", first_serve=True)
            elif serving_player.fault():
                print(f"\t\t{serving_player.name}'s second serve")
                probability_of_winning_point = serving_player.calculate_point_probability(opponent=receiving_player, court_position=court_position, weather_conditions="sunny", first_serve=False)
            else:
                print(f"\t\t{receiving_player.name} wins the point\n")
                if self.tiebreak:
                    self.update_score(receiving_player, serving_player, receiving_player)
                else:
                    self.update_score(receiving_player, serving_player, receiving_player)

            if probability_of_winning_point:
                if random.random() < probability_of_winning_point:
                    print(f"\t\t{serving_player.name} wins the point\n")
                    winner = serving_player
                else:
                    print(f"\t\t{receiving_player.name} wins the point\n")
                    winner = receiving_player

                self.update_score(winner, serving_player, receiving_player)

class Player:
    """Class representing a tennis player.

    Attributes:
        name (str): The name of the player.
        skill_level (float): The skill level of the player, as a value between 0 and 1.
        score (int): The player's current score in the game.
        recent_results (list): A list of the player's recent game results.
        games (int): The number of games won by the player in the current set.
        fault_count (int): The number of consecutive faults by the player in the current serve.
        match (Match): The match object that the player is participating in.
    """
    def __init__(self, name: str, skill_level: float, match: Match):
        """Constructor for the Player class.

        Args:
            name (str): The name of the player.
            skill_level (float): The skill level of the player, as a value between 0 and 1.
            match (Match): The match object that the player is participating in.
        """
        self.name = name
        self.skill_level = skill_level
        self.score = 0
        self.advantage = False
        self.recent_results = []
        self.games = 0
        self.fault_count = 0
        self.match = match

    def increment_games(self) -> None:
        """Increment the number of games won by the player in the current set."""
        self.games += 1

    def fault(self) -> bool:
        """Check if the player faulted on their serve and handle accordingly.

        Returns:
            bool: True if the player's serve is successful, False otherwise.
        """
        if random.random() < (1 - self.skill_level):
            if self.fault_count == 0:
                print(f"\t\t{self.name} faults!")
                self.fault_count = 1
            else:
                print(f"\t\t{self.name} double faults!")
                self.fault_count = 0
                return False
        else:
            self.fault_count = 0
            return True

    def win_point(self) -> None:
        """Increment the player's score by 1."""
        self.score += 1

    def reset_score(self) -> None:
        """Reset the player's game score to 0."""
        self.score = 0


    def reset_set_score(self) -> None:
        """Reset the player's set score to 0."""
        self.games = 0


    def calculate_point_probability(self, opponent: "Player", court_position: Tuple[float, float], weather_conditions: str, first_serve: bool) -> float:
        """Calculate the player's probability of winning the current point based on various factors.

        Args:
            opponent (Player): The player's opponent.
            court_position (Tuple[float, float]): The position of the players on the court, as a tuple of x and y coordinates.
            weather_conditions (str): The current weather conditions, represented as a string.
            first_serve (bool): Whether the serving player is using their first or second serve.

        Returns:
            float: The probability of the player winning the point, as a value between 0 and 1.
        """
        # Calculate base probability based on skill levels
        base_prob = (self.skill_level + (1 - opponent.skill_level)) / 2

        # Adjust probability based on court position
        x, y = court_position
        if y > 0.5:
            base_prob += 0.05 # this code will need to be updated to evaluate the player's serve performance from left and right hand side. also comparing to receiver's skill
        else:
            base_prob -= 0.05 # this code will need to be updated to evaluate the player's serve performance from left and right hand side. also comparing to receiver's skill
        
        # Adjust probability based on weather conditions
        if weather_conditions == "sunny":
            base_prob += 0.05
        elif weather_conditions == "rainy":
            base_prob -= 0.05

        # Adjust probability based on recent performance
        if len(self.recent_results) > 0:
            win_pct = sum([1 if r == "W" else 0 for r in self.recent_results]) / len(self.recent_results)
            if win_pct > 0.8:
                base_prob += 0.05
            elif win_pct < 0.4:
                base_prob -= 0.05

        # Adjust probability based on fault
        if not first_serve:
            base_prob -= 0.05

        # Add noise to the probability to account for unpredictability
        prob_with_noise = base_prob + random.gauss(0, 0.03)

        # Clip the probability to ensure it's within the range [0, 1]
        return max(0, min(1, prob_with_noise))

    def plot_court_position(self, court_position: Tuple[float, float]) -> None:
        """Plot the player's current court position on a tennis court.

        Args:
            court_position (Tuple[float, float]): The position of the players on the court, as a tuple of x and y coordinates.
        """
        # Get the court surface from the match object
        court_surface = self.match.court_surface

        # Define the court dimensions
        court_length = 78
        court_width = 36

        # Define the color based on the court surface
        if court_surface == 'clay':
            court_color = '#a9765d'
        elif court_surface == 'grass':
            court_color = '#008566'
        elif court_surface == 'hard':
            court_color = '#1E8FD5'
        else:
            raise ValueError("Invalid court surface type. Please enter 'clay', 'grass', or 'hard'.")

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Set the x and y limits of the axis
        ax.set_xlim(-10, court_length+10)
        ax.set_ylim(-10, court_width+10)

        # Remove the tick marks on the x and y axis
        ax.set_xticks([])
        ax.set_yticks([])

        # Add the court lines
        service_boxes = [
            Rectangle((0, 4.5), 18, 27, linewidth=1, edgecolor='white', facecolor='none'),
            Rectangle((court_length - 18, 4.5), 18, 27, linewidth=1, edgecolor='white', facecolor='none'),
            Rectangle((0, 0), court_length, 4.5, linewidth=1, edgecolor='white', facecolor='none'),
            Rectangle((0, court_width-4.5), court_length, 4.5, linewidth=1, edgecolor='white', facecolor='none') 
        ]
        # Define the center line
        h_center_line = plt.Line2D([18, court_length - 18], [court_width/2, court_width/2], linewidth=1, color='white')
        v_center_line = plt.Line2D([court_length / 2, court_length / 2], [0, court_width], linewidth=1, color='white')

        # Add the court lines and service boxes to the axis
        for box in service_boxes:
            ax.add_patch(box)
            # Add the length of each service box as a label
            ax.text(box.get_x() + box.get_width() / 2, box.get_y() - 0.5, f'{box.get_width()} ft x {box.get_height()} ft', ha='center', color='white')

        # Add the center line to the axis
        ax.add_artist(h_center_line)
        ax.add_artist(v_center_line)

        # Add the length of the center line as a label
        ax.text(court_length / 2, court_width / 2 - 0.5, f'{court_length - 36} ft', ha='center', color='white')

        # Add the player's position to the axis as a blue dot
        ax.plot(court_position[0]*court_length, court_position[1]*court_width, 'ro', markersize=15, markeredgecolor='black', markeredgewidth=2)

        # Add the player's name as a label
        ax.text(court_position[0]*court_length + 1, court_position[1]*court_width + 1, self.name, ha='left', va='bottom', color='black', fontsize=12)

        # Set the facecolor of the axes to the court color
        ax.set_facecolor(court_color)

        # Show the plot
        plt.show()

def simulate_match(player1: Player, player2: Player, best_of: int, court_surface: str, indoor: False, weather: str) -> None:
    """
    Simulates a tennis match between two players until a player wins the required number of sets.

    Args:
        player1 (Player): The first player.
        player2 (Player): The second player.
        best_of (int): The number of sets required to win the match. Must be either 3 or 5.
        court_surface (str): The surface of the court, either 'clay', 'grass', or 'hard'.
        indoor (bool): Boolean value to indicate if match is being held inside or outside. 
        weather (str): The weather for the match
    Raises:
        ValueError: If best_of is not 3 or 5.
        ValueError: If court_surface is not 'clay', 'grass', or 'hard'.

    Returns:
        None
    """
    if best_of not in [3, 5]:
        raise ValueError("best_of must be either 3 or 5")
    if court_surface not in ['clay', 'grass', 'hard']:
        raise ValueError("Invalid court surface type. Please enter 'clay', 'grass', or 'hard'.")

    match = Match(player1, player2, best_of, court_surface, indoor, weather)
    match.first_serve()

    print(f"{match.first_serving_player.name} is serving first in the match")

    while not match.match_over:
        match.simulate_set()



