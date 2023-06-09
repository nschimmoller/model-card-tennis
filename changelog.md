# Changelog for Version 0.1.1 (Released on 2023-04-27)

All notable changes to this project will be documented in this file. This project adheres to Semantic Versioning and uses the Keep a Changelog format.

## Tennis_Model.ipynb

#### Added

- Tennis_Model.ipynb to the GitHub repository.

#### Changed

- Updated the format of the Changelog entry for Tennis_Model.ipynb to conform to the Semantic Versioning and Keep a Changelog format.

#### Description

- Match Ingestion and Processing: Performs several data processing tasks on two datasets related to tennis matches and tournaments. The processed data is then saved as a CSV and a Pickle file, and also as separate CSV files for each tournament.
- Removing Duplicate Records: Removes duplicate rows in a dataset of tennis match results and saves the cleaned dataset as a CSV file.
- Player Data Ingestion and Processing: Reads in data about all tennis players and the ATP players from separate CSV files, and merges them to create a final player dataset. It then filters out players with missing information, cleans up the player names, and exports the resulting cleaned dataset to two new files: players_data.csv and players_data.pkl.
- Clean Player Data: Imports two different datasets - matches data and players data - and merges them based on the player ID. After the merge, it filters out players born before 1960 and exports the remaining player data to a CSV file.
- Calculate Elo: Calculates the Elo ratings for each player based on their past matches and plots the Elo rating for a specific player over time.
- The notebook also includes explanations of what each code section does and how to run it.

## Kaggle_Tennis_Data_PreProcessing.py

### Changed

- Code optimization: The code has been optimized to remove redundancies and unnecessary code, resulting in a more streamlined and efficient process.
- Inline commenting: The code has more inline comments, which makes the code easier to read and understand.
- Python best practices:
  - Used appropriate variable names for better readability and understanding of the code.
  - Avoided unnecessary loops and used list comprehension for better performance.
  - Encapsulated related code into functions for better organization and reusability.
- Readability: The code is more readable than the "kaggle_extraction" code due to better formatting, indentation, and consistent use of spacing.

## Remove_Duplicate_Matches.py

### Changed

- Optimized code for better performance:
  - The remove_doubles function is now defined to identify and remove duplicate matches based on player ids and opponent ids.
  - The function uses list comprehension to generate indexes of duplicate matches and drop them.
  - The function also uses reset_index to reset the indexes in the data after duplicates have been removed.
  - The function now returns the cleaned data.
- The code is now well-commented and more readable:
  - Inline comments have been added to explain what each line of code does.
  - Variable names are more descriptive

## fix_players_name.py

Optimized code for better performance:

- The function `remove_old_players` has been added to remove players born before 1960 from the player data set.
- The function `find_needed_players` has been added to find the players that are in the matches data set but not in the player data set.
- The code is now well-commented and more readable:

    - Inline comments have been added to explain what each line of code does.
    - Variable names are more descriptive, making it easier to understand what each variable represents.
    - Functions have been defined for better organization and clarity.

# Changelog for Version 0.1.1 (Released on 2023-05-01)

All notable changes to this project will be documented in this file. This project adheres to Semantic Versioning and uses the Keep a Changelog format.

## simulate_tennis_match.py 

### Added

- New module for simulating a tennis match between two players
- The module allows for customization of the match parameters (e.g. number of sets, court surface, weather conditions) and outputs the final match score and winner.

## GETTINGSTARTED.md

### Added
- Added document for new users to get started with this repository, assuming no prior python experience
- Includes instructions for downloading Python directly or downloading via Anaconda
- Includes instructions for both Mac and Windows
- Includes instructions on how to access GitHub repo either through Git or a web browser

# Changelog for Version 0.1.1 (Released on 2023-05-06)

All notable changes to this project will be documented in this file. This project adheres to Semantic Versioning and uses the Keep a Changelog format.

## Entire Repo

### Changed
- Files that I have modified are now all lower case as opposed to camel case.
- Updated `.gitignore` file to ignore large files in `/tennis_betting_ml/data` folder

## kaggle_tennis_data_preprocessing.py

### Added
- Ability to perform operations on incremental rows
- Implementation of logging using the `setup_logger` function from `helper_functions.py`

### Changed
- Renamed `match_identifiers` to `all_matches` to improve code clarity
- Updated column names in `all_matches` to be more descriptive
- Refactored `all_matches` code for readability
- Updated tournaments and `all_matches` filtering criteria to exclude tournaments and matches before 2000
- Renamed tournaments unique identifier column from `tourney_date_id` to `tournament_id`
- Renamed matches unique identifier column from `match_id` to `match_id_player_1` and added a corresponding `match_id_player_2` column
- Removed doubles matches from `all_matches` data
- Removed incomplete matches from `all_matches` data
- Sort `all_matches` by `start_date` ascending
- Added `running_total_matches_played` column to `all_matches` data
- Converted `player_victory` column in `all_matches` data to 1s and 0s
- Added `running_total_victories` column to `all_matches` data
- Split `all_matches` data into two dataframes, `player_1_matches` and `player_2_matches`
- Merged `player_1_matches` and `player_2_matches` to create `final_dataset`
- Renamed columns in `final_dataset`
- Reordered columns in `final_dataset`
- Sorted `final_dataset` by `player_id`, `year`, `start_date`, and `round_num_1`
- Saved `final_dataset` as a CSV file and a pickle file
- Updated `README.md` to reflect current code functionality

### Removed
- None
