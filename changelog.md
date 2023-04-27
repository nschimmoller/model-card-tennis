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

Best practices for naming conventions and style have been followed:

- Variable and function names use snake_case.
- The code follows PEP 8 style guidelines.
