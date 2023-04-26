#!/usr/bin/env python

import pandas as pd
from helper_functions import setup_logger

# Set up the logger
logger = setup_logger(__name__, 'my_log_file.log')
logger.info('Starting script execution')

try:
    from helper_functions import print_elapsed_time
	
	# Read in all player data
	print_elapsed_time('Reading in all player data')
	data = pd.read_csv("all_players.csv")

	# Read in ATP player data, format date of birth
	print_elapsed_time('Reading in atp player_data')
	players_data = pd.read_csv("atp_players.csv",header=0,names=['id','name','surname','hand','dob','country','height','wikidata_id'])
	players_data['dob'] = pd.to_datetime(players_data['dob'], format='%Y%m%d', errors='coerce')

	# Replace underscore with "No" in player_id column
	print_elapsed_time('Add underscore to name')
	data.loc[data['player_id'].str.contains('_', na=False), 'player_id'] = 'No'

	# Filter out rows with player_id "No"
	print_elapsed_time("Filter out missing player ids")
	data = data.loc[data['player_id'] != "No"]

	# Set cleaned_players_data equal to data
	cleaned_players_data = data

	# Create a new player_id column by concatenating name and surname
	# Then, set all characters to lowercase and replace spaces with dashes
	print_elapsed_time('Clean player name')
	players_data['player_id'] = " "
	players_data['player_id'] = players_data['name'].replace(" ","-") + "-" + players_data['surname'].replace(' ','-')
	players_data['player_id'] = players_data['player_id'].str.lower()

	# Join cleaned_players_data and players_data on player_id
	print_elapsed_time('Join clean and original player data')
	final_players = cleaned_players_data.merge(players_data,left_on='player_id',right_on='player_id',how='outer')

	# Remove rows with missing date of birth (DOB)
	# Then, format DOB as a datetime object
	print_elapsed_time('Remove players with missing date of birth (DOB), and format as date')
	final_players = final_players.loc[(final_players['dob'].isna()) == False]
	final_players['dob'] = pd.to_datetime(final_players['dob'], errors='coerce')

	# Reset index
	final_players = final_players.reset_index(drop=True)

	# Drop useless columns and save as CSV and Pickle file
	print_elapsed_time('Export data to players_data.csv and players_data.pkl')
	final_players=final_players.drop(['country_x','id','name','surname'],axis=1)
	final_players.to_csv('players_data.csv', index=False)
	final_players.to_pickle('players_data.pkl')

    logger.info("Exiting")
    sys.exit(0)

except Exception as e:
    logger.exception("An error occurred during script execution")
    sys.exit(1)
