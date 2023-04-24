#!/usr/bin/env python

import pandas as pd
from helper_functions import print_elapsed_time
import os

print("Importing Matches Data")
print_elapsed_time()
data = pd.read_csv("all_matches.csv")
data.head()

print("Importing Tournaments Data")
print_elapsed_time()
tournaments = pd.read_csv("all_tournaments.csv")
tournaments.head()

#getting only the tournaments we need
print("Filter out old tournaments")
print_elapsed_time()
tourneys = tournaments[tournaments.masters >= 100]
tourneys = tourneys[tourneys.year>=2000]
tourneys

#making a new dataset with tourney_id and year
print("Make tourney id")
print_elapsed_time()
tourneys_id = [tourneys.year,tourneys.tournament] 
tourneys_id = pd.concat(tourneys_id,axis=1)
tourneys_id

#selecting only the matches from the tournaments previously selected
print("Select matches from tournments we previously filtered")
print_elapsed_time()
data = data[data.year >= 2000]
data = data[data.masters >= 100]

#merging tourney_id and year 
t_years = tourneys_id.year
t_names = tourneys_id.tournament
t_info = t_names + t_years.astype(str)
t_info = t_info.reset_index(drop=True)
t_info

print("Creating final dataframe for matches")
print_elapsed_time()
#creating final dataframe for storing matches
final_df = pd.DataFrame(index = range(0,352512),columns = [#general info
                                   'start_date',
                                   'end_date',
                                   'location',
                                   'court_surface',
                                   'prize_money', #not relevant
                                   'currency', #not relevant
                                   'year',
                                   'player_id',
                                   'player_name', #not relevant, we only need id
                                   'opponent_id',
                                   'opponent_name', #not relevant, we only need id
                                   'tournament', 
                                   'round', 
                                   'num_sets',
                                   'doubles', #doubles need to be excluded
                                   'masters',
                                   'round_num',
                                   'duration',
                                   'total_points',
                                   #player 1 info, all relevant
                                   'sets_won_1',
                                   'games_won_1',
                                   'games_against_1',
                                   'tiebreaks_won_1',
                                   'tiebreaks_total',
                                   'serve_rating_1',
                                   'aces_1', 
                                   'double_faults_1', 
                                   'first_serve_made_1',
                                   'first_serve_attempted_1',
                                   'first_serve_points_made_1',
                                   'first_serve_points_attempted_1',
                                   'second_serve_points_made_1',
                                   'second_serve_points_attempted_1',
                                   'break_points_saved_1',
                                   'break_points_against_1', 
                                   'service_games_won_1', 
                                   'return_rating_1',
                                   'first_serve_return_points_made_1', 
                                   'first_serve_return_points_attempted_1',
                                   'second_serve_return_points_made_1',
                                   'second_serve_return_points_attempted_1',
                                   'break_points_made_1',
                                   'break_points_attempted_1',
                                   'return_games_played_1',
                                   'service_points_won_1',
                                   'service_points_attempted_1',
                                   'return_points_won_1',
                                   'return_points_attempted_1',
                                   'total_points_won_1',
                                   'player_1_victory',
                                   'retirement_1',
                                   'seed', 
                                   'won_first_set_1',
                                   'nation_1',
                                   #player 2 info, all relevant
                                   'sets_won_2',
                                   'games_won_2',
                                   'games_against_2',
                                   'tiebreaks_won_2',
                                   'serve_rating_2',
                                   'aces_2', 
                                   'double_faults_2', 
                                   'first_serve_made_2',
                                   'first_serve_attempted_2',
                                   'first_serve_points_made_2',
                                   'first_serve_points_attempted_2',
                                   'second_serve_points_made_2',
                                   'second_serve_points_attempted_2',
                                   'break_points_saved_2',
                                   'break_points_against_2', 
                                   'service_games_won_2', 
                                   'return_rating_2',
                                   'first_serve_return_points_made_2', 
                                   'first_serve_return_points_attempted_2',
                                   'second_serve_return_points_made_2',
                                   'second_serve_return_points_attempted_2',
                                   'break_points_made_2',
                                   'break_points_attempted_2',
                                   'return_games_played_2',
                                   'service_points_won_2',
                                   'service_points_attempted_2',
                                   'return_points_won_2',
                                   'return_points_attempted_2',
                                   'total_points_won_2',
                                   'player_2_victory',
                                   'retirement_2',
                                   'won_first_set_2',                                   
                                   'nation_2'])
    
final_df.fillna(0)

#creating a dictionary of dataframes, every df is a torunament
tourney_dfs = {}

#remove doubles
print("Remove doubles matches")
print_elapsed_time()
data = data.loc[data["doubles"]=='f']
data = data.reset_index(drop=True)
i=0

#creating individual dataframes for every tournament and storing them into a dictionary tourneys_dfs
print("Create DF for each tournmanet")
print_elapsed_time()
for year,tournament in zip(t_years,t_names) :
    tourney_dfs[t_info[i]] = data.loc[(data["year"]==year) & (data["tournament"]==tournament)]
    i+=1

#test
data = data.reset_index(drop=True)

#getting number of total rows in tourney_dfs
k=0
for key,value in tourney_dfs.items():
    j=len(value)
    k = k + j
    
final_dfs = {}
i=0
print("Create structure for final_df")
print_elapsed_time()
for  key,value in tourney_dfs.items() :
    final_dfs[key] = pd.DataFrame(index = range(0,len(value)),columns = [#general info
                                   'start_date',
                                   'end_date',
                                   'location',
                                   'court_surface',
                                   'prize_money', #not relevant
                                   'currency', #not relevant
                                   'year',
                                   'player_id',
                                   'player_name', #not relevant, we only need id
                                   'opponent_id',
                                   'opponent_name', #not relevant, we only need id
                                   'tournament', 
                                   'round', 
                                   'num_sets',
                                   'doubles', #doubles need to be excluded
                                   'masters',
                                   'round_num',
                                   'duration',
                                   'total_points',
                                   #player 1 info, all relevant
                                   'sets_won_1',
                                   'games_won_1',
                                   'games_against_1',
                                   'tiebreaks_won_1',
                                   'tiebreaks_total',
                                   'serve_rating_1',
                                   'aces_1', 
                                   'double_faults_1', 
                                   'first_serve_made_1',
                                   'first_serve_attempted_1',
                                   'first_serve_points_made_1',
                                   'first_serve_points_attempted_1',
                                   'second_serve_points_made_1',
                                   'second_serve_points_attempted_1',
                                   'break_points_saved_1',
                                   'break_points_against_1', 
                                   'service_games_won_1', 
                                   'return_rating_1',
                                   'first_serve_return_points_made_1', 
                                   'first_serve_return_points_attempted_1',
                                   'second_serve_return_points_made_1',
                                   'second_serve_return_points_attempted_1',
                                   'break_points_made_1',
                                   'break_points_attempted_1',
                                   'return_games_played_1',
                                   'service_points_won_1',
                                   'service_points_attempted_1',
                                   'return_points_won_1',
                                   'return_points_attempted_1',
                                   'total_points_won_1',
                                   'player_1_victory',
                                   'retirement_1',
                                   'seed_1', 
                                   'won_first_set_1',
                                   'nation_1',
                                   #player 2 info, all relevant
                                   'sets_won_2',
                                   'games_won_2',
                                   'games_against_2',
                                   'tiebreaks_won_2',
                                   'serve_rating_2',
                                   'aces_2', 
                                   'double_faults_2', 
                                   'first_serve_made_2',
                                   'first_serve_attempted_2',
                                   'first_serve_points_made_2',
                                   'first_serve_points_attempted_2',
                                   'second_serve_points_made_2',
                                   'second_serve_points_attempted_2',
                                   'break_points_saved_2',
                                   'break_points_against_2', 
                                   'service_games_won_2', 
                                   'return_rating_2',
                                   'first_serve_return_points_made_2', 
                                   'first_serve_return_points_attempted_2',
                                   'second_serve_return_points_made_2',
                                   'second_serve_return_points_attempted_2',
                                   'break_points_made_2',
                                   'break_points_attempted_2',
                                   'return_games_played_2',
                                   'service_points_won_2',
                                   'service_points_attempted_2',
                                   'return_points_won_2',
                                   'return_points_attempted_2',
                                   'total_points_won_2',
                                   'player_2_victory',
                                   'retirement_2',
                                   'won_first_set_2',                                   
                                   'nation_2'])
    final_dfs[key].fillna(0)
    i+=1

print("Write match data into each dataframe")
print_elapsed_time()
#populating dataframes with matches data
for key, value in tourney_dfs.items():

    main = value
    temp = value
    
    inner = main.merge(temp, how='inner', left_on=['player_id', 'opponent_id'], right_on=['opponent_id', 'player_id'])
    
    final_dfs[key] = final_dfs[key].assign(
        start_date=inner['start_date_x'],
        end_date=inner['end_date_x'],
        location=inner['location_x'],
        court_surface=inner['court_surface_x'],
        prize_money=inner['prize_money_x'],
        currency=inner['currency_x'],
        year=inner['year_x'],
        player_id=inner['player_id_x'],
        player_name=inner['player_name_x'],
        opponent_id=inner['opponent_id_x'],
        opponent_name=inner['opponent_name_x'],
        tournament=inner['tournament_x'],
        round=inner['round_x'],
        num_sets=inner['num_sets_x'],
        doubles=inner['doubles_x'],
        masters=inner['masters_x'],
        round_num=inner['round_num_x'],
        duration=inner['duration_x'],
        total_points=inner['total_points_x'],
        sets_won_1=inner['sets_won_x'],
        games_won_1=inner['games_won_x'],
        games_against_1=inner['games_against_x'],
        tiebreaks_won_1=inner['tiebreaks_won_x'],
        tiebreaks_total=inner['tiebreaks_total_x'],
        serve_rating_1=inner['serve_rating_x'],
        aces_1=inner['aces_x'],
        double_faults_1=inner['double_faults_x'],
        first_serve_made_1=inner['first_serve_made_x'],
        first_serve_attempted_1=inner['first_serve_attempted_x'],
        first_serve_points_made_1=inner['first_serve_points_made_x'],
        first_serve_points_attempted_1=inner['first_serve_points_attempted_x'],
        second_serve_points_made_1=inner['second_serve_points_made_x'],
        second_serve_points_attempted_1=inner['second_serve_points_attempted_x'],
        break_points_saved_1=inner['break_points_saved_x'],
        break_points_against_1=inner['break_points_against_x'],
        service_games_won_1=inner['service_games_won_x'],
        return_rating_1=inner['return_rating_x'],
        first_serve_return_points_made_1=inner['first_serve_return_points_made_x'],
        first_serve_return_points_attempted_1=inner['first_serve_return_points_attempted_x'],
        second_serve_return_points_made_1=inner['second_serve_return_points_made_x'],
        second_serve_return_points_attempted_1=inner['second_serve_return_points_attempted_x'],
        break_points_made_1=inner['break_points_made_x'],
        break_points_attempted_1=inner['break_points_attempted_x'],
        return_games_played_1=inner['return_games_played_x'],
        service_points_won_1=inner['service_points_won_x'],
        service_points_attempted_1=inner['service_points_attempted_x'],
        return_points_won_1=inner['return_points_won_x'],
        return_points_attempted_1=inner['return_points_attempted_x'],
        total_points_won_1=inner['total_points_won_x'],
        player_1_victory=inner['player_victory_x'],
        retirement_1=inner['retirement_x'],
        seed_1=inner['seed_x'],
        won_first_set_1=inner['won_first_set_x'],
        nation_1=inner['nation_x'],
        
        sets_won_2=inner['sets_won_y'],
        games_won_2=inner['games_won_y'],
        games_against_2=inner['games_against_y'],
        tiebreaks_won_2=inner['tiebreaks_won_y'],
        serve_rating_2=inner['serve_rating_y'],
        aces_2=inner['aces_y'],
        double_faults_2=inner['double_faults_y'],
        first_serve_made_2=inner['first_serve_made_y'],
        first_serve_attempted_2=inner['first_serve_attempted_y'],
        first_serve_points_made_2=inner['first_serve_points_made_y'],
        first_serve_points_attempted_2=inner['first_serve_points_attempted_y'],
        second_serve_points_made_2=inner['second_serve_points_made_y'],
        second_serve_points_attempted_2=inner['second_serve_points_attempted_y'],
        break_points_saved_2=inner['break_points_saved_y'],
        break_points_against_2=inner['break_points_against_y'],
        service_games_won_2=inner['service_games_won_y'],
        return_rating_2=inner['return_rating_y'],
        first_serve_return_points_made_2=inner['first_serve_return_points_made_y'],
        first_serve_return_points_attempted_2=inner['first_serve_return_points_attempted_y'],
        second_serve_return_points_made_2=inner['second_serve_return_points_made_y'],
        second_serve_return_points_attempted_2=inner['second_serve_return_points_attempted_y'],
        break_points_made_2=inner['break_points_made_y'],
        break_points_attempted_2=inner['break_points_attempted_y'],
        return_games_played_2=inner['return_games_played_y'],
        service_points_won_2=inner['service_points_won_y'],
        service_points_attempted_2=inner['service_points_attempted_y'],
        return_points_won_2=inner['return_points_won_y'],
        return_points_attempted_2=inner['return_points_attempted_y'],
        total_points_won_2=inner['total_points_won_y'],
        player_2_victory=inner['player_victory_y'],
        retirement_2=inner['retirement_y'],
        won_first_set_2=inner['won_first_set_y'],
        nation_2=inner['nation_y']
    )

#getting list of dataframe names
frames=[]
for key,value in final_dfs.items():
    frames.append(value)
    
#getting final dataframe
final_df = pd.concat(frames)

print("Save final_kaggle_dataset CSV and PKL file")
print_elapsed_time()
#saving final 
final_df.to_csv('final_kaggle_dataset.csv')

#saving final
final_df.to_pickle('final_kaggle_dataset.pkl')

#saving each tournament dataframe to a specific .csv file
print("Save each tournaments data to it's own CSV")
print_elapsed_time()
directory = './Tournaments_Data'
if not os.path.exists(directory):
    os.makedirs(directory)

os.chdir(directory)
for key,value in final_dfs.items():
    csv_name = str(key) + ".csv"
    if len(value) > 0:
        value.to_csv(csv_name)
    else:
        pass
    
    csv_name=''

os.chdir('../')