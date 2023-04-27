#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os

original_dir = os.getcwd()
os.chdir('./Tournaments_Data')
files = os.listdir(cwd)

#concatenate all tournament csv's
first_file = files[0]
final_df = pd.read_csv(first_file, index_col=0)
for file in files:
    if str(file) == first_file:
        continue
    temp = pd.read_csv(str(file), index_col=0)
    final_df = pd.concat([final_df,temp])
    temp = None

final_df

#turn the start_date and end_date columns into date format
# yy-mm-dd is the format used in the dataset
final_df.head()
final_df['start_date'] = pd.to_datetime(final_df['start_date'], format = '%Y-%m-%d')
final_df.head()

os.chdir(original_dir)
final_df.to_csv('final_df.csv',index=False)

final_df_2 = pd.read_csv('final_df.csv',parse_dates = [3,4],infer_datetime_format=True)
final_df_2 =  final_df_2.sort_values(by='start_date')
final_df_2.to_csv('final_df.csv',index=False)