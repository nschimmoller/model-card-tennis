# Introduction

I'm working to build an OpenSource Tennis betting algorithim that can ideally predict the outcomes of singles matches in ATP, WTA, and the ATP Challenger Series. After doing some initial research it does not look like such an algorithim currently exists ready to use out of the box so some work is needed. 

I did find a [Tennis-Betting-ML Model written by GitHub user BrandonPolistirolo](https://github.com/BrandoPolistirolo/Tennis-Betting-ML) that should be able to serve as the basis for a production model. 

However, the model itself is predicated off of a large [Kaggle Tennis Dataset](https://www.kaggle.com/ehallmar/a-large-tennis-dataset-for-atp-and-itf-betting?select=all_matches.csv) that is not being updated. As an alternative there is an [OpenSource Tennis_ATP Dataset by JeffSackmann](https://github.com/JeffSackmann/tennis_atp) that could prove useful.

# Next Steps

## Reproduce Results from Initial Tennis-Betting-ML Model

Before spending time swapping out the Kaggle ATP & ITF Dataset for the GitHub Tennis ATP dataset I want to ensure that the Tennis ATP dataset has the necessary datapoints that proved useful in predicting the outcomes of matches. 

Therefore, I am slowly trying to replicate the results reported in the Tennis-Betting-ML Model to check for those data points that were of high SHAP values. 

### Difficulties Reproducing

For as good of an understanding it seems the original author had on DataScience principles in building and training a model, their usage of the Pandas library leaves a lot to be desired. Inefficient code leads to long run times, which will not scale to a production model. 

I am reading through each file to determine the proper order of their usage and updating portions of the code that run slowly.

# What I've learned so far

1. You need to download all_matches, all_players, and all_tournaments data from the [Kaggle Dataset](https://www.kaggle.com/ehallmar/a-large-tennis-dataset-for-atp-and-itf-betting?select=all_matches.csv). 
 * These are too large to include in the repo, so you'll have to reproduce locally
2. I'm pretty sure that original author also got the atp_players dataset from [Tennis_ATP](https://github.com/JeffSackmann/tennis_atp). 
 * I've included this in the repo

# Order of Operation

1. `Kaggle_Tennis_Data_PreProcessing.py` goes first
 * I've updated some of this code to run more efficiently, but have not moved it to the file yet. It is currently in the `tennis_model.ipynb` file
2. Next is `Remove Double Matches.py`
 * I've updated some of this code to run more efficiently, but have not moved it to the file yet. It is currently in the `tennis_model.ipynb` file
 * The original dataset has a row per player and match so Djokovic vs Nadal and Nadal vs Djokovic. 
 * The first file joined these two matches together, this one removes the duplicate data
3. Next is `Players_Data_PreProc.py`
 * I have updated this file with more efficient code
4. Next is `Players_names_fix.py`
* I have updated this file with more efficient code
5. Next is `Features_extraction_ELO_Rankings.py`
* I have updated this file with more efficient code, but have not yet finished doing so.

The remaining order of operation is still a mystery to me