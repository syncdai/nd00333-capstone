# March Madness Matchup Predictor

Use AzureML to train a classifier on the last decade of NCAA college basketball team stats to predict which team in a match will win.  AutoML and Hyperdrive models are compared by accuracy, and the best of the two classifiers is deployed and consumed as an endpoint.

![capstone_arch](https://user-images.githubusercontent.com/80217508/129452199-86cde1de-302c-4ef4-8d1d-d17296e448d7.png)

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
NCAA data was collected by the Akkio company.  It records stats from both teams playing against each other in March Madness tournaments dating all the way back to 1988, of which I only use a subset of both years and stats.  Rows are duplicated because they say it reduces bias towards higher seeded teams being ordered as Team1.

### Task
The task is to train a classifier (effectively binary) that given some key features for each NCAA team in a match, it will predict either Team1 (0) or Team2 (1) as the winner.  Only the last decade (2010 - 2019) is considered, because prior to that there were gaps in statistics.  After ignoring obvious predictors (e.g. final score) and using AutoML featurization on all 50+ remaining columns, I pruned the data to the top 10 features, or top 5 stats from each team.  I was thinking ahead to how I'd have to build a query to the model later, and definitely not going to look up 50+ stats, but it also did coincidentally improve AutoML accuracy to use a compact dataset.

![top10_features](https://user-images.githubusercontent.com/80217508/129283941-caa4f0f6-afff-4ebd-81aa-4f54319b6a36.png)

### Access
Data is downloaded as CSV from external URL: https://s3.amazonaws.com/static.akk.io/Akkio-NCAAHistoricalData.csv

Preprocessing steps are mostly the same for AutoML in Jupyter notebook (orange) and the Hyperdrive training script (blue).  Create a TabularDataset from URL, convert into a Pandas Dataframe, and filter on year and features as determined previously.  Afterwards, AutoML only needs the dataset registered to blobstore, whereas for Hyperdrive I convert the winner column to binary and create a KFold object for cross validation to support SKLearn Logistic Regression.

![access_data](https://user-images.githubusercontent.com/80217508/129452469-fa5dc9dc-b69e-4781-b71d-874251cd20c6.png)

## Automated ML
Classification with 10-fold cross validation:
- classifier because the problem looked like binary classification to me
- 10-fold because it lines up neatly with 10 years of historical data

Accuracy as the primary metric, because I mainly care about predicting the correct outcome, and the data was structured to be balanced towards Team1 vs Team2.

Timeout and concurrency were just a balancing act of using my lab session resources effectively.

### Results
For once, VotingEnsemble was merely tied for best accuracy against logistic regression at 0.85952, in which case AzureML chooses the simpler model to report as best.  We can see in the screenshot below that it was SKLearn LogisticRegression, with C = 1.2, and I was surprised it used multinomial classes.  I think, but I'm not sure, that cross validation in AzureML randomizes the data, so I might be able to improve accuracy if I setup custom folds using cv_split_column_names after encoding each year into separate cv columns.

![automl_RunDetails](https://user-images.githubusercontent.com/80217508/129283962-03c9fade-0692-451a-aab4-17b6edff17cd.png)
![automl_BestModel](https://user-images.githubusercontent.com/80217508/129283967-dce8d048-9e34-439b-8f2c-a8b058506d2e.png)

## Hyperparameter Tuning
I chose to target my hyperparameter search using lessons learned from AutoML.  Namely, SKLearn LogisticRegression with Bayesian sampling on what I thought were more promising parameters.  I took note of parameters from runner up logistic regression models to observe C = {1.7575, 5.4286, 0.12648, 2.5595}, penalty = {L1, L2}, and decided to avoid multinomial classification altogether (by substituting Team1, Team2 for 0, 1).
- C : 1.5 - 5.5 continuous
- max_iter : 500, 1000, 2000 -- C was on the higher end, so I figured to try more iterations to give time to converge
- penalty : L1, L2
- solver : liblinear, saga -- I picked all the ones that supported both L1 and L2, which apparently were just these 2

### Results
Accuracy was computed as the mean of scores from 10-folds, and highest achieved was 0.86746, which was better than AutoML.  Parameters were C = 5.1, max_iter = 2000, penalty = L1, solver = liblinear.

![hyper_RunDetails](https://user-images.githubusercontent.com/80217508/129283989-7397a2ed-b461-455a-8706-160fa27211d1.png)
![hyper_BestModel](https://user-images.githubusercontent.com/80217508/129283991-0295b463-562b-4019-af28-466536e0da64.png)

I also thought maybe there was a side effect of Bayesian sampling, where when I compared the top25 vs bottom25 models sorted by accuracy there seemed to be a preference towards specific parameters: C > 4.0, max_iter > 500, L1, liblinear.  It's worth noting that even the bottom25 were better than my best AutoML model.

![hyper_compare25](https://user-images.githubusercontent.com/80217508/129235253-bbfa1cad-1b3d-4f56-9a41-36ae32a39e11.png)

With this in mind, if I were to try to improve on the model, I would try to draw conclusions based on my results and refine a different search.  For example, fix L1 liblinear and try Random sampling with a very tight Bandit policy on new ranges of C, maybe this time below 1.0 to account for that 0.12648 finding from earlier.

## Model Deployment
The entry script converts input data into a 10 column dataframe, which works out to requiring a list containing a dict with 10 entries.
```
[
    {
        "G1": 30,
        "G2": 32,
        "Seed1": 1,
        "Seed2": 1,
        "PF1": 17.1,
        "PF2": 16.3,
        "3P%1": 0.413,
        "3P%2": 0.368,
        "TOV1": 11.5,
        "TOV2": 12.0,
    }
]
```
Stats ending in 1 refer to Team1, and the entire row is classified as [0] by the model if it thinks they win.

Stats ending in 2 refer to Team2, and the entire row is classified as [1] by the model if it thinks they win.

Meanings of each stat, from top to bottom, are Games played, bracket Seed, Personal Fouls per game, 3 Pointer percent success, and Turnovers per game.  Those last three actually give some unexpected insight into what one might consider differentiators from "perfect play" (guaranteed 2 points per possession) in a game of basketball.

Screenshot of model enpoint:
![hyper_endpoint](https://user-images.githubusercontent.com/80217508/129284011-771ee63a-d543-4cfc-a6e8-b712ad27b939.png)

## Screen Recording
https://drive.google.com/file/d/1xwWScIA8drdHSKGtThvHVmxNUeOBQAWd/view?usp=sharing

Note that the model used in the screencast differs slightly from the one in the screenshots, due to rerunning the notebook, but the performance is the same.

## Standout Suggestions
Application Insights is enabled for the deployed model.  It's original purpose was for me to print and debug malformed input queries, so the eyecatch is a formatted dataframe that results from converting the received data.
```
Received matchup =

   G1  G2  Seed1  Seed2   PF1   PF2   3P%1   3P%2  TOV1  TOV2
0  30  32      1      1  17.1  16.3  0.413  0.368  11.5  12.0
```
The final classification is also logged, though we get that from the response too.  And for some reason everything is logged twice, but it didn't bother me.
