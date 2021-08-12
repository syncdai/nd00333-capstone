# March Madness Matchup Predictor

Use AzureML to train a classifier on the last decade of NCAA college basketball team stats to predict which team in a match will win.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
NCAA data was collected by the Akkio company.  It records stats from both teams playing against each other in March Madness tournaments dating all the way back to 1988, of which I only use a subset of both years and stats.  Rows are duplicated because they say it reduces bias towards higher seeded teams being ordered as Team1.

### Task
The task is to train a classifier (effectively binary) that given some key features for each NCAA team in a match, it will predict either Team1 (0) or Team2 (1) as the winner.  Only the last decade (2010 - 2019) is considered, notionally because 10-fold cross validation can very neatly consider each year as a fold.  After ignoring obvious predictors (e.g. final score) and using AutoML featurization on all 50+ remaining columns, I pruned the data to the top 10 features, or top 5 stats from each team.  I was thinking ahead to how I'd have to build a query to the model later, and definitely not going to look up 50+ stats, but it also did coincidentally improve AutoML accuracy to use a compact dataset.

![top10_features](https://user-images.githubusercontent.com/80217508/129225062-f8cb58a9-67a7-4b98-8780-0f74c6844016.png)

### Access
Data is downloaded as CSV from external URL: https://s3.amazonaws.com/static.akk.io/Akkio-NCAAHistoricalData.csv

## Automated ML
Classification with 10-fold cross validation:
- classifier because the problem looked like binary classification to me
- 10-fold because it lines up neatly with 10 years of historical data

Accuracy as the primary metric, because I mainly care about predicting the correct outcome, and the data was structured to be balanced towards Team1 vs Team2.

Timeout and concurrency were just a balancing act of using my lab session resources effectively.

### Results
For once, VotingEnsemble was merely tied for best accuracy against logistic regression at 0.85952, in which case AzureML chooses the simpler model to report as best.  We can see in the screenshot below that it was SKLearn LogisticRegression, with C = 1.2, and I was surprised it used multinomial classes.  I think, but I'm not sure, that cross validation in AzureML randomizes the data, so I might be able to improve accuracy if I setup custom folds using cv_split_column_names after encoding each year into separate cv columns.

![automl_RunDetails](https://user-images.githubusercontent.com/80217508/129227608-c87114ea-e0b1-48b0-9b1f-5f874eb7b744.png)
![automl_BestModel](https://user-images.githubusercontent.com/80217508/129227634-0b41019e-c5bd-4e26-ad77-fb615c20d370.png)


## Hyperparameter Tuning
I chose to target my hyperparameter search using lessons learned from AutoML.  Namely, SKLearn LogisticRegression with what I thought were more promising parameters.  I took note of parameters from runner up logistic regression models to observe C = {1.7575, 5.4286, 0.12648, 2.5595}, penalty = {L1, L2}, and decided to avoid multinomial classification altogether (by substituting Team1, Team2 for 0, 1).
- C : 1.5 - 5.5 continuous
- max_iter : 500, 1000, 2000 -- C was on the higher end, so I figured to try more iterations to give time to converge
- penalty : L1, L2
- solver : liblinear, saga -- I picked all the ones that supported both L1 and L2, which apparently were just these 2

### Results
Accuracy was computed as the mean of scores from 10-folds, and highest achieved was 0.86746, which was better than AutoML.  Parameters were C = 5.1, max_iter = 2000, penalty = L1, solver = liblinear.

![hyper_RunDetails](https://user-images.githubusercontent.com/80217508/129236018-b2c0ba34-8ad6-4687-9f2f-ae41e9a64cf6.png)
![hyper_BestModel](https://user-images.githubusercontent.com/80217508/129236026-38408efb-d651-49cc-95ea-ffbd0afffaec.png)

I also thought maybe there was a side effect of Bayesian sampling, where when I compared the top25 vs bottom25 models sorted by accuracy there seemed to be a preference towards specific parameters: C > 4.0, max_iter > 500, L1, liblinear.  It's worth noting that even the bottom25 were better than my best AutoML model.

![hyper_compare25](https://user-images.githubusercontent.com/80217508/129235253-bbfa1cad-1b3d-4f56-9a41-36ae32a39e11.png)

With this in mind, if I were to try to improve on the model, I would try to draw conclusions based on my results and refine a different search.  For example, fix L1 liblinear and try Random sampling with a very tight Bandit policy on new ranges of C, maybe this time below 1.0 to account for that 0.12648 finding from earlier.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
